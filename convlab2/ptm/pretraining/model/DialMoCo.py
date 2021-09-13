import os
import torch
import torch.nn as nn

from .modeling_dialog_bert import DialogBertForPretraining

class DialMoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, model_args, data_args, training_args, config, metadataloader, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(DialMoCo, self).__init__()

        self.K = model_args.moco_K
        self.m = model_args.moco_m
        self.T = model_args.moco_T
        self.dim = config.hidden_size

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = self._build_encoder(model_args, training_args, data_args, config, metadataloader)
        self.encoder_k = self._build_encoder(model_args, training_args, data_args, config, metadataloader)

        # if mlp:  # hack: brute-force replacement
        #     dim_mlp = self.encoder_q.fc.weight.shape[1]
        #     self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
        #     self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(self.dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # for gradient acc.
        self.k_buffer = []

        # topk acc
        self.topk = [1, 5, 10, 100, 1000]

    def _build_encoder(self, model_args, training_args, data_args, config, metadataloader):
        if model_args.model_name_or_path:
            model = DialogBertForPretraining.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                training_args=training_args,
                model_args=model_args,
                data_args=data_args,
                metadataloader=metadataloader
            )
        else:
            model = DialogBertForPretraining(config, training_args, model_args, data_args, metadataloader)

        return model

    def resize_token_embeddings(self, new_len):
        self.encoder_q.resize_token_embeddings(new_len)
        self.encoder_k.resize_token_embeddings(new_len)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def save(self, output_dir):
        self.encoder_q.save_pretrained(output_dir)
        moco_output_dir = "{}-moco".format(output_dir)
        os.makedirs(moco_output_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    def forward(self, task, dataset, dial_q, dial_k, evaluate=False, update_k_encoder=False, update_queue=False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        assert task == "moco"

        outputs = {}

        # compute query features
        q = self.encoder_q(task=task, dataset=dataset, **dial_q)["cls_output"]  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            if update_k_encoder:
                # for gradient acc. at the beginning of each batch, update the key encoder
                self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            # im_k, idx_unshuffle = self._batch_shuffle_ddp(**dial_k)

            k = self.encoder_k(task=task, dataset=dataset, **dial_k)["cls_output"]  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        outputs["moco_logits"] = logits

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        criterion = nn.CrossEntropyLoss()

        moco_loss = criterion(logits, labels)

        outputs["moco_loss"] = moco_loss

        outputs["loss"] = outputs.get("moco_loss", 0)

        acc = accuracy(logits, labels, topk=self.topk)

        outputs["acc"] = acc

        # dequeue and enqueue
        self.k_buffer.append(k)
        if update_queue:
            # for gradient acc. at the end of each batch, update the queue
            for kk in self.k_buffer:
                self._dequeue_and_enqueue(kk)
            self.k_buffer = []

        return outputs


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = {}
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res[k] = correct_k.mul_(100.0 / batch_size)
        return {k:v.item() for k, v in res.items()}
