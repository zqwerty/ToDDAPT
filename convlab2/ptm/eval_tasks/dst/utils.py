import copy
import os

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix


def cal_scores(preds, labels, only_acc=False):
    if only_acc:
        return {
            "acc": accuracy_score(preds, labels),
        }
    else:
        return {
            "acc": accuracy_score(preds, labels),
            "f1": (f1_score(preds, labels, average="macro"), f1_score(preds, labels, average=None)),
            "recall": (recall_score(preds, labels, average="macro"), recall_score(preds, labels, average=None)),
            "prec": (precision_score(preds, labels, average="macro"), precision_score(preds, labels, average=None)),
            "som-f1": cal_f1(preds, labels)
        }

def cal_matrix(preds, labels):
    return confusion_matrix(labels, preds)


def add_states(labels, upd_states, tokenizer, all_values, id2slot, ontology, split=False, prev_truth=False, upd_truth=False, value_truth=False):
    # for every dialog
    prev_states = [["" for _ in range(len(id2slot))]] + copy.deepcopy(labels[:-1])
    history_states = ["" for _ in range(len(id2slot))]
    all_states = []
    all_upd_res = []
    all_upd_err = []
    for label, prev_state, upd_state in zip(labels, prev_states, upd_states):
        cur_state = prev_state.copy() if prev_truth else history_states.copy()
        joint_upd = True
        joint_upd_err = []
        for sid, slot_pred in enumerate(upd_state):
            # if slot_pred["update_truth"] == 1:
            #     cur_state[sid] = "dontcare"
            #     continue
            # if slot_pred["update_truth"] == 2:
            #     cur_state[sid] = ""
            #     continue
            if slot_pred["update_truth"] != slot_pred["update"]:
                joint_upd = False
                joint_upd_err.append((slot_pred["update_truth"], slot_pred["update"]))

            upd = slot_pred["update_truth"] if upd_truth else slot_pred["update"]
            if upd == 0:
                pass
            elif upd == 1:
                cur_state[sid] = "dontcare"
            elif upd == 2:
                cur_state[sid] = ""
            else:
                assert upd == 3
                domain_name, slot_name = id2slot[sid]
                if value_truth:
                    if upd == slot_pred["update_truth"]:
                        # cur_state[sid] = label[sid]
                        # continue
                        pass
                    else:
                        cur_state[sid] = "[wrong!!@#@]"
                        continue
                if ontology['domains'][domain_name]['slots'][slot_name]["is_categorical"]:
                    # value = slot_pred["value"]
                    value = slot_pred["value_label"] if value_truth else slot_pred["value"]
                    # assert slot_pred["value_mask"][value] == 1, "{}, {}".format(slot_pred["value_mask"], value)
                    if not split:
                        value_id = sum(slot_pred["value_mask"][:value])
                    # for split setting, value is just value id
                    if value < 0:
                        cur_state[sid] = "[wrong!!@#@]"
                    else:
                        if split:
                            cur_state[sid] = all_values[sid][value]
                        else:
                            cur_state[sid] = all_values[sid][value_id]
                else:
                    # start = slot_pred["start"]
                    start = slot_pred["start_label"] if value_truth else slot_pred["start"]
                    # end = slot_pred["end"]
                    end = slot_pred["end_label"] if value_truth else slot_pred["end"]
                    if start < 0 or end < 0:
                        cur_state[sid] = "[wrong!!@#@]"
                    else:
                        tokens = slot_pred["inputs"][start: end]
                        cur_state[sid] = "".join(tokenizer.decode(tokens).strip().split(" "))

        all_states.append(cur_state)
        all_upd_res.append(joint_upd)
        all_upd_err.append(joint_upd_err)
        if not prev_truth:
            history_states = cur_state

    return all_states, all_upd_res, all_upd_err


def cal_error(all_outputs, all_labels, tokenizer, all_values, id2slot, ontology, out_dir, split=False):
    output_path = os.path.join(out_dir, "errors.txt")
    f = open(output_path, "w")
    for (did, outputs_d), labels_d in zip(enumerate(all_outputs), all_labels):
        f.write("dialog_id #{}\n".format(did))
        for (uid, outputs_u), labels_u in zip(enumerate(outputs_d), labels_d):
            for (sid, outputs), label in zip(enumerate(outputs_u), labels_u):
                upd = outputs["update"]
                value = ""
                domain_name, slot_name = id2slot[sid]
                if upd == 0:
                    if upd != outputs["update_truth"]:
                        f.write("utt_id: {}, domain: {}, slot: {}, upd: {}, upd_true: {}\n\n".format(uid, domain_name, slot_name, upd, outputs["update_truth"]))
                else:
                    if upd == 1:
                        value = "dontcare"
                    elif upd == 2:
                        value = ""
                    else:
                        if ontology['domains'][domain_name]['slots'][slot_name]["is_categorical"]:
                            # value = slot_pred["value"]
                            value = outputs["value"]
                            # assert slot_pred["value_mask"][value] == 1, "{}, {}".format(slot_pred["value_mask"], value)
                            if split:
                                value = all_values[sid][value]
                            else:
                                value_id = sum(outputs["value_mask"][:value])
                                value = all_values[sid][value_id]
                        else:
                            # start = slot_pred["start"]
                            start = outputs["start"]
                            # end = slot_pred["end"]
                            end = outputs["end"]
                            tokens = outputs["inputs"][start: end]
                            value = "".join(tokenizer.decode(tokens).strip().split(" "))

                    if value != label:
                        f.write("utt_id: {}, domain: {}, slot: {}, pred: {}, label: {}\n".format(uid, domain_name, slot_name, value, label))
                        f.write("upd: {}, upd_true: {}\n".format(upd, outputs["update_truth"]))
                        if upd == 3:
                            f.write("start: {}, start_true: {}; end: {}, end_true: {}; value: {}, value_true: {}\n".format(
                                outputs["start"],
                                outputs["start_label"],
                                outputs["end"],
                                outputs["end_label"],
                                outputs["value"],
                                outputs["value_label"]
                            ))
                        f.write("\n")
        f.write("\n\n\n")
    
    f.close()
    

def cal_added_scores(all_preds, all_labels, all_upd_res, slot_num):
    all_preds = [p for x in all_preds for p in x]
    all_labels = [p for x in all_labels for p in x]
    all_upd_res = [p for x in all_upd_res for p in x]
    slot_acc, joint_acc, slot_acc_total, joint_acc_total = 0, 0, 0, 0
    # print(len(all_labels))
    fail_num = 0
    for labels, preds, upd in zip(all_labels, all_preds, all_upd_res):
        joint = 0
        for slot_id in range(slot_num):
            pred = preds[slot_id]
            label = labels[slot_id]
            if pred == label:
                slot_acc += 1
                joint += 1
            elif "|" in label:
                if pred in label.split("|"):
                    slot_acc += 1
                    joint += 1
            slot_acc_total += 1

        if joint == slot_num:
            joint_acc += 1
            # assert upd is True, "{}, {}".format(preds, labels)
        else:
            fail_num += 1
            # print("upd: ", upd)
            # print(preds)
            # print(labels)
            # print(joint_acc_total)
        joint_acc_total += 1

    joint_acc = joint_acc / joint_acc_total
    slot_acc = slot_acc / slot_acc_total

    # print("fail_num: ", fail_num)
    # print("total turn: ", joint_acc_total)

    return joint_acc, slot_acc

def cal_f1(pred, gold):
    id2op = ["carry over", "dontcare", "clear", "value"]
    op2id = {v: i for i, v in enumerate(id2op)}
    tp_dic = [0 for _ in op2id]
    fn_dic = [0 for _ in op2id]
    fp_dic = [0 for _ in op2id]
    all_op_F1_count = [0 for _ in op2id]
    op_F1_count = [0 for _ in op2id]
    for p, g in zip(pred, gold):
        all_op_F1_count[g] += 1
        if p == g:
            tp_dic[g] += 1
            op_F1_count[g] += 1
        else:
            fn_dic[g] += 1
            fp_dic[p] += 1

    op_F1_score = {}
    for op in id2op:
        k = op2id[op]
        tp = tp_dic[k]
        fn = fn_dic[k]
        fp = fp_dic[k]
        precision = tp / (tp+fp) if (tp+fp) != 0 else 0
        recall = tp / (tp+fn) if (tp+fn) != 0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        op_F1_score[op] = F1
    
    return op_F1_score
