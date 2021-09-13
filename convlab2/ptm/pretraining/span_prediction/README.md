## 运行

`python3 preprocess_data.py [--max_utts_to_keep] [--max_utts_to_keep_cat] [--max_utts_to_keep_noncat]
`

## 数据集
- multiwoz25
- schema
- woz
- m2m


## 数据格式


每个数据集被处理成4个数据文件，在`processed_data`文件夹里。
- `cat_stateupdate_data_train.json`
    - `input_ids`, (*List[int]*)
    - `position_ids`
    - `turn_ids`
    - `role_ids`
    - `cls_label` (*str*), one of `[has_value, none, dontcare, delete]`
    - `value_token_mask` (optional, *List[{0, 1}]*), only when `cls_label` in `[has_value]`
    - `value_label_idx` (optional, *int*), only when `cls_label` in `[has_value]`
- `cat_stateupdate_dev_train.json`
   - same as before
- `noncat_stateupdate_data_train.json`
    - `input_ids`, (*List[int]*)
    - `position_ids`
    - `turn_ids`
    - `role_ids`
    - `cls_label` (*str*), one of `[has_value, none, dontcare, delete]`
    - `start` (optional, *int*), only when `cls_label` in `[has_value]`. but not nessessarily exists.
    - `end` (optional, *int*), inclusive end token idx.
- `noncat_stateupdate_dev_train.json`
    - same as before


## 统计

各数据集train/dev数量

| dataset |  cat train example | cat dev example | noncat train example | noncat dev example |
| ---- |  ---- | ----  | ---- | ---- |
|multiwoz25 |  5049 | 62 | 106683 | 1252 |
| schema |   31883 | 351 | 196561 | 2361 |
| woz   | 2241 | 22 | 3227 | 24  |
| m2m    | 0   | 0  | 21288 | 196 |

各数据集slot delete出现次数

| dataset |  count |
| ---- |  ---- |
|multiwoz25 |  229 |
| other | 0 |


- slot delete:把label设为none
- 负采样：对于每轮对话，其中涉及的每个domain，都采样1个未涉及的slot作为none样本
- non-categorical：有些non-categorical slot没有span标注，这种只给`cls_label`不给`span label`