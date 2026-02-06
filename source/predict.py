import os
import pickle

import torch
from transformers import BertTokenizer

from config import Config


def get_entities_result(query):
    """
    进一步封装识别结果，最终结果格式如下:
    [
      {'type': 'address_value', 'value': '江苏南京', 'begin': 3, 'end': 7},
      {'type': 'first_name', 'value': '张', 'begin': 8, 'end': 9},
      {'type': 'last_name', 'value': '三', 'begin': 9, 'end': 10}
    ]
    :param query: 查询问句
    :return:
    """
    path = "../../output/20210811104042"
    sentence_list, predict_labels = predict(query, path)

    if len(predict_labels) == 0:
        print("句子: {0}\t实体识别结果为空".format(query))
        return []

    entities = []
    if len(sentence_list) == len(predict_labels):
        result = _bio_data_handler(sentence_list, predict_labels)
        if len(result) != 0:
            end = 0
            prefix_len = 0

            for word, label in result:
                sen = query.lower()[end:]
                begin = sen.find(word) + prefix_len
                end = begin + len(word)
                prefix_len = end
                if begin != -1:
                    ent = dict(value=query[begin:end], type=label, begin=begin, end=end)
                    entities.append(ent)
    return entities


def predict(sentence, model_path):
    """
    模型预测
    :param sentence:
    :param model_path:
    :return:
    """
    config = Config()
    max_seq_length = config.max_seq_length
    if len(sentence) > max_seq_length:
        return list(sentence), []

    with open(os.path.join(model_path, "label2id.pkl"), "rb") as f:
        label2id = pickle.load(f)
    id2label = {value: key for key, value in label2id.items()}

    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=config.do_lower_case)
    tokens = [item for sublist in [tokenizer.tokenize(s) for s in list(sentence)] for item in sublist]

    # 当句子长度大于自定义的最大句子长度时，删除多余的字符
    if len(tokens) >= max_seq_length - 1:
        # -2的原因是因为序列需要加一个句首和句尾标志
        tokens = tokens[0:(max_seq_length - 2)]

    # 获取句子的input_ids、token_type_ids、attention_mask
    result = tokenizer.encode_plus(tokens, add_special_tokens=True, max_length=max_seq_length,
                                   padding="max_length")
    input_ids, token_type_ids, attention_mask = result["input_ids"], result["token_type_ids"], result["attention_mask"]

    assert len(input_ids) == max_seq_length
    assert len(token_type_ids) == max_seq_length
    assert len(attention_mask) == max_seq_length

    input_ids = torch.tensor([input_ids], dtype=torch.long)
    token_type_ids = torch.tensor([token_type_ids], dtype=torch.long)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long)

    # 加载模型
    model = torch.load(os.path.join(model_path, "ner_model.ckpt"), map_location="cpu")
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    # 模型预测，不需要反向传播
    with torch.no_grad():
        predict_val = model.predict(input_ids, token_type_ids, attention_mask)

    predict_labels = []
    for i, label in enumerate(predict_val[0]):
        if i != 0 and i != len(predict_val[0]) - 1:
            predict_labels.append(id2label[label])

    return list(sentence), predict_labels


def _bio_data_handler(sentence, predict_label):
    """
    处理BIO开头的标签信息
    输入：sentence=['张', '三', '的', '老', '婆', '是', '谁', '？'], predict_label=['B-FNAME', 'B-LNAME', 'O', 'O', 'O', 'O', 'O', 'O']
    输出：entities=[['张', 'first_name'], ['三', 'last_name']]
    :param sentence:查询问句数组
    :param predict_label:模型预测的结果
    :return:实体结果
    """
    entities = []
    # 获取初始位置实体标签
    pre_label = predict_label[0]
    # 实体词初始化
    word = ""
    for i in range(len(sentence)):
        # 记录问句当前位置词的实体标签
        current_label = predict_label[i]
        # 若当前位置的实体标签是以B开头的，说明当前位置是实体开始位置
        if current_label.startswith('B'):
            # 当前位置所属标签类别与前一位置所属标签类别不相同且实体词不为空，则说明开始记录新实体，前面的实体需要加到实体结果中
            if pre_label[2:] is not current_label[2:] and word != "":
                entities.append([word, pre_label[2:]])
                # 将当前实体词清空
                word = ""
            # 记录当前位置标签为前一位置标签
            pre_label = current_label
            # 并将当前的词加入到实体词中
            word += sentence[i]
        # 若当前位置的实体标签是以I开头的，说明当前位置是实体中间位置，将当前词加入到实体词中
        elif current_label.startswith('I'):
            word += sentence[i]
            pre_label = current_label
        # 若当前位置的实体标签是以O开头的，说明当前位置不是实体，需要将实体词加入到实体结果中
        elif current_label.startswith('O'):
            # 当前位置所属标签类别与前一位置所属标签类别不相同且实体词不为空，则说明开始记录新实体，前面的实体需要加到实体结果中
            if pre_label[2:] is not current_label[2:] and word != "":
                entities.append([word, pre_label[2:]])
            # 记录当前位置标签为前一位置标签
            pre_label = current_label
            # 并将当前的词加入到实体词中
            word = ""
    # 收尾工作，遍历问句完成后，若实体刚好处于最末位置，将剩余的实体词加入到实体结果中
    if word != "":
        entities.append([word, pre_label[2:]])
    return entities


if __name__ == '__main__':
    sent = "江苏南京的张三的老爸是哪个"
    get_entities_result(sent)
