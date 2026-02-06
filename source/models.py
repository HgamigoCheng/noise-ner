# -*- coding: utf-8 -*-
# @description:
# @author: zchen
# @time: 2020/11/30 18:23
# @file: models.py.py
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertPreTrainedModel, BertModel


class BERT_BiLSTM_CRF(BertPreTrainedModel):

    def __init__(self, config, need_birnn=False, rnn_dim=128):
        super(BERT_BiLSTM_CRF, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        out_dim = config.hidden_size

            
        self.softmax = nn.Softmax(dim=-1)

        self.hidden2tag = nn.Linear(in_features=out_dim, out_features=config.num_labels)

        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        self.bert_layers = list(self.bert.encoder.layer.children())


    def forward(self, input_ids, tags, token_type_ids=None, attention_mask=None):
        """
        BERT_BiLSTM_CRF模型的正向传播函数

        :param input_ids:      torch.Size([batch_size,seq_len]), 代表输入实例的tensor张量
        :param token_type_ids: torch.Size([batch_size,seq_len]), 一个实例可以含有两个句子,相当于标记
        :param attention_mask:     torch.Size([batch_size,seq_len]), 指定对哪些词进行self-Attention操作
        :param tags:
        :return:
        """
        if attention_mask is not None:
            attention_mask = attention_mask.byte()
        else:
            attention_mask = torch.ones_like(tags, dtype=torch.uint8)  # 使用全1的掩码
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        
        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)
        # 处理可能的 NoneType
        
        loss = -1 * self.crf(emissions, tags, mask=attention_mask.byte())
        logits = emissions[:, :,-1] 
        return loss,logits

    def predict(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        模型预测
        :param input_ids:
        :param token_type_ids:
        :param attention_mask:
        :return:
        """
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
    
        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)
        return self.crf.decode(emissions, attention_mask.byte())
