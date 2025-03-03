# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
# from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss


class CodeBERTClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)    # config.num_labels=3,  shape => [batch_size, 3]

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take [CLS] token, 分类
        x = x.reshape(-1,x.size(-1))
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x



class Model(nn.Module):
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier = CodeBERTClassificationHead(config)
        self.args=args
    
    def forward(self, input_ids=None,labels=None, private_labels=None, reduction_labels=None):
        # 获取编码器输出
        attention_mask = input_ids.ne(1)
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state  # last_hidden_state

        # 分类头处理
        logits = self.classifier(pooled_output)  # [batch_size, 3]
        logits = torch.sigmoid(logits)  # 将logits通过sigmoid函数，转为概率值

        if labels is not None and private_labels is not None and reduction_labels is not None:
            # 创建多标签目标张量
            target = torch.stack((labels.float(), private_labels.float(), reduction_labels.float()), dim=1)

            # 使用 BCELoss 计算多标签损失
            loss_fct = BCELoss()
            loss = loss_fct(logits, target)
            return loss, logits
        else:
            return logits
        


# class Model(nn.Module):   
#     def __init__(self, encoder,config,tokenizer,args):
#         super(Model, self).__init__()
#         self.encoder = encoder
#         self.config=config
#         self.tokenizer=tokenizer
#         self.args=args
    
        
#     def forward(self, input_ids=None,labels=None): 
#         logits=self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]
#         prob=torch.softmax(logits,-1)
#         if labels is not None:
#             loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
#             loss = loss_fct(logits,labels)
#             return loss,prob
#         else:
#             return prob
      
        
 
