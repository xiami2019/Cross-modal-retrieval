#Author by Cqy2019
import torch
import torch.nn as nn
from torchvision import models
from pytorch_transformers import BertModel, BertConfig

class Image_Model(nn.Module):
    '''
    CNN for image retrieval
    '''
    def __init__(self, params, fine_tune_flag=True):
        super(Image_Model, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(in_features=512, out_features=params.code_size)
        self.tanh = nn.Tanh()
        for p in self.resnet.parameters():
            p.requires_grad = True

    def forward(self, image):
        out = self.resnet(image)
        out = self.tanh(out)

        return out

class Text_Model(nn.Module):
    '''
    Bert model for extract text feature.
    '''
    def __init__(self, params, fine_tune_flag=True):
        super(Text_Model, self).__init__()
        config = BertConfig.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained(params.bertmodel_dir, config=config)
        # self.fc1 = nn.Linear(in_features=768, out_features=768)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(in_features=768, out_features=params.code_size)
        self.tanh = nn.Tanh()
        # self.relu = nn.ReLU()
        for p in self.model.parameters():
            p.requires_grad = True
        # for p in self.fc1.parameters():
        #     p.requires_grad = True
        for p in self.fc2.parameters():
            p.requires_grad = True

    def forward(self, input_tokens_ids, segment_ids, input_mask):
        output = self.model(input_tokens_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        feature = output[0][:,0,:]
        # feature = self.fc1(feature)
        # feature = self.relu(feature)
        feature = self.dropout(feature)
        feature = self.fc2(feature)
        output = self.tanh(feature)

        return output

if __name__ == '__main__':
    print('test')