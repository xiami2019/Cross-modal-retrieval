#Author by Cqy2019

import re
import os
import torch
import numpy as np
import random

from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pytorch_transformers import BertTokenizer, BertModel

class IPAR_TC_12(Dataset):
    '''
    IPAR_TC_12 Dataset for cross-modal retrieval
    '''
    def __init__(self, root, if_train, lang, sample_index=None):
        '''
        root: data root
        if_train: to identify train set or test set
        '''
        if if_train == False:
            assert sample_index is not None
            self.sample_index = sample_index
        else:
            self.sample_index = None

        self.root = os.path.join(root, 'iaprtc12')
        self.root_image = self.root
        self.root_label = os.path.join(root, 'saiaprtc12ok', 'benchmark', 'saiapr_tc-12')

        assert lang == 'en' or lang == 'de'
        if lang == 'en':
            self.root = os.path.join(self.root, 'annotations_complete_eng')
        if lang == 'de':
            self.root = os.path.join(self.root, 'annotations_complete_ger')
        
        self.if_train = if_train
        self.lang = lang
        self.all_images = []
        self.all_texts = []
        self.all_labels = []
        self.sampled_images = []
        self.sampled_texts = []
        self.sampled_labels = []
        self.image2label = {}
        self.diff_label = {}
        self.true_label = {}

        if self.if_train is True:
            self.transforms = transforms.Compose([
                            transforms.Resize(256),
                            transforms.RandomCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
        else:
            self.transforms = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])

        self.load_labels()
        self.load_data()

        self.sample_dataset()

    def load_labels(self):
        for root_, dirs, files in os.walk(self.root_label):
            file_path = os.path.join(root_, 'labels.txt')
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f.readlines():
                        line = line.strip('\n').split()
                        if len(line) < 3:
                            continue
                        if line[0] not in self.image2label:
                            self.image2label[line[0]] = []
                        self.image2label[line[0]].append(int(line[2]))
                        self.diff_label[line[2]] = 1

        #build true label
        for i in self.diff_label.keys():
            self.true_label[i] = len(self.true_label)

        #correctness check                
        assert len(self.diff_label) == len(self.true_label) == 255 and len(self.image2label) == 20000

        #retrict labels in range 0~254 and change to one hot
        for i in self.image2label.keys():
            for j in range(len(self.image2label[i])):
                self.image2label[i][j] = self.true_label[str(self.image2label[i][j])]
            temp_str = np.zeros(255)
            for j in self.image2label[i]:
                temp_str[j] = 1
            self.image2label[i] = temp_str

    def load_data(self):
        for root_, dirs, files in os.walk(self.root):

            for name in files:
                file_path = os.path.join(root_, name)
                img_id = name.split('.')[0]
                if img_id not in self.image2label:
                    continue
                self.all_labels.append(self.image2label[img_id])

                with open(file_path, 'r', encoding='cp1252') as f:
                    for line in f.readlines():
                        line = line.strip('\n')
                        line = re.split('<|>', line)
                        if len(line) < 3:
                            continue
                        if line[1] == 'DESCRIPTION':
                            self.all_texts.append(line[2])
                        elif line[1] == 'IMAGE':
                            assert img_id in line[2]
                            self.all_images.append(line[2])

        #correctness check  
        assert len(self.all_labels) == len(self.all_images) == len(self.all_texts) == 19999
        
    def sample_dataset(self):

        if self.if_train:
            index = [i for i in range(len(self.all_images))]
            random.shuffle(index)
            index = index[:10000]
            self.sampled_images = [self.all_images[item] for item in index]
            self.sampled_labels = [self.all_labels[item] for item in index]
            self.sampled_texts = [self.all_texts[item] for item in index]
            assert len(self.sampled_images) == len(self.sampled_labels) == len(self.sampled_texts) == 10000
        else:
            self.sampled_images = [self.all_images[item] for item in self.sample_index]
            self.sampled_labels = [self.all_labels[item] for item in self.sample_index]
            self.sampled_texts = [self.all_texts[item] for item in self.sample_index]
            assert len(self.sampled_images) == len(self.sampled_labels) == len(self.sampled_texts)

    def __len__(self):

        return len(self.sampled_images)

    def __getitem__(self, idx):

        image_name = self.sampled_images[idx]
        label = self.sampled_labels[idx]
        text = self.sampled_texts[idx]
        text = '[CLS] ' + text + ' [SEP]'
        image = Image.open(os.path.join(self.root_image, image_name)).convert('RGB')
        image = self.transforms(image)
        return image, text, label
        # return image

class Iteration():
    '''
    dataloader for IPAR_TC_12
    '''
    def __init__(self, params, dataset):
        self.params = params
        self.dataloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=False, num_workers=8)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        
        return len(self.dataloader)

    def __iter__(self):
        
        for index ,(images, texts, labels) in enumerate(self.dataloader):
            tokens = []
            segment = []
            input_mask = []
            for i in texts:
                tokenized_text = self.tokenizer.tokenize(i)
                indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
                tokens.append(indexed_tokens)
                segment.append([0] * len(indexed_tokens))
                input_mask.append([1] * len(indexed_tokens))      

            max_len = max([len(single) for single in tokens])

            #get padding and mask
            for j in range(len(tokens)):
                padding = [0] * (max_len - len(tokens[j]))
                tokens[j] += padding
                input_mask[j] += padding
                segment[j] += padding

            tokens = torch.tensor(tokens)
            segment = torch.tensor(segment)
            input_mask = torch.tensor(input_mask)

            yield images.cuda(), tokens.cuda(), segment.cuda(), input_mask.cuda(), labels

if __name__ == '__main__':
    dataset = IPAR_TC_12('/home/disk1/chengqinyuan/datasets', True, 'en')
    print('creating iterator...')
    dataloader = Iteration(100, dataset)
    model = BertModel.from_pretrained('bert-base-uncased')
    model.cuda()
    model.eval()
    for images, tokens, segment, input_mask, labels in dataloader:
        # print(input_mask)
        output = model(tokens, token_type_ids=segment, attention_mask=input_mask)
        feature = output[0][:,0,:]
        print(feature.size())
        break
        # print(tokens.size())
        # print(input_mask.size())