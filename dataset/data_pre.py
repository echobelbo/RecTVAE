import numpy as np
import os
from transformers import BertTokenizer, BertModel
import torch

class DataProcesser():
    def __init__(self, data_path, dataset_name):
        self.data_path = data_path
        self.dataset_name = dataset_name

    def inter_dataset(self, task):
        if task != 'text':
            # data_path = self.data_path + '/' + 'data_raw'+ '/' + self.dataset_name + '/' + self.dataset_name + '.' + task + '.inter'
            data_path = os.path.join(self.data_path, 'data_raw', self.dataset_name, self.dataset_name + '.' + task + '.inter')
            with open(data_path) as f:
                dataset = np.genfromtxt(f, delimiter='\t', dtype=None, names=True)
                data_proc = np.column_stack((dataset['user_idtoken'], dataset['item_idtoken']))
                save_path = os.path.join(self.data_path , 'data' , self.dataset_name)
                if not os.path.exists(save_path) :
                    os.mkdir(save_path)
                np.savetxt(save_path + '/' + task + '.txt', data_proc, fmt='%d', delimiter='\t')
        else:
            data_path = os.path.join(self.data_path, 'data_raw', self.dataset_name, self.dataset_name + '.text')
            map_path = os.path.join(self.data_path, 'data_raw', self.dataset_name, self.dataset_name + '.item2index')
            with open(data_path) as data_file:
                with open(map_path) as map_file:
                    text = np.genfromtxt(data_file, delimiter='\t', dtype=None, names=True)
                    item2index = np.genfromtxt(map_file, delimiter='\t', dtype=str)
                    # item_id = item2index[:, 0]
                    text_item_id = text['item_idtoken']
                    # text_item_id = np.array([item2index['item_idtoken'].index(i) for i in text_item_id])
                    # indices = np.where(item2index[:,0] == text_item_id)
                    # text_item_id = np.where(indices[1] >= 0, item2index[indices[0], 1], text_item_id)
                    # text_item_id = np.where(np.isin(text_item_id, item2index[:, 0]), item2index[:, 1], text_item_id)
                    item2index_dict = dict(zip(item2index[:, 0], item2index[:, 1]))
                    text_item_id = np.array([item2index_dict[i] for i in text_item_id])
                    text = np.column_stack((text_item_id, text['texttoken_seq']))
                    save_path = os.path.join(self.data_path , 'data' , self.dataset_name)
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                    np.savetxt(save_path + '/' + task + '.txt', text, fmt='%s', delimiter='\t')
    
    def text2feat(self, model_name, load=True, save=True):
        if load:
            feat_path = os.path.join(self.data_path, 'data', self.dataset_name, 'text_feat.npy')
            if os.path.exists(feat_path):
                return np.load(feat_path)
        text_path = os.path.join(self.data_path, 'data', self.dataset_name, 'text.txt')
        text = np.genfromtxt(text_path, delimiter='\t', dtype=str)
        text = text[:, 1]
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name, mirror='ustc')
        feat = []
        for i in range(len(text)):
            if(i % 1000 == 0):
                print(i/len(text)*100+"finished"+"\n")
            input = tokenizer.encode_plus(text[i]
                                          , add_special_tokens=True
                                          , max_length=512
                                          , truncation=True,
                                          return_tensors='pt'
                                          )
            with torch.no_grad():
                outputs = model(**input)
                features = outputs.last_hidden_state.sum(dim=1).squeeze().numpy()
            feat.append(features)
        if save:
            feat = np.array(feat)
            np.save(os.path.join(self.data_path, 'data', self.dataset_name, 'text_feat.npy'), feat)
        return feat
            


if __name__ == '__main__':
    data_path = '/root/autodl-tmp/yankai/RecTVAE'
    dataset_name = 'CDs'
    # task = 'text'
    data_processer = DataProcesser(data_path, dataset_name)
    # dataset = data_processer.inter_dataset(task)
    text2feat = data_processer.text2feat('bert-base-uncased')
