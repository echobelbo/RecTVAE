import numpy as np
import os
from transformers import BertTokenizer, BertModel
from scipy.sparse import coo_matrix, save_npz, load_npz
import torch
import json
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
                user_ids = dataset['user_idtoken'].astype(int)
                item_ids = dataset['item_idtoken'].astype(int)
                print(user_ids.dtype)
                print(item_ids.dtype)
                print(user_ids.shape)
                print(item_ids.shape)
                sparse_matrix = coo_matrix((np.ones(len(dataset['item_idtoken'])), (user_ids, item_ids)))
                save_path = os.path.join(self.data_path , 'data' , self.dataset_name)
                if not os.path.exists(save_path) :
                    os.mkdir(save_path)
                # np.savetxt(save_path + '/' + task + '.txt', data_proc, fmt='%d', delimiter='\t')
                save_npz(os.path.join(save_path, task + '.npz'), sparse_matrix)
        else:
            data_path = os.path.join(self.data_path, 'data_raw', self.dataset_name, self.dataset_name + '.text')
            map_path = os.path.join(self.data_path, 'data_raw', self.dataset_name, self.dataset_name + '.item2index')
            with open(data_path) as data_file:
                with open(map_path) as map_file:
                    text = np.genfromtxt(data_file, delimiter='\t', dtype=None, names=True)
                    item2index = np.genfromtxt(map_file, delimiter='\t', dtype=str)
                    # item_id = item2index[:, 0]
                    text_item_id = text['item_idtoken']
                    item2index_dict = dict(zip(item2index[:, 0], item2index[:, 1]))
                    text_item_id = np.array([item2index_dict[i] for i in text_item_id])
                    text = np.column_stack((text_item_id, text['texttoken_seq']))
                    sorted_index = np.argsort(text[:, 0].astype(int))
                    text = text[sorted_index]
                    save_path = os.path.join(self.data_path , 'data' , self.dataset_name)
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                    np.savetxt(save_path + '/' + task + '.txt', text, fmt='%s', delimiter='\t')
    
    def text2feat(self, model_name, batch_size, load=True, save=True):
        if torch.cuda.is_available():
            device = "cuda"
        if load:
            feat_path = os.path.join(self.data_path, 'data', self.dataset_name, 'text_feat.npy')
            if os.path.exists(feat_path):
                print('Loading text feature file...')
                return np.load(feat_path)
            else:
                print('No text feature file found, generating...')
        text_path = os.path.join(self.data_path, 'data', self.dataset_name, 'text.txt')
        text = np.genfromtxt(text_path, delimiter='\t', dtype=str)
        index = text[:, 0]
        text = text[:, 1]

        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name, mirror='ustc').to(device)

        feat = []
        num_samples = len(text)
        num_batches = (num_samples - 1) // batch_size + 1

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_text = text[start_idx:end_idx]
            inputs = tokenizer.batch_encode_plus(
                batch_text,
                add_special_tokens=True,
                max_length=512,
                truncation=True,
                padding='longest',
                return_tensors='pt'
            ).to(device)
        
            with torch.no_grad():
                outputs = model(**inputs)
                features = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
                # features = features / np.linalg.norm(features, axis=1, keepdims=True)

            feat.append(features)
            print('Batch {}/{} finished'.format(batch_idx + 1, num_batches))
        feat = np.concatenate(feat, axis=0)
        index_dict ={}
        for i , idx in enumerate(index):
            if idx not in index_dict:
                index_dict[idx] = feat[i]
            else:
                index_dict[idx] += feat[i]
        feat = np.array(list(index_dict.values()))

        if save:
            # feat = np.array(feat)
            np.save(os.path.join(self.data_path, 'data', self.dataset_name, 'text_feat.npy'), feat)
        return feat
    def json_pre(self):
        dir = os.path.join(self.data_path, 'data_raw')
        json_dir = os.path.join(dir, self.dataset_name + '_5.json')
        data = ""
        with open(json_dir, 'r') as f:
            data = f.read()
            data = data.replace('}\n{', '},\n{')
            data = '[' + data + ']'
        with open(json_dir, 'w') as f:
            f.write(data)

    def json2txt(self):
        dir = os.path.join(self.data_path, 'data_raw')
        json_dir = os.path.join(dir, self.dataset_name + '_5.json')
        target_dir = os.path.join(dir, self.dataset_name)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        with open(json_dir) as f:
            data = f.read()
            data = json.loads(data)
            new_data = [item for item in data if (  item.get("overall") != None and 
                                                    item.get("reviewerID") != None and 
                                                    item.get("asin") != None and 
                                                    item.get("reviewText") != None and 
                                                    item.get("verified") != None and 
                                                    item.get("overall") > 3.5 and 
                                                    item.get("verified") == True)]
            user_id = [item["reviewerID"] for item in new_data]
            item_id = [item["asin"] for item in new_data]
            text = [item["reviewText"] for item in new_data]
            

            user2index = np.unique(user_id)
            item2index = np.unique(item_id)
            user2index = np.column_stack((user2index, np.arange(len(user2index))))
            item2index = np.column_stack((item2index, np.arange(len(item2index))))
            text = np.column_stack((item_id, text))

            user2index_dict = dict(zip(user2index[:, 0], user2index[:, 1]))
            item2index_dict = dict(zip(item2index[:, 0], item2index[:, 1]))

            user_id = [user2index_dict[user] for user in user_id]
            item_id = [item2index_dict[item] for item in item_id]
            text[:, 0] = [item2index_dict[item] for item in text[:, 0]]
            text = np.unique(text, axis = 0)
            sorted_index = np.argsort(text[:, 0].astype(int))
            text = text[sorted_index]
        
            inter = np.column_stack((user_id, item_id))
            inter = np.unique(inter, axis = 0)
            sorted_index = np.argsort(inter[:, 0].astype(int))
            inter = inter[sorted_index]
            

            np.savetxt(target_dir + '/inter.txt', inter, fmt='%s', delimiter='\t', comments='')
            np.savetxt(target_dir + '/text.txt', text, fmt='%s', delimiter='\t', comments='')



    def new_text2feat():
        pass  






        

if __name__ == '__main__':
    data_path = '/root/autodl-tmp/yankai/RecTVAE'
    dataset_name = 'All_Beauty'
    # task = 'text'
    data_processer = DataProcesser(data_path, dataset_name)
    # data_processer.json_pre()
    data_processer.json2txt()
    # dataset = data_processer.inter_dataset('train')
    # dataset = data_processer.inter_dataset('valid')
    # dataset = data_processer.inter_dataset('test')
    # text2feat = data_processer.text2feat(model_name='bert-base-uncased', batch_size=32, load=False, save=True)
