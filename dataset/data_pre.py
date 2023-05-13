import numpy as np
import os

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
                    item2index = np.genfromtxt(map_file, delimiter='\t', dtype=None)
                    # item_id = item2index[:, 0]
                    text_item_id = text['item_idtoken']
                    # text_item_id = np.array([item2index['item_idtoken'].index(i) for i in text_item_id])
                    # indices = np.where(item2index[:,0] == text_item_id)
                    # text_item_id = np.where(indices[1] >= 0, item2index[indices[0], 1], text_item_id)
                    text_item_id = np.array([item2index['f0'].index(i) if i in item2index['f0'] else i for i in text_item_id])
                    text = np.column_stack((text_item_id, text['texttoken_seq']))
                    save_path = os.path.join(self.data_path , 'data' , self.dataset_name)
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                    np.savetxt(save_path + '/' + task + '.txt', text, fmt='%s', delimiter='\t')


if __name__ == '__main__':
    data_path = '/root/autodl-tmp/yankai/RecTVAE'
    dataset_name = 'CDs'
    task = 'text'
    data_processer = DataProcesser(data_path, dataset_name)
    dataset = data_processer.inter_dataset(task)