from dataset.data_pre import DataProcesser
import argparse
from vae.utils import metric_map, metric_recall, sort2query, csr2test, Evaluator
from vae.vae import VAE, TVAE, BPR, BERTVAE
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from scipy.sparse import coo_matrix, load_npz
from tqdm import tqdm
from torch import optim
# import profile
# from sklearn.model_selection import train_test_split
import os
import datetime
# from functools import lru_cache
import logging
import torchmetrics
from joblib import Memory
# @lru_cache(maxsize=None)
# def load_npz_cached(path):
#     return load_npz(path)
os.environ['TORCH_USE_CUDA_DSA'] = '1'
anneal = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cache = Memory('./cache', verbose=0)
class BPRDataset(Dataset):
    def __init__(self, interactions, num_users, num_items):
        self.interactions = interactions
        self.num_users = num_users
        self.num_items = num_items

    def __len__(self):
        return self.num_users

    def __getitem__(self, index):
        user = index
        positive_items = self.interactions[user].nonzero()[1]
        item_i = torch.tensor(np.random.choice(positive_items), dtype=torch.long).to(device)
        item_j = torch.tensor(np.random.choice(self.num_items), dtype=torch.long).to(device)
        return user, item_i, item_j
    

class SparseTensorDataset(Dataset):
    def __init__(self, sparse_tensor):
        self.sparse_tensor = sparse_tensor

    def __getitem__(self, index):
        return self.sparse_tensor[index]

    def __len__(self):
        return self.sparse_tensor.size(0)


def Guassian_loss(recon_x, x):
    recon_x = torch.sigmoid(recon_x)
    x_dense = x.to_dense()
    weights = x.dense * args.alpha + (1 - x_dense)
    loss = x_dense - recon_x
    loss = torch.sum(weights * (loss ** 2))
    return loss


def BCE_loss(recon_x, x):
    recon_x = torch.sigmoid(recon_x)
    eps = 1e-8
    x_dense = x.to_dense()
    loss = -torch.sum(args.alpha * torch.log(recon_x + eps) * x_dense + torch.log(1 - recon_x + eps) * (1 - x_dense))
    return loss

def Softmax_loss(recon_x, x):
    m = torch.nn.LogSoftmax(dim=1)
    # recon_x = torch.softmax(recon_x,dim=1)
    x_dense = x.to_dense()
    # recon_x = recon_x - recon_x.max(dim=1, keepdim=True)[0]

    loss = (-torch.sum(m(recon_x) * x_dense, dim=-1) / torch.sum(x_dense, dim=-1)).mean()

    return loss

def regularization(mu, logvar):
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))

# def train_batch(epoch, train_loader):
#     model.train()
#     loss_value = 0
#     for batch_idx, data in enumerate(tqdm(train_loader, unit='epoch')):
#         data = data.to(args.device)
#         optimizer.zero_grad()
#         recon_batch, mu, logvar = model(data)
#         loss = loss_function(recon_batch, data) + regularization(mu, logvar) * args.beta
#         loss.backward()
#         max_norm = 1.0
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
#         loss_value += loss.item()
#         optimizer.step()
#     logger.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch, loss_value / len(train_loader.dataset)))

def slim_train(epoch):
    model.train()
    loss_value = 0
    for batch_idx, data in enumerate(tqdm(train_loader, unit='epoch')):
        data = data.to(args.device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data)
        loss.backward()
        loss_value += loss.item()
        optimizer.step()






def train(epoch):
    model.train()
    loss_value = []        

    for batch_idx, data in enumerate(tqdm(train_loader, unit='epoch')):
        data = data.to(args.device)
        
            # 拼接特征 
        optimizer.zero_grad()
        if args.model == 'TVAE':
            recon_batch, mu, logvar = model(data, features)
        elif args.model == 'VAE':
            recon_batch, mu, logvar = model(data)

        anneal_t = min(0.2, args.anneal)
        args.anneal = min(0.2,
                          args.anneal + (1.0 / 2000000))
        loss = loss_function(recon_batch, data)  + regularization(mu, logvar) * args.beta * anneal_t
        loss.backward()       
        optimizer.step()
        # max_norm = 1.0
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        loss_value.append(loss.item())
        
    logger.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch,  sum(loss_value)/len(loss_value)))

def batch_evaluate(split='valid'):
    y_true = eval(split + '_data')
    model.eval()
    result = {
        'recall_5': 0,
        'recall_10': 0,
        'recall_15': 0,
        'recall_20': 0,
        'map_cut_5': 0,
        'map_cut_10': 0,
        'map_cut_15': 0,
        'map_cut_20': 0
    }
    cnt = 0
    for batch_idx, data in enumerate(tqdm(train_loader, unit='epoch')):
        cnt+=1
        end_idx = min((batch_idx + 1) * args.batch, train_data.shape[0])

        batch_data = data.to(args.device)
        nonzero_idx = torch.nonzero(torch.abs(batch_data.to_dense()) > 1e-12).cpu().numpy()

        if args.model == 'VAE':
            y_score, _, _ = model(batch_data)
        else:
            y_score, _, _ = model(batch_data, features)
        y_score.detach_()
        y_score = y_score.squeeze(0)
        y_score[nonzero_idx[:,0], nonzero_idx[:,1]] = 0
        _, y_pred = torch.topk(y_score, args.N, dim=1)
        run = sort2query(y_pred[:, 0:args.N])
        test = csr2test(y_true.tocsr()[batch_idx*args.batch:end_idx])
        evaluator = Evaluator({'recall', 'map_cut'})
        evaluator.evaluate(run, test)
        result_t = evaluator.show(
            ['recall_5', 'recall_10', 'recall_15', 'recall_20', 'map_cut_5', 'map_cut_10', 'map_cut_15', 'map_cut_20'])
        result['recall_5'] += result_t['recall_5']
        result['recall_10'] += result_t['recall_10']
        result['recall_15'] += result_t['recall_15']
        result['recall_20'] += result_t['recall_20']
        result['map_cut_5'] += result_t['map_cut_5']
        result['map_cut_10'] += result_t['map_cut_10']
        result['map_cut_15'] += result_t['map_cut_15']
        result['map_cut_20'] += result_t['map_cut_20']
    result['recall_5'] /= cnt
    result['recall_10'] /= cnt
    result['recall_15'] /= cnt
    result['recall_20'] /= cnt
    result['map_cut_5'] /= cnt
    result['map_cut_10'] /= cnt
    result['map_cut_15'] /= cnt
    result['map_cut_20'] /= cnt
    logger.info('result: {}'.format(result))


@torch.no_grad()
def evaluate(split='valid'):
    y_true = eval(split + '_data')
    # Ttensor = eval(split + '_tensor')
    model.eval()
    # y_score, _, _ = model(train_tensor)
    # y_score.detach_()
    # y_score = y_score.squeeze(0)
    # y_score[train_data.row, train_data.col] = 0
    # _, rec_items = torch.topk(y_score, args.N, dim=1)
    # y_pred = torch.gather(Ttensor, 1, rec_items).cpu().numpy()
    result = {
        'recall_5': 0,
        'recall_10': 0,
        'recall_15': 0,
        'recall_20': 0,
        'map_cut_5': 0,
        'map_cut_10': 0,
        'map_cut_15': 0,
        'map_cut_20': 0,
        'ndcg': 0
    }
    cnt = 0
    # recall = 0
    # map_cut = 0        
    # indices = torch.LongTensor([train_data.row, train_data.col]).to(args.device)
    # values = torch.FloatTensor(train_data.data).to(args.device)
    # size = torch.Size(train_data.shape)
    # sparse_tensor = torch.sparse_coo_tensor(indices, values, size).to(args.device)
    for start_idx in range(0, train_data.shape[0], 2048):
        cnt += 1
        
        end_idx = min(start_idx + 2048, train_data.shape[0])

        # batch_data = sparse_tensor[start_idx:end_idx].to_dense().to(args.device)
        # nonzero_indices = torch.nonzero(batch_data)
        
        batch_data = train_data.tocsr()[start_idx:end_idx].toarray().astype('float32')
        train_tensor = torch.from_numpy(batch_data).to(args.device)
        nonzero_indices = torch.nonzero(train_tensor > 1e-12, as_tuple=True)
        # batch_data = train_tensor[start_idx:end_idx].to(args.device)
        
        
        if args.model == 'VAE':
            y_score, _, _ = model(train_tensor)
        else:
            y_score, _, _ = model(train_tensor, features)
        y_score.detach_()
        y_score = y_score.squeeze(0)
        y_score[nonzero_indices] = -float('inf')
        # y_score[train_data.row[:end_idx-start_idx], np.unique(train_data.col)] = 0
        _, y_pred = torch.topk(y_score, args.N, dim=1)
        run = sort2query(y_pred[:, 0:args.N])
        test = csr2test(y_true.tocsr()[start_idx:end_idx])
        evaluator = Evaluator({'recall', 'map_cut', 'ndcg'})
        evaluator.evaluate(run, test)
        result_t = evaluator.show(
            ['recall_5', 'recall_10', 'recall_15', 'recall_20', 
             'map_cut_5', 'map_cut_10', 'map_cut_15', 'map_cut_20',
             'ndcg'])
        result['recall_5'] += result_t['recall_5']
        result['recall_10'] += result_t['recall_10']
        result['recall_15'] += result_t['recall_15']
        result['recall_20'] += result_t['recall_20']
        result['map_cut_5'] += result_t['map_cut_5']
        result['map_cut_10'] += result_t['map_cut_10']
        result['map_cut_15'] += result_t['map_cut_15']
        result['map_cut_20'] += result_t['map_cut_20']
        result['ndcg'] += result_t['ndcg']
        if cnt % 10 == 0:
            print('====> {} / {}'.format(start_idx, train_data.shape[0]))
    result['recall_5'] /= cnt
    result['recall_10'] /= cnt
    result['recall_15'] /= cnt
    result['recall_20'] /= cnt
    result['map_cut_5'] /= cnt
    result['map_cut_10'] /= cnt
    result['map_cut_15'] /= cnt
    result['map_cut_20'] /= cnt
    result['ndcg'] /= cnt
    # result['ndcg_10'] /= cnt
    # result['ndcg_15'] /= cnt
    # result['ndcg_20'] /= cnt
    # recall /= cnt
    # map_cut /= cnt
    # logger.info('====> Recall@{}: {:.4f}'.format(args.N, recall), '====> MAP@{}: {:.4f}'.format(args.N, map_cut))
    logger.info('result: {}'.format(result))
    return result['ndcg']


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Variational Auto Encoder')
    parser.add_argument('--batch', type=int, default=512, help='input batch size for training (default: 100)')
    parser.add_argument('-m', '--maxiter', type=int, default=30, help='number of epochs to train (default: 10)')
    parser.add_argument('--gpu', action='store_true', default = True, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 1)')
    parser.add_argument('--log', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    # parser.add_argument('--dir', help='dataset directory', default='/Users/chenyifan/jianguo/dataset')
    # parser.add_argument('--data', help='specify dataset', default='test')
    parser.add_argument('--layer', nargs='+', help='number of neurals in each layer', type=int, default=[1024,512,256])
    parser.add_argument('-N', help='number of recommended items', type=int, default=20)
    parser.add_argument('--lr', help='learning rate', type=float, default=1e-3)
    parser.add_argument('-a', '--alpha', help='parameter alpha', type=float, default=1)
    parser.add_argument('-b', '--beta', help='parameter beta', type=float, default=1)
    parser.add_argument('--rating', help='feed input as rating', action='store_true', default=True)
    parser.add_argument('--save', help='save model', action='store_true', default=True)
    parser.add_argument('--load', help='load model', type=int, default=0)
    parser.add_argument('--data_path', help='path of raw dataset', default='/root/autodl-tmp/yankai/RecTVAE')
    parser.add_argument('--data_name', help='name of dataset', default='CDs')
    parser.add_argument('--model_name', help='name of model to text feat', default='bert-base-uncased')
    parser.add_argument('--data_process', help='whether process dataset', default=False)
    parser.add_argument('--feat_load', help='whether load text feat', default=True)
    parser.add_argument('--feat_save', help='whether save text feat', default=True)
    parser.add_argument('--model', help='model type', default='TVAE')
    parser.add_argument('--feat_layer', help='number of neurals in bert layer', type=int, default=[500, 200])
    parser.add_argument('--weight_decay', help='', type=float, default=1e-5)
    parser.add_argument('--eval_step', help='', type=int, default=1)

    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    args.device = torch.device("cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")
    args.anneal = 0
    print('dataset directory: ' + args.data_path)
    directory = os.path.join(args.data_path)
    logger = logging.getLogger("mylogger")
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    log_dir = os.path.join(directory, 'log')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_dir = os.path.join(log_dir, args.data_name+args.model+str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))+'.txt')
    file_handler = logging.FileHandler(log_dir)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.info(args)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    data_processer = DataProcesser(args.data_path, args.data_name)
    if args.data_process:
        logger.info('data processing...')
        data_processer.inter_dataset('train')
        data_processer.inter_dataset('valid')
        data_processer.inter_dataset('test')
        data_processer.inter_dataset('text')
        logger.info('data processing finished')
    


    @cache.cache
    def load_data(directory, data_name):
        path = os.path.join(directory, 'data', data_name, 'train.npz')
        logger.info('train data path: ' + path)
        train_data = load_npz(path)
        # train_tensor = torch.from_numpy(train_data.toarray().astype('float32'))
        # train_data = input_inter(path)
        # train_tensor = torch.from_numpy(train_data.astype('float32')).to(args.device)
        # train_data = csr_matrix(train_data)
        path = os.path.join(directory, 'data', data_name, 'valid.npz')
        valid_data = load_npz(path)
        # valid_data = csr_matrix(valid_data)
        # valid_tensor = torch.from_numpy(valid_data.astype('float32')).to(args.device)
        # path = os.path.join(directory, 'data', args.data_name, 'test.npz')
        # test_data = load_npz(path)
        return train_data, valid_data
    train_data, valid_data= load_data(directory, args.data_name)
    # indices = torch.LongTensor([valid_data.row, valid_data.col]).to(args.device)
    # values = torch.FloatTensor(valid_data.data).to(args.device)
    # size = torch.Size(valid_data.shape)
    # valid_tensor = torch.sparse_coo_tensor(torch.LongTensor([valid_data.row, valid_data.col]).to(args.device),
    #                                        torch.FloatTensor(valid_data.data).to(args.device), 
    #                                        torch.Size(valid_data.shape)).to(args.device)

    # test_tensor = torch.sparse_coo_tensor(torch.LongTensor([test_data.row, test_data.col]).to(args.device),
    #                                        torch.FloatTensor(test_data.data).to(args.device), 
    #                                        torch.Size(test_data.shape)).to(args.device)
    # valid_tensor = test_tensor + valid_tensor
    # valid_tensor = valid_tensor.coalesce()

    # valid_data = coo_matrix((valid_tensor.values().cpu().numpy(), 
    #                                       (valid_tensor.indices()[0].cpu().numpy(), 
    #                                        valid_tensor.indices()[1].cpu().numpy())), 
    #                                      shape=valid_tensor.shape)
    
    # test_data = input_inter(path)
    # test_data = csr_matrix(test_data)
    # test_tensor = torch.from_numpy(test_data.astype('float32')).to(args.device)
    if args.model == 'VAE':
        if args.rating:
            # train_tensor = torch.from_numpy(train_data.toarray().astype('float32')).to(args.device)
            args.d = train_data.col.max() + 1
            indices = torch.LongTensor([train_data.row, train_data.col]).to(args.device)
            values = torch.FloatTensor(train_data.data).to(args.device)
            size = torch.Size(train_data.shape)
            sparse_tensor = torch.sparse_coo_tensor(indices, values, size).to(args.device)
            train_tensor = SparseTensorDataset(sparse_tensor)
            train_loader = DataLoader(train_tensor, args.batch, shuffle=True)
            loss_function = Softmax_loss
            pass
        else:
            features = data_processer.text2feat(args.model_name, args.batch,args.feat_load, args.feat_save)
            features = torch.from_numpy(features.astype('float32')).to(args.device)
            features = torch.transpose(features, 0, 1)
            args.d = features.shape[1]
            # features, valid_data = train_test_split(features, test_size=0.2, random_state=42)
            # valid_data, test_data = train_test_split(valid_data, test_size=0.5, random_state=42)
            train_loader = DataLoader(features, args.batch, shuffle=True)
            loss_function = Guassian_loss
        # args.d = train_loader.dataset.shape[1]
        model = VAE(args).to(args.device)
        if args.load > 0:
            name = 'cvae' if args.load == 2 else 'fvae'
            path = os.path.join(directory, 'model', args.data_name, name)
            for l in args.layer:
                path += '_' + str(l)
            logger.info('load model from path: ' + path)
            model.load_state_dict(torch.load(path))

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
        nowtime = datetime.datetime.now()
        logger.info('start training...' + str(nowtime))

        # evaluate()
        nexttime = datetime.datetime.now()            
        ndcg_max = 0
        cnt = 0
        logger.info('first evaluation finished. Time=' + str(nexttime-nowtime))
        for epoch in range(1, args.maxiter + 1):

            
            start = datetime.datetime.now()
            logger.info('epoch: ' + str(epoch) + ' start' + str(start))
            train(epoch)
            # evaluate()
            if epoch%args.eval_step == 0:
                ndcg = evaluate()
                if ndcg <= ndcg_max:
                    cnt += 1
                    logger.info('ndcg not improved for ' + str(cnt) + ' epochs')
                else:
                    cnt = 0
                    logger.info('ndcg improved from ' + str(ndcg_max) + ' to ' + str(ndcg))
                    ndcg_max = ndcg


            logger.info('epoch: ' + str(epoch) + ' finished, used' + str(datetime.datetime.now()-start))            
            if cnt >= 10/args.eval_step:
                logger.info('ndcg not improved for 10 epochs, stop training')
                break
        evaluate()    
        logger.info('training finished, used' + str(nowtime-datetime.datetime.now()))
        if args.save:
            name = 'cvae' if args.rating else 'fvae'
            path = os.path.join(directory, 'model', args.data_name)
            if not os.path.exists(path):
                os.mkdir(path)
            path = os.path.join(path, name)
            for l in args.layer:
                path += '_' + str(l)
            model.cpu()
            torch.save(model.state_dict(), path)
            logger.info('model saved in path: ' + path)
    elif args.model == 'TVAE':
        features = data_processer.text2feat('bert-large-uncased', args.batch,args.feat_load, args.feat_save)
        features = torch.from_numpy(features.astype('float32')).to(args.device)

        args.d = train_data.col.max() + 1
        args.feat_len = features.shape[1]



        indices = torch.LongTensor([train_data.row, train_data.col]).to(args.device)
        values = torch.FloatTensor(train_data.data).to(args.device)
        size = torch.Size(train_data.shape)
        sparse_tensor = torch.sparse_coo_tensor(indices, values, size).to(args.device)
        train_tensor = SparseTensorDataset(sparse_tensor)
        train_loader = DataLoader(train_tensor, args.batch, shuffle=True)
        loss_function = Softmax_loss
        name = 'tvae'
        if(len(args.feat_layer) == 0):
            model = TVAE(args).to(args.device)
            name = 'tvae'
        else:
            model = BERTVAE(args).to(args.device)
            name = 'bertvae'
        if args.load > 0:
            path = os.path.join(directory, 'model', args.data_name, name)
            for l in args.layer:
                path += '_' + str(l)
            path+= 'feat'
            for l in args.feat_layer:
                path += '_' + str(l)
            # path = os.path.join(path, name)
            # path = ('/root/autodl-tmp/yankai/RecTVAE/model/CDs/bertvae_1024_512_256')
            logger.info('load model from path: ' + path)
            model.load_state_dict(torch.load(path))

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
        nowtime = datetime.datetime.now()
        logger.info('start training...' + str(nowtime))

        # evaluate()
        nexttime = datetime.datetime.now()
        logger.info('first evaluation finished. Time=' + str(nexttime-nowtime))            
        ndcg_max = 0
        cnt = 0
        for epoch in range(1, args.maxiter + 1):

            
            start = datetime.datetime.now()
            logger.info('epoch: ' + str(epoch) + ' start' + str(start))
            train(epoch)
            # evaluate()
            if epoch%args.eval_step == 0:
                ndcg = evaluate()
                if ndcg <= ndcg_max:
                    cnt += 1
                    logger.info('ndcg not improved for ' + str(cnt) + ' epochs')
                else:
                    cnt = 0
                    logger.info('ndcg improved from ' + str(ndcg_max) + ' to ' + str(ndcg))
                    ndcg_max = ndcg


            logger.info('epoch: ' + str(epoch) + ' finished, used' + str(datetime.datetime.now()-start))            
            if cnt >= 10/args.eval_step:
                logger.info('ndcg not improved for 10 epochs, stop training')
                break
        # evaluate()    
        logger.info('training finished, used' + str(datetime.datetime.now()-nowtime))
        if(len(args.feat_layer) == 0):
            model = TVAE(args).to(args.device)
            name = 'tvae'
        else:
            model = BERTVAE(args).to(args.device)
            name = 'bertvae'
        if args.save:
            path = os.path.join(directory, 'model', args.data_name)
            if not os.path.exists(path):
                os.mkdir(path)
            path = os.path.join(path, name)
            for l in args.layer:
                path += '_' + str(l)
            path+= 'feat'
            for l in args.feat_layer:
                path += '_' + str(l)
            model.cpu()
            torch.save(model.state_dict(), path)
            logger.info('model saved in path: ' + path)
    elif args.model == 'SLIM':
        args.d = train_data.col.max() + 1
        indices = torch.LongTensor([train_data.row, train_data.col]).to(args.device)
        values = torch.FloatTensor(train_data.data).to(args.device)
        size = torch.Size(train_data.shape)
        sparse_tensor = torch.sparse_coo_tensor(indices, values, size).to(args.device)
        train_tensor = SparseTensorDataset(sparse_tensor)
        train_loader = DataLoader(train_tensor, args.batch, shuffle=True)
        loss_function = BCE_loss
        pass
    elif args.model == 'BPR':
        args.d = train_data.col.max() + 1
        indices = torch.LongTensor([train_data.row, train_data.col]).to(args.device)
        values = torch.FloatTensor(train_data.data).to(args.device)
        size = torch.Size(train_data.shape)
        sparse_tensor = torch.sparse_coo_tensor(indices, values, size).to(args.device)
        train_tensor = SparseTensorDataset(sparse_tensor)
        train_loader = DataLoader(train_tensor, args.batch, shuffle=True)





    logger.removeHandler(console_handler)
    logger.removeHandler(file_handler)
    console_handler.close()
    file_handler.close()
