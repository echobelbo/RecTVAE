from dataset.data_pre import DataProcesser
import argparse
from vae.utils import metric_map, metric_recall, sort2query, csr2test, Evaluator
from vae.vae import VAE, TVAE
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from scipy.sparse import coo_matrix, load_npz
from tqdm import tqdm
from torch import optim
# from sklearn.model_selection import train_test_split
import os
import datetime
# from functools import lru_cache
import logging
from joblib import Memory
# @lru_cache(maxsize=None)
# def load_npz_cached(path):
#     return load_npz(path)

cache = Memory('./cache', verbose=0)

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


def regularization(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def train_batch(epoch, train_loader):
    model.train()
    loss_value = 0
    for batch_idx, data in enumerate(tqdm(train_loader, unit='epoch')):
        data = data.to(args.device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data) + regularization(mu, logvar) * args.beta
        loss.backward()
        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        loss_value += loss.item()
        optimizer.step()
    logger.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch, loss_value / len(train_loader.dataset)))

def train(epoch):
    model.train()
    loss_value = 0
    for batch_idx, data in enumerate(tqdm(train_loader, unit='epoch')):
        data = data.to(args.device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data) + regularization(mu, logvar) * args.beta
        loss.backward()
        # max_norm = 1.0
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        loss_value += loss.item()
        optimizer.step()


    logger.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch, loss_value / len(train_loader.dataset)))


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
        'map_cut_20': 0
    }
    cnt = 0
    for start_idx in range(0, train_data.shape[0], args.batch):
        cnt += 1
        
        end_idx = min(start_idx + args.batch, train_data.shape[0])
        batch_data = train_data.tocsr()[start_idx:end_idx].toarray().astype('float32')
        nonzero_indices = np.nonzero(np.abs(batch_data) > 1e-12)

        
        train_tensor = torch.from_numpy(batch_data).to(args.device)

        y_score, _, _ = model(train_tensor)
        y_score.detach_()
        y_score = y_score.squeeze(0)
        y_score[nonzero_indices] = 0
        # y_score[train_data.row[:end_idx-start_idx], np.unique(train_data.col)] = 0
        _, rec_items = torch.topk(y_score, args.N, dim=1)
    
        run = sort2query(rec_items[:, 0:args.N])
        test = csr2test(y_true.tocsr()[start_idx:end_idx])
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
    logger.info(result)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Variational Auto Encoder')
    parser.add_argument('--batch', type=int, default=512, help='input batch size for training (default: 100)')
    parser.add_argument('-m', '--maxiter', type=int, default=20, help='number of epochs to train (default: 10)')
    parser.add_argument('--gpu', action='store_true', default = True, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=2022, help='random seed (default: 1)')
    parser.add_argument('--log', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    # parser.add_argument('--dir', help='dataset directory', default='/Users/chenyifan/jianguo/dataset')
    # parser.add_argument('--data', help='specify dataset', default='test')
    parser.add_argument('--layer', nargs='+', help='number of neurals in each layer', type=int, default=[512, 256, 128])
    parser.add_argument('-N', help='number of recommended items', type=int, default=10)
    parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)
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
    parser.add_argument('--model', help='model type', default='VAE')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    args.device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

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
    log_dir = os.path.join(log_dir, str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))+'.txt')
    file_handler = logging.FileHandler(log_dir)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

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
    def load_data(directory):
        path = os.path.join(directory, 'data', args.data_name, 'train.npz')
        logger.info('train data path: ' + path)
        train_data = load_npz(path)
        # train_tensor = torch.from_numpy(train_data.toarray().astype('float32'))
        # train_data = input_inter(path)
        # train_tensor = torch.from_numpy(train_data.astype('float32')).to(args.device)
        # train_data = csr_matrix(train_data)
        path = os.path.join(directory, 'data', args.data_name, 'valid.npz')
        valid_data = load_npz(path)
        # valid_data = csr_matrix(valid_data)
        # valid_tensor = torch.from_numpy(valid_data.astype('float32')).to(args.device)
        path = os.path.join(directory, 'data', args.data_name, 'test.npz')
        test_data = load_npz(path)
        return train_data, valid_data, test_data
    train_data, valid_data, test_data= load_data(directory)
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
            loss_function = BCE_loss
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
            path = os.path.join(directory, 'model', name)
            for l in args.layer:
                path += '_' + str(l)
            logger.info('load model from path: ' + path)
            model.load_state_dict(torch.load(path))

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        nowtime = datetime.datetime.now()
        logger.info('start training...' + str(nowtime))

        # evaluate()
        nexttime = datetime.datetime.now()
        logger.info('first evaluation finished. Time=' + str(nexttime-nowtime))
        for epoch in range(1, args.maxiter + 1):
            start = datetime.datetime.now()
            logger.info('epoch: ' + str(epoch) + ' start' + str(start))
            train(epoch)
            # evaluate()

            logger.info('epoch: ' + str(epoch) + ' finished, used' + str(datetime.datetime.now()-start))
        evaluate(split='test')    
        logger.info('training finished, used' + str(nowtime-datetime.datetime.now()))
        if args.save:
            name = 'cvae' if args.rating else 'fvae'
            path = os.path.join(directory, 'model', name)
            if not os.path.exists(path):
                os.mkdir(path)
            for l in args.layer:
                path += '_' + str(l)
            model.cpu()
            torch.save(model.state_dict(), path)
    elif args.model == 'TVAE':
        features = data_processer.text2feat(args.model_name, args.batch,args.feat_load, args.feat_save)
        features = torch.from_numpy(features.astype('float32')).to(args.device)

        args.d = train_data.col.max() + 1
        args.feat_len = features.shape[1]

        indices = torch.LongTensor([train_data.row, train_data.col]).to(args.device)
        values = torch.FloatTensor(train_data.data).to(args.device)
        size = torch.Size(train_data.shape)
        sparse_tensor = torch.sparse_coo_tensor(indices, values, size).to(args.device)
        train_tensor = SparseTensorDataset(sparse_tensor).to(args.device)
        train_loader = DataLoader(train_tensor, args.batch, shuffle=True)
        loss_function = BCE_loss
        model = TVAE(args).to(args.device)
        if args.load > 0:
            name = 'tvae'
            path = os.path.join(directory, 'model')
            for l in args.layer:
                name += '_' + str(l)
            path = os.path.join(path, name)
            logger.info('load model from path: ' + path)
            model.load_state_dict(torch.load(path))

    logger.removeHandler(console_handler)
    logger.removeHandler(file_handler)
    console_handler.close()
    file_handler.close()
