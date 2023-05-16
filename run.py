from dataset.data_pre import DataProcesser
import argparse
from vae.utils import metric_map, metric_recall, sort2query, csr2test, Evaluator
from vae.vae import VAE
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.sparse import coo_matrix, load_npz
from tqdm import tqdm
from torch import optim
from sklearn.model_selection import train_test_split
import os
import datetime




def Guassian_loss(recon_x, x):
    recon_x = torch.sigmoid(recon_x)
    weights = x * args.alpha + (1 - x)
    loss = x - recon_x
    loss = torch.sum(weights * (loss ** 2))
    return loss


def BCE_loss(recon_x, x):
    recon_x = torch.sigmoid(recon_x)
    eps = 1e-8
    loss = -torch.sum(args.alpha * torch.log(recon_x + eps) * x + torch.log(1 - recon_x + eps) * (1 - x))
    return loss


def regularization(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())



def train(epoch):
    model.train()
    loss_value = 0
    for batch_idx, data in enumerate(tqdm(train_loader, unit='epoch')):
        data = data.to(args.device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data) + regularization(mu, logvar) * args.beta
        loss.backward()
        loss_value += loss.item()
        optimizer.step()


    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, loss_value / len(train_loader.dataset)))


def evaluate(split='valid'):
    y_true = eval(split + '_data')
    # Ttensor = eval(split + '_tensor')
    model.eval()
    y_score, _, _ = model(train_tensor)
    y_score.detach_()
    y_score = y_score.squeeze(0)
    y_score[train_data.row, train_data.col] = 0
    _, rec_items = torch.topk(y_score, args.N, dim=1)
    # y_pred = torch.gather(Ttensor, 1, rec_items).cpu().numpy()
    run = sort2query(rec_items[:, 0:args.N])
    test = csr2test(y_true.tocsr())
    evaluator = Evaluator({'recall', 'map_cut'})
    evaluator.evaluate(run, test)
    result = evaluator.show(
        ['recall_5', 'recall_10', 'recall_15', 'recall_20', 'map_cut_5', 'map_cut_10', 'map_cut_15', 'map_cut_20'])
    print(result)

def input_inter(data_path):
    data = np.loadtxt(data_path, dtype=int)
    user = data[:, 0]
    item = data[:, 1]

    user_num = np.max(user) + 1
    item_num = np.max(item) + 1

    user_item = np.zeros((user_num, item_num))
    user_item[user, item] = 1
    return user_item

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Variational Auto Encoder')
    parser.add_argument('--batch', type=int, default=64, help='input batch size for training (default: 100)')
    parser.add_argument('-m', '--maxiter', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--gpu', action='store_true', default = True, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=2022, help='random seed (default: 1)')
    parser.add_argument('--log', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    # parser.add_argument('--dir', help='dataset directory', default='/Users/chenyifan/jianguo/dataset')
    # parser.add_argument('--data', help='specify dataset', default='test')
    parser.add_argument('--layer', nargs='+', help='number of neurals in each layer', type=int, default=[100, 20])
    parser.add_argument('-N', help='number of recommended items', type=int, default=10)
    parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)
    parser.add_argument('-a', '--alpha', help='parameter alpha', type=float, default=1)
    parser.add_argument('-b', '--beta', help='parameter beta', type=float, default=1)
    parser.add_argument('--rating', help='feed input as rating', action='store_true', default=False)
    parser.add_argument('--save', help='save model', action='store_true', default=True)
    parser.add_argument('--load', help='load model', type=int, default=0)
    parser.add_argument('--data_path', help='path of raw dataset', default='/root/autodl-tmp/yankai/RecTVAE')
    parser.add_argument('--data_name', help='name of dataset', default='CDs')
    parser.add_argument('--model_name', help='name of model to text feat', default='bert-base-uncased')
    parser.add_argument('--data_process', help='whether process dataset', default=False)
    parser.add_argument('--feat_load', help='whether load text feat', default=True)
    parser.add_argument('--feat_save', help='whether save text feat', default=True)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    args.device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    print('dataset directory: ' + args.data_path)
    directory = os.path.join(args.data_path)
    data_processer = DataProcesser(args.data_path, args.data_name)
    if args.data_process:
        print('data processing...')
        data_processer.inter_dataset('train')
        data_processer.inter_dataset('valid')
        data_processer.inter_dataset('test')
        data_processer.inter_dataset('text')
        print('data processing finished')
    
    path = os.path.join(directory, 'data', args.data_name, 'train.npz')
    print('train data path: ' + path)
    train_data = load_npz(path)
    train_tensor = torch.from_numpy(train_data.toarray().astype('float32')).to(args.device)
    # train_data = input_inter(path)
    # train_tensor = torch.from_numpy(train_data.astype('float32')).to(args.device)
    # train_data = csr_matrix(train_data)
    path = os.path.join(directory, 'data', args.data_name, 'valid.npz')
    valid_data = load_npz(path)
    # valid_data = csr_matrix(valid_data)
    # valid_tensor = torch.from_numpy(valid_data.astype('float32')).to(args.device)
    path = os.path.join(directory, 'data', args.data_name, 'test.npz')
    test_data = load_npz(path)
    # test_data = input_inter(path)
    # test_data = csr_matrix(test_data)
    # test_tensor = torch.from_numpy(test_data.astype('float32')).to(args.device)
    
    if args.rating:
        

        train_loader = DataLoader(train_tensor, args.batch, shuffle=True)
        loss_function = BCE_loss
    else:
        features = data_processer.text2feat(args.model_name, args.batch,args.feat_load, args.feat_save)
        features = torch.from_numpy(features.astype('float32')).to(args.device)
        features = torch.transpose(features, 0, 1)
        # features, valid_data = train_test_split(features, test_size=0.2, random_state=42)
        # valid_data, test_data = train_test_split(valid_data, test_size=0.5, random_state=42)
        train_loader = DataLoader(features, args.batch, shuffle=True)
        loss_function = Guassian_loss
    args.d = train_loader.dataset.shape[1]
    model = VAE(args).to(args.device)
    if args.load > 0:
        name = 'cvae' if args.load == 2 else 'fvae'
        path = os.path.join(directory, 'model', name)
        for l in args.layer:
            path += '_' + str(l)
        print('load model from path: ' + path)
        model.load_state_dict(torch.load(path))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    nowtime = datetime.datetime.now()
    print('start training...' + str(nowtime))

    evaluate()
    nexttime = datetime.datetime.now()
    print('first evaluation finished. Time=' + str(nexttime-nowtime))
    for epoch in range(1, args.maxiter + 1):
        start = datetime.datetime.now()
        print('epoch: ' + str(epoch) + ' start' + str(start))
        train(epoch)
        evaluate()
        evaluate('test')
        print('epoch: ' + str(epoch) + ' finished, used' + str(start-datetime.datetime.now()))
    print('training finished, used' + str(nowtime-datetime.datetime.now()))
    if args.save:
        name = 'cvae' if args.rating else 'fvae'
        path = os.path.join(directory, 'model', name)
        if not os.path.exists(path):
            os.mkdirs(path)
        for l in args.layer:
            path += '_' + str(l)
        model.cpu()
        torch.save(model.state_dict(), path)
