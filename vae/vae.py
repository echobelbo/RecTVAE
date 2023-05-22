import torch
from torch.nn import functional
from torch import nn
from .utils import trace
from transformers import BertTokenizer, BertModel
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_


def xavier_normal_initialization(module):
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
        if module.padding_idx is not None:
            constant_(module.weight.data[module.padding_idx], 0.)
    elif isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.l = len(args.layer)
        self.device = args.device
        self.inet = nn.ModuleList()
        darray = [args.d] + args.layer
        for i in range(self.l - 1):
            self.inet.append(nn.Linear(darray[i], darray[i + 1]))
        self.mu = nn.Linear(darray[self.l - 1], darray[self.l])
        self.sigma = nn.Linear(darray[self.l - 1], darray[self.l])
        self.gnet = nn.ModuleList()
        for i in range(self.l):
            self.gnet.append(nn.Linear(darray[self.l - i], darray[self.l - i - 1]))

        self.apply(xavier_normal_initialization)

    def encode(self, x):
        h = x.to(self.device)
        for i in range(self.l - 1):
            h = functional.relu(self.inet[i](h))
        return self.mu(h), self.sigma(h)

    def decode(self, z):
        h = z.to(self.device)
        for i in range(self.l - 1):
            h = functional.relu(self.gnet[i](h))
        return self.gnet[self.l - 1](h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def infer_reg(self):
        reg = 0
        for infer in self.inet:
            for param in infer.parameters():
                reg += trace(param)
        return reg

    def gen_reg(self):
        reg = 0
        for infer in self.gnet:
            for param in infer.parameters():
                reg += trace(param)
        return reg

class TVAE(nn.Module):
    def __init__(self, args):
        super(TVAE, self).__init__()
        self.l = len(args.layer)
        self.device = args.device
        self.inet = nn.ModuleList()
        darray = [args.d] + args.layer
        if self.l > 1:
            self.inet.append(nn.Linear(darray[0] + args.feat_len, darray[1]))
            for i in range(1, self.l - 1):
                self.inet.append(nn.Linear(darray[i], darray[i + 1]))
            self.mu = nn.Linear(darray[self.l - 1], darray[self.l])
            self.sigma = nn.Linear(darray[self.l - 1], darray[self.l])
        else:
            self.mu = nn.Linear(darray[self.l - 1]+args.feat_len, darray[self.l])
            self.sigma = nn.Linear(darray[self.l - 1]+args.feat_len, darray[self.l])
        # self.inet.append(nn.Linear(darray[self.l - 1], darray[self.l]*2))
        self.gnet = nn.ModuleList()
        for i in range(self.l):
            self.gnet.append(nn.Linear(darray[self.l - i], darray[self.l - i - 1]))

    def encode(self, x):
        h = x.to(self.device)
        for i in range(self.l - 1):
            h = functional.relu(self.inet[i](h))
        
        return self.mu(h), self.sigma(h)

    def decode(self, z):
        h = z.to(self.device)
        for i in range(self.l - 1):
            h = functional.relu(self.gnet[i](h))
        return self.gnet[self.l - 1](h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, feat):
        feat = torch.matmul(x, feat)
        # feat = feat.detach()
        feat = torch.nn.functional.normalize(feat, p=1, dim=1)
        x = x.to_dense()
        x = torch.cat((x, feat), dim=1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def infer_reg(self):
        reg = 0
        for infer in self.inet:
            for param in infer.parameters():
                reg += trace(param)
        return reg

    def gen_reg(self):
        reg = 0
        for infer in self.gnet:
            for param in infer.parameters():
                reg += trace(param)
        return reg
    
class BPR(nn.Module):
    def __init__(self, num_users, num_items, latent_dim):
        super(BPR, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.U = nn.Embedding(num_users, latent_dim)
        self.I = nn.Embedding(num_items, latent_dim)

    def forward(self, user, item_i, item_j):
        user_embedding = self.U(user)
        item_i_embedding = self.I(item_i)
        item_j_embedding = self.I(item_j)
        x_uij = torch.sum(user_embedding * item_i_embedding, dim=1) - torch.sum(user_embedding * item_j_embedding, dim=1)
        return x_uij

class BERTVAE(nn.Module):
    def __init__(self, args):
        super(BERTVAE, self).__init__()
        self.l = len(args.layer)
        self.inet = nn.ModuleList()
        darray = [args.d] + args.layer
        self.bertnet = nn.ModuleList()
        self.bertnet.append(nn.Linear(args.feat_len, args.feat_layer[0]))
        self.bertnet.append(nn.ReLU())
        for i in range(1, len(args.feat_layer)):
            self.bertnet.append(nn.Linear(args.feat_layer[i-1], args.feat_layer[i]))
            if i != len(args.feat_layer) - 1:
                self.bertnet.append(nn.ReLU())
            # self.bertnet.append(nn.ReLU())
        if self.l > 1:
            self.inet.append(nn.Linear(darray[0] + args.feat_layer[-1], darray[1]))
            for i in range(1, self.l - 1):
                self.inet.append(nn.Linear(darray[i], darray[i + 1]))
            self.mu = nn.Linear(darray[self.l - 1], darray[self.l])
            self.sigma = nn.Linear(darray[self.l - 1], darray[self.l])
        else:
            self.mu = nn.Linear(darray[self.l - 1]+args.feat_layer[-1], darray[self.l])
            self.sigma = nn.Linear(darray[self.l - 1]+args.feat_layer[-1], darray[self.l])
        # self.inet.append(nn.Linear(darray[self.l - 1], darray[self.l]*2))
        self.gnet = nn.ModuleList()
        for i in range(self.l):
            self.gnet.append(nn.Linear(darray[self.l - i], darray[self.l - i - 1]))\
        
        # self.apply(xavier_normal_initialization)
            
    def encode(self, x):
        h = x
        for i in range(self.l - 1):
            h = functional.relu(self.inet[i](h))
        
        return self.mu(h), self.sigma(h)

    def decode(self, z):
        h = z
        for i in range(self.l - 1):
            h = functional.relu(self.gnet[i](h))
        return self.gnet[self.l - 1](h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def bert_layer_forward(self, feat):
        for i in range(len(self.bertnet)):
            feat = self.bertnet[i](feat)
        return feat

    def forward(self, x, feat):
        feat = torch.matmul(x, feat)
        feat = feat / x.sum(dim=-1, keepdim=True).to_dense()
        feat = self.bert_layer_forward(feat)
        # feat = feat.detach()
        # feat = torch.nn.functional.normalize(feat, p=1, dim=1)
        x = x.to_dense()
        x = torch.cat((x, feat), dim=1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def infer_reg(self):
        reg = 0
        for infer in self.inet:
            for param in infer.parameters():
                reg += trace(param)
        return reg

    def gen_reg(self):
        reg = 0
        for infer in self.gnet:
            for param in infer.parameters():
                reg += trace(param)
        return reg

class SLIM(nn.Module):
    def __init__(self, args):
        super(SLIM, self).__init__()
        self.l = len(args.layer)
        self.net = nn.ModuleList()
        darray = [args.d] + args.layer + [args.d]
        for i in range(self.l):
            self.net.append(nn.Linear(darray[i], darray[i + 1]))

    def forward(self, x):
        for i in range(self.l):
            x = self.net[i](x)
        return x

    