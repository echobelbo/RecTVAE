from dataset.data_pre import DataProcesser
import argparse
import vae
import torch
import numpy as np



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