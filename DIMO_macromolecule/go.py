import os.path as osp
import numpy as np
from functools import partial
from typing import Type, Any, Callable, Union, List, Optional
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torchvision
import torch_geometric.transforms as T
from datasets import GODataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, fps, global_max_pool, global_mean_pool, radius
from models import Model
from utils import fmax
from torch.utils.data import Subset
import time
import sys


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def forward_diffusion(x, t, alpha_bars):
    noise = torch.randn_like(x)
    alpha_bar_t = alpha_bars[t].unsqueeze(-1)
    x_t = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise
    return x_t, noise


def reverse_diffusion(x_t, t, data, betas):
    _, _, pred_noise = model(data, 0.0)
    pred_noise = model_mlp(pred_noise)
    alpha_t = 1.0 - betas[t]
    alpha_t = alpha_t.unsqueeze(-1).expand_as(x_t)
    alpha_t = alpha_t.to(pred_noise.device)
    x_t = x_t.to(pred_noise.device)
    x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
    return x_0_pred


def diffusion_compute_loss(data):
    num_steps = 1000
    betas = torch.linspace(1e-4, 0.02, num_steps)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0).to(data.pos.device)
    losses = []
    x = data.pos
    t = torch.randint(0, num_steps, (x.size(0),), device=x.device)
    x_t, noise = forward_diffusion(x, t, alpha_bars)
    x_0_pred = reverse_diffusion(x_t, t, data, betas)
    x = x.to(x_0_pred.device)
    sub_loss = F.mse_loss(x_0_pred, x)
    losses.append(sub_loss)
    return torch.mean(torch.stack(losses))


def gcl_wo_aug(x1, x2, data):
    gcl_loss = contrastive_loss(x1, x2)
    gcl_aa_loss = contrastive_aa_loss(x1, x2, data)
    return gcl_loss + gcl_aa_loss


def contrastive_aa_loss(x1, x2, data):
    data_seq = data.seq.squeeze()
    num_ids = int(data_seq.max().item() + 1)

    mean_vectors_1 = []
    mean_vectors_2 = []
    for i in range(num_ids):
        mask = (data_seq == i)
        mean_vector1 = x1[mask].mean(dim=0)
        mean_vectors_1.append(mean_vector1)
        mean_vector2 = x2[mask].mean(dim=0)
        mean_vectors_2.append(mean_vector2)
    z_i = torch.stack(mean_vectors_1)
    z_j = torch.stack(mean_vectors_2)

    batch_size = z_i.size(0)
    feature_dim = z_i.size(1)
    eps=1e-15
    lambda_ = 1. / feature_dim
    z1_norm = (z_i - z_i.mean(dim=0)) / (z_i.std(dim=0) + eps)
    z2_norm = (z_j - z_j.mean(dim=0)) / (z_j.std(dim=0) + eps)
    c = (z1_norm.T @ z2_norm) / batch_size
    off_diagonal_mask = ~torch.eye(feature_dim).bool()
    gcl_aa_loss = (1 - c.diagonal()).pow(2).sum() + lambda_ * c[off_diagonal_mask].pow(2).sum()

    return gcl_aa_loss


def aug_fea(x):
    x_new = copy.deepcopy(x)
    x_new.x = drop_feature(x_new.x, 0.3)
    x_new.node_s = drop_feature(x_new.node_s, 0.3)
    return x_new


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x

def sim(z1, z2):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

def contrastive_loss(z1, z2):
    tau = 0.8
    f = lambda x: torch.exp(x / tau)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    gcl_loss = - torch.log(between_sim.diag() / (refl_sim.sum(1) - refl_sim.diag() + between_sim.sum(1)))
    return gcl_loss.mean()

def pretrain(epoch, dataloader, loss_fn, prune1, prune2):
    model.train()
    model_mlp.train()
    for data in dataloader:
        data = data.to(device)
        # warm-up phase (diffusion loss)
        optimizer_diffusion.zero_grad()
        diffusion_loss = diffusion_compute_loss(data)
        diffusion_loss.backward()
        optimizer_diffusion.step()
        # pretraining (gcl loss)
        optimizer.zero_grad()
        out1, _, x1 = model(data, prune1)
        out2, _, x2 = model(data, prune2)
        gcl_loss = gcl_wo_aug(x1, x2, data)
        gcl_loss.backward()
        optimizer.step()


def train(epoch, dataloader, loss_fn):
    model.train()
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        y = torch.from_numpy(np.stack(data.y, axis=0)).to(device)
        out, _, x = model(data, 0.0)
        loss = loss_fn(out.sigmoid(), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

def test(dataloader):
    model.eval()
    # Iterate over the validation data.

    probs = []
    labels = []
    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            out, _, _ = model(data, 0.0)
            prob = out.sigmoid().detach().cpu().numpy()
            y = np.stack(data.y, axis=0)
        probs.append(prob)
        labels.append(y)
    probs = np.concatenate(probs, axis=0)
    labels = np.concatenate(labels, axis=0)

    return fmax(probs, labels)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='DIMO')
    parser.add_argument('--data-dir', default='./data/go', type=str, metavar='N', help='data root directory')
    parser.add_argument('--level', default='cc', type=str, metavar='N', help='mf: molecular function, bp: biological process, cc: cellular component')
    parser.add_argument('--geometric-radius', default=4.0, type=float, metavar='N', help='initial 3D ball query radius')
    parser.add_argument('--sequential-kernel-size', default=21, type=int, metavar='N', help='1D sequential kernel size')
    parser.add_argument('--kernel-channels', nargs='+', default=[24], type=int, metavar='N', help='kernel channels')
    parser.add_argument('--base-width', default=32, type=float, metavar='N', help='bottleneck width')
    parser.add_argument('--channels', nargs='+', default=[256, 512, 1024, 2048], type=int, metavar='N', help='feature channels')
    parser.add_argument('--num-epochs', default=300, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('--batch-size', default=64, type=int, metavar='N', help='batch size')
    parser.add_argument('--lr', default=1e-3, type=float, metavar='N', help='learning rate')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)', dest='weight_decay')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--lr-milestones', nargs='+', default=[300, 400], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--ckpt-path', default='', type=str, help='path where to save checkpoint')
    parser.add_argument('--gpu', default=3, type=int, help='running gpu')
    parser.add_argument('--num-pretrain-epochs', default=3, type=int, metavar='N', help='number of pretraining epochs')
    parser.add_argument('--prune1', default=0.0, type=float, help='dropout of model1')
    parser.add_argument('--prune2', default=0.3, type=float, help='dropout of model2')
    parser.add_argument('--pretrain', default=False, type=str, help='whether to pretrain the model')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')

    train_dataset = GODataset(root=args.data_dir, random_seed=args.seed, level=args.level, split='train')
    valid_dataset = GODataset(root=args.data_dir, random_seed=args.seed, level=args.level, split='valid')
    test_dataset_95 = GODataset(root=args.data_dir, random_seed=args.seed, level=args.level, percent=95, split='test')
    
    print("train_dataset.num_classes: ", train_dataset.num_classes)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader_95 = DataLoader(test_dataset_95, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = Model(geometric_radii=[2*args.geometric_radius, 3*args.geometric_radius, 4*args.geometric_radius, 5*args.geometric_radius],
                  sequential_kernel_size=args.sequential_kernel_size,
                  kernel_channels=args.kernel_channels, channels=args.channels, base_width=args.base_width,
                  num_classes=train_dataset.num_classes).to(device)
    model_mlp = MLP(16, 32, 3).to(device)

    print("Training Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    optimizer = torch.optim.SGD(model.parameters(), weight_decay=args.weight_decay, lr=args.lr, momentum=args.momentum)
    optimizer_diffusion = torch.optim.SGD(list(model.parameters()) + list(model_mlp.parameters()), weight_decay=args.weight_decay, lr=args.lr, momentum=args.momentum)
    
    loss_fn = torch.nn.BCELoss(weight=torch.as_tensor(train_dataset.weights).to(device))

    # learning rate scheduler
    lr_weights = []
    for i, milestone in enumerate(args.lr_milestones):
        if i == 0:
            lr_weights += [np.power(args.lr_gamma, i)] * milestone
        else:
            lr_weights += [np.power(args.lr_gamma, i)] * (milestone - args.lr_milestones[i-1])
    if args.lr_milestones[-1] < args.num_epochs:
        lr_weights += [np.power(args.lr_gamma, len(args.lr_milestones))] * (args.num_epochs + 1 - args.lr_milestones[-1])
    lambda_lr = lambda epoch: lr_weights[epoch]
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    best_valid = best_test_95 = best_95 = 0.0
    best_epoch = 0

    # load pretraining model
    model_path = './pretrain_model_cc.pth'

    if args.pretrain == 'True':
        for epoch in range(args.num_pretrain_epochs):
            print("Cureent pretraining epoch: ", epoch)
            pretrain(epoch, train_loader, loss_fn, args.prune1, args.prune2)
        torch.save(model.state_dict(), model_path)
        print("Pretraining is finished!")

    else:
        if not os.path.exists(args.ckpt_path):
            os.mkdir(args.ckpt_path)
        model.load_state_dict(torch.load(model_path, map_location=device))
        epoch_times = 0

        for epoch in range(args.num_epochs):
            start_time = time.time()

            train(epoch, train_loader, loss_fn)
            lr_scheduler.step()
            valid_fmax = test(valid_loader)
            test_95 = test(test_loader_95)

            end_time = time.time()
            epoch_time = end_time - start_time
            epoch_times += epoch_time
            print(f'Epoch: {epoch+1:03d}, Validation: {valid_fmax:.4f}, Test: {test_95:.4f}, Time: {epoch_time:.4f}')
            if valid_fmax >= best_valid:
                best_valid = valid_fmax
                best_95 = test_95
                best_epoch = epoch
                checkpoint = model.state_dict()
            best_test_95 = max(test_95, best_test_95)

        print(f'Best: {best_epoch+1:03d}, Validation: {best_valid:.4f}, Test: {best_test_95:.4f}, Valided Test: {best_95:.4f}, Total time: {epoch_times:.4f}')
        if args.ckpt_path:
            torch.save(checkpoint, osp.join(args.ckpt_path, 'final_model.pth'))
