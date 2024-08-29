
from .loss_computer import NCESoftmaxLoss
import torch.nn.functional as F
import torch.nn as nn
import torch
import logging
logger = logging.getLogger()


class ContrastiveLoss(nn.Module):
    def __init__(self, loss_computer: str, temperature: float, args) -> None:
        super().__init__()
        self.device = args.device

        if loss_computer == 'nce_softmax':
            self.loss_computer = NCESoftmaxLoss(self.device)
        else:
            raise NotImplementedError(f"Loss Computer {loss_computer} not Support!")
        self.temperature = temperature

    def forward(self, z_i, z_j):
        # SimCSE
        batch_size = z_i.size(0)

        emb = F.normalize(torch.cat([z_i, z_j]))

        similarity = torch.matmul(emb, emb.t()) - torch.eye(batch_size*2).to(self.device) * 1e12
        similarity = similarity * 20
        loss = self.loss_computer(similarity)
        
        return loss


class PointwiseLoss(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.device = args.device

    def forward(self, z_i, z_j):
        # Pointwise contrast
        # z_i_norm = F.normalize(z_i, dim=-1, p=2)
        # z_j_norm = F.normalize(z_j, dim=-1, p=2)
        # return torch.norm(z_i_norm - z_j_norm)

        batch_size = z_i.size(0)
        feature_dim = z_i.size(1)
        eps=1e-15
        lambda_ = 1. / feature_dim
        z1_norm = (z_i - z_i.mean(dim=0)) / (z_i.std(dim=0) + eps)
        z2_norm = (z_j - z_j.mean(dim=0)) / (z_j.std(dim=0) + eps)
        c = (z1_norm.T @ z2_norm) / batch_size
        off_diagonal_mask = ~torch.eye(feature_dim).bool()
        # loss_ponit = torch.norm(z1_norm - z2_norm)
        loss_gbt = (1 - c.diagonal()).pow(2).sum() + lambda_ * c[off_diagonal_mask].pow(2).sum()

        return loss_gbt


class FlatNCE(nn.Module):
    def __init__(self, temperature):
        self.temperature = temperature
        super().__init__()
    
    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)

        features = torch.cat([z_i, z_j], dim=0)
        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(labels.shape[0], -1)

        # logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(positives.shape[0], dtype=torch.long)
        logits = (negatives - positives)/self.temperature
        clogits = torch.logsumexp(logits, dim=1, keepdim=True)
        loss = torch.exp(clogits - clogits.detach())




# _, features = self.model(images)
# logits, labels = self.flat_loss(features)
# v = torch.logsumexp(logits, dim=1, keepdim=True) #(512,1)
# loss_vec = torch.exp(v-v.detach())

# assert loss_vec.shape == (len(logits),1)
# dummy_logits = torch.cat([torch.zeros(logits.size(0),1).to(self.args.device), logits],1)
# loss = loss_vec.mean()-1 + self.criterion(logits, labels).detach() 
