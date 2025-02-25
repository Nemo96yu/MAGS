import math
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class partial_loss(nn.Module):
    def __init__(self, confidence, conf_ema_m=0.95):
        super().__init__()
        # self.confidence = torch.tensor([item.cpu().detach().numpy() for item in confidence]).cuda()

        self.confidence = confidence.cuda(6)
        # self.memory_conf = torch.zeros_like(self.confidence)
        self.conf_ema_m = conf_ema_m

    def set_conf_ema_m(self, epoch):
        start = 0.95
        end = 0.8
        epoch = epoch - 20
        self.conf_ema_m = 1. * epoch / 200 * (end - start) + start

    def forward(self, outputs, index):
        index = index.type(torch.long)
        logsm_outputs = F.log_softmax(outputs, dim=1)
        final_outputs = logsm_outputs * self.confidence[index, :]
        average_loss = - ((final_outputs).sum(dim=1)).mean()

        return average_loss
    
    def confidence_update(self, epoch, outputs,  batch_index, batchY):
        with torch.no_grad():

            sm_outputs = torch.softmax(outputs, dim=1)
            new_batch_confidence = sm_outputs * batchY.cuda(6)
            new_batch_confidence = new_batch_confidence / new_batch_confidence.sum(dim=1, keepdim=True)
            # # batch = new_batch_confidence.detach()
            # # oself.confidence[batch_index, :] = self.conf_ema_m * self.confidence[batch_index, :] + (1-self.conf_ema_m) * batch
            self.confidence[batch_index, :] = new_batch_confidence

            # Max Label Disambiguation
            # _, prot_pred = (outputs.cuda(6) * batchY.cuda(6)).max(dim=1)
            # pseudo_label = F.one_hot(prot_pred, batchY.shape[1]).float().cuda(6).detach()
            # self.confidence[batch_index, :] = pseudo_label

        return

class semantic_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, partialY):

        n, m = partialY.size()
        output1 = torch.softmax(output, dim=1).cuda(6)
        output2 = output1 * partialY.cuda(6)
        d = math.exp(-1)
        max_p, _ = output2.max(dim=1)
        output_max = output1 - max_p.reshape(-1, 1)
        output_exp = torch.exp(output_max)
        loss = (output_exp.sum() / n - 1 - d * (m-1)) /m
        return loss


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2):
        # euclidean_distance = F.pairwise_distance(output1, output2)
        # loss_contrastive = torch.mean(torch.pow(euclidean_distance, 2))

        KLDivLoss = nn.KLDivLoss(reduction='sum')
        p_output = F.softmax(output1, dim=1)
        q_output = F.softmax(output2, dim=1)
        log_mean_output = ((p_output + q_output) / 2).log()
        loss_contrastive = (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2
        return loss_contrastive
