import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import binary_cross_entropy_with_logits, normalize


def contrastive_loss4x4(projections, max_nsample, temperature, cu_labels=None):
    [B, H, W] = cu_labels.shape
    u_new = cu_labels.reshape([B, 256, 64, 4]).permute(0, 2, 3, 1)
    u_new = u_new.reshape([B, 64, 4, 64, 4]).permute(0, 3, 1, 4, 2)
    u_new = u_new.reshape([B, 64, 64, 16]).reshape([-1, 16])
    sums = torch.sum(u_new, 1)

    idx_1 = sums==16.
    idx_0 = sums==0.

    channel = projections.shape[1]
    projections = projections.permute(0, 2, 3, 1)
    projections = projections.reshape([-1, channel])

    u_new0 = u_new[idx_0, :]
    u_new1 = u_new[idx_1, :]
    projections0 = projections[idx_0, :]
    projections1 = projections[idx_1, :]
    sampler0 = np.arange(u_new0.shape[0])
    sampler1 = np.arange(u_new1.shape[0])

    np.random.shuffle(sampler0)
    np.random.shuffle(sampler1)
    idx = min(len(sampler0), len(sampler1), max_nsample)
    
    if idx <= 2:
        return 0
    else:
        mask = torch.eye(idx, dtype=torch.bool)
        u_new0 = u_new0[sampler0[:idx],:]
        u_new1 = u_new1[sampler1[:idx],:]
        projections0 = projections0[sampler0[:idx], :]
        projections1 = projections1[sampler1[:idx], :]
        projections0 = normalize(projections0, p=2, dim=1)
        projections1 = normalize(projections1, p=2, dim=1)

        logits_00 = torch.matmul(projections0, projections0.T) / temperature
        logits_11 = torch.matmul(projections1, projections1.T) / temperature
        logits_01 = torch.matmul(projections0, projections1.T) / temperature
        
        # exclude diagonal parts
        logits_00 = logits_00[~mask].view(logits_00.shape[0], -1)
        logits_11 = logits_11[~mask].view(logits_11.shape[0], -1)

        loss_contrast_00 = binary_cross_entropy_with_logits(logits_00, torch.ones_like(logits_00))
        loss_contrast_11 = binary_cross_entropy_with_logits(logits_11, torch.ones_like(logits_11))
        loss_contrast_01 = binary_cross_entropy_with_logits(logits_01, torch.zeros_like(logits_01))
        loss_contrast = loss_contrast_00 + loss_contrast_11 + loss_contrast_01
    return loss_contrast
  
  
def contrastive_loss1x1(scr_gts, projections, max_nsample, temperature, cu_labels=None):
    # ready for sampling 'pixel-wise features' cuz it's too much
    if cu_labels==None:
        scr0_bs, scr0_ys, scr0_xs = torch.nonzero(scr_gts==0, as_tuple=True)
        scr1_bs, scr1_ys, scr1_xs = torch.nonzero(scr_gts==1, as_tuple=True)
    else:  
        scr0_bs, scr0_ys, scr0_xs = torch.nonzero(torch.logical_or(scr_gts==0, cu_labels==0), as_tuple=True)
        scr1_bs, scr1_ys, scr1_xs = torch.nonzero(torch.logical_or(scr_gts==1, cu_labels==1), as_tuple=True)
  
    scr_sampler0 = np.arange(len(scr0_bs))
    scr_sampler1 = np.arange(len(scr1_bs))
    np.random.shuffle(scr_sampler0)
    np.random.shuffle(scr_sampler1)
    
    # class balancing & feature number control
    idx = min(len(scr_sampler0), len(scr_sampler1), max_nsample)
    scr0_bs, scr0_ys, scr0_xs = scr0_bs[scr_sampler0[:idx]], scr0_ys[scr_sampler0[:idx]], scr0_xs[scr_sampler0[:idx]]
    scr1_bs, scr1_ys, scr1_xs = scr1_bs[scr_sampler1[:idx]], scr1_ys[scr_sampler1[:idx]], scr1_xs[scr_sampler1[:idx]]
    scr0_mat = normalize(projections[scr0_bs, :, scr0_ys, scr0_xs], p=2, dim=1) # [idx, proj_ch]
    scr1_mat = normalize(projections[scr1_bs, :, scr1_ys, scr1_xs], p=2, dim=1) # [idx, proj_ch]

    # L2 normalization on feature vectors (pixel)
    cls0_vecs, cls1_vecs = list(), list()
    cls0_vecs.append(scr0_mat) # [[idx, proj_ch], [idx, proj_ch], ... , [idx, proj_ch]]
    cls1_vecs.append(scr1_mat) # [[idx, proj_ch], [idx, proj_ch], ... , [idx, proj_ch]]

    cls0_mat = torch.cat(cls0_vecs, dim=0) if len(cls0_vecs) > 1 else cls0_vecs[0] # [idx*proj_ch, proj_ch]
    cls1_mat = torch.cat(cls1_vecs, dim=0) if len(cls1_vecs) > 1 else cls1_vecs[0] # [idx*proj_ch, proj_ch]

    # p-p, n-n, p-n similarity
    mask = torch.eye(idx, dtype=torch.bool)
    logits_00 = torch.matmul(cls0_mat, cls0_mat.T) / temperature # [idx,proj_ch] * [proj_ch,idx] = [idx, idx]
    logits_11 = torch.matmul(cls1_mat, cls1_mat.T) / temperature # [idx,proj_ch] * [proj_ch,idx] = [idx, idx]
    logits_01 = torch.matmul(cls0_mat, cls1_mat.T) / temperature # [idx, idx]

    # exclude diagonal parts
    if idx != 0:
        logits_00 = logits_00[~mask].view(logits_00.shape[0], -1)
        logits_11 = logits_11[~mask].view(logits_11.shape[0], -1)

    loss_contrast_00 = binary_cross_entropy_with_logits(logits_00, torch.ones_like(logits_00))
    loss_contrast_11 = binary_cross_entropy_with_logits(logits_11, torch.ones_like(logits_11))
    loss_contrast_01 = binary_cross_entropy_with_logits(logits_01, torch.zeros_like(logits_01))

    loss_contrast = loss_contrast_00 + loss_contrast_11 + loss_contrast_01
    return loss_contrast