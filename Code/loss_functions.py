import numpy as np
import csv
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Normal
from scipy.special import loggamma


def squared_error(C, alphas, betas, mask):
    '''
    C:  T x batch x 45
    alpha_c: T x batch x 45
    alpha_c: T x batch x 45
    '''
    assert not (C <= 0).any()
    assert not (C >= 1).any()
    preds = alphas / (alphas + betas)
    
    return torch.square(preds - C)


def beta_log_lik_np(C, alphas, betas, mask, reduce_axis=2, ret_mus=None, ret_sigmas=None, return_gts=None):
    '''
    C:  T x batch x 45
    alpha_c: T x batch x 45
    alpha_c: T x batch x 45
    '''
#     print(C.shape, alphas.shape, betas.shape, mask.shape)
    assert not (C*(1-mask) < 0).any()
    assert not (C >= 1).any()
    log_likelihoods = (alphas - 1) * np.log(np.maximum(C, 0)) + (betas - 1) * np.log(1-C)
    assert not np.logical_and(np.isnan(log_likelihoods), (mask == 0)).any(), \
        (np.logical_and(np.isnan(np.log(np.maximum(C, 0))), (mask == 0)).any(),
        np.logical_and(np.isnan(np.log(1-C)), (mask == 0)).any(),
         np.logical_and(np.isnan(alphas), (mask == 0)).any(),
        np.logical_and(np.isnan(betas), (mask == 0)).any(),
        )
    log_likelihoods += loggamma(alphas + betas) - loggamma(alphas) - loggamma(betas)
    assert not np.logical_and(np.isnan(log_likelihoods), (mask == 0)).any(), \
        (np.logical_and(np.isnan(np.log(np.maximum(C, 0))), (mask == 0)).any(),
        np.logical_and(np.isnan(np.log(1-C)), (mask == 0)).any(),
         np.logical_and(np.isnan(alphas), (mask == 0)).any(),
        np.logical_and(np.isnan(betas), (mask == 0)).any(),
        )
    assert not np.logical_and(np.isnan(log_likelihoods), (mask == 0)).any()
    
    log_l =  -1 * (np.nan_to_num(log_likelihoods) * (1-mask)).sum(axis=reduce_axis)
    return log_l


def beta_ll_loss(C, alphas, betas, mask, reduce_axis=2, ret_mus=None, ret_sigmas=None, return_gts=None):
    '''
    C:  T x batch x 45
    alpha_c: T x batch x 45
    alpha_c: T x batch x 45
    '''
#     print(C.shape, alphas.shape, betas.shape, mask.shape)
    assert not (C*(1-mask) < 0).any()
    assert not (C >= 1).any()
    log_likelihoods = (alphas - 1) * torch.log(torch.nn.functional.relu(C) + mask) + (betas - 1) * torch.log(1-C)
    assert not torch.isnan(log_likelihoods).any()
    assert not torch.logical_and(torch.isnan(log_likelihoods), (mask == 0)).any()
    log_likelihoods += torch.lgamma(alphas + betas) - torch.lgamma(alphas) - torch.lgamma(betas)
    assert not torch.logical_and(torch.isnan(log_likelihoods), (mask == 0)).any()
    
    log_l =  -1 * (log_likelihoods * (1-mask)).sum(axis=reduce_axis)
    if ret_mus is not None:
        log_probs = -1*torch.square(return_gts - ret_mus) / (2 * ret_sigmas)\
                    - torch.log(torch.sqrt(2 * np.pi * ret_sigmas))
        log_l -= log_probs.squeeze()
    return log_l



def beta_ll_loss_mean_scale(C, mus, scales, mask):
    '''
    C:  T x batch x 45
    alpha_c: T x batch x 45
    alpha_c: T x batch x 45
    '''
    alphas = mus * scales
    betas = scales - alphas
    assert not (C <= 0).any()
    assert not (C >= 1).any()
#     print(alphas.shape)
#     print(betas.shape)
    log_likelihoods = (alphas - 1) * torch.log(C) + (betas - 1) * torch.log(1-C)
    assert not torch.isnan(log_likelihoods).any()
    log_likelihoods += torch.lgamma(alphas + betas) - torch.lgamma(alphas) - torch.lgamma(betas)
    assert not torch.isnan(log_likelihoods).any()
    
    return -1 * (torch.nan_to_num(log_likelihoods) * mask).sum(axis=2)
    