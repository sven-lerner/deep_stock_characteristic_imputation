import numpy as np
import csv
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import loss_functions
import time
from tqdm.notebook import tqdm
import math

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def dec_portfolio_returns(ts_maps):
    portfolio_returns = [[] for _ in range(11)]
    for ts in sorted(ts_maps.keys()):
        data = ts_maps[ts]
        if len(data) > 10:
            data = np.array(data)
            preds = data[:,0]
            gts = data[:,1]
            num_per_portfolio = math.floor(gts.shape[0] / 10)
            pred_sort = np.argsort(preds)
            for i in range(10):
                if i == 9:
                    end = len(pred_sort) + 1
                else:
                    end = (i+1)*num_per_portfolio
                in_portfolio = pred_sort[i*num_per_portfolio:end]
                portfolio_returns[i].append(np.mean(gts[in_portfolio]))
            portfolio_returns[-1].append(portfolio_returns[9][-1] - portfolio_returns[0][-1])

    return [np.mean(x)/np.std(x) for x in portfolio_returns]

def get_factors(hidden_ts_maps, return_ts_maps):
    factor_returns = []
    for ts in sorted(hidden_ts_maps.keys()):
        hiddens = np.array(hidden_ts_maps[ts])
        returns = np.array(return_ts_maps[ts])
        if len(hidden_ts_maps[ts]) > 1:
            factors = np.linalg.lstsq(hiddens, returns, rcond=None)[0]
            factor_returns.append(factors)
    return np.array(factor_returns)


def get_factors_returns(expected_batch_dim, model, ar_model, data_loader, expect_oos_data=False):
    ts_factor_maps = defaultdict(list)
    ts_return_maps = defaultdict(list)
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            if expect_oos_data:
                time_steps, stock_idxs, data, mask, ordered_mask, factors, _, _, returns = data
            else:
                time_steps, stock_idxs, data, mask, ordered_mask, factors, returns = data            

            assert not torch.isnan(data).any()
        #         print(data, mask)
            assert not torch.logical_and(data <= 0, ~mask).any(), torch.min(data * mask)

            C_train = data.transpose_(0, 1).float().to(device)[:-1]
            time_steps = time_steps.transpose_(0, 1).float().to(device)[:-1]
            stock_idxs = stock_idxs.transpose_(0, 1).float().to(device)[:-1]

            returns = returns.transpose_(0, 1).float().to(device)[:-1]

            C_mask = mask.float().to(device).transpose_(0, 1)[:-1]
            C_ordered_mask = ordered_mask.float().to(device).transpose_(0, 1)[:-1]

            factors = factors.transpose_(0, 1).float().to(device)[1:]

            train_input = torch.cat([C_train, C_mask, factors], axis=2)

            if train_input.shape[1] == expected_batch_dim:
                if model is not None:
                    alpha_pred, beta_pred, hidden_out = model(train_input)

                    model_input = torch.cat([hidden_out[:-1].detach(), C_train[1:]], axis=2)
                    batch_size, batch_length, dim = model_input.shape
                    model_input = model_input.reshape(batch_size * batch_length, dim)

                    C_ordered_mask_reshape = C_ordered_mask[1:].reshape(batch_size * batch_length, 45)
                    C_mask_reshape = C_ordered_mask[1:].reshape(batch_size * batch_length, 45)
                    C_train_reshape = C_train[1:].reshape(batch_size * batch_length, 45)

                    time_steps_reshape = time_steps[1:].reshape(batch_size * batch_length, 1)
                    stock_idxs_reshape = stock_idxs[1:].reshape(batch_size * batch_length, 1)

                    returns_reshape = returns[1:].reshape(batch_size * batch_length, 1)
                else:
                    batch_size, batch_length, dim = C_train.shape
                    model_input = C_train.reshape(batch_size * batch_length, dim)

                    C_ordered_mask_reshape = C_ordered_mask.reshape(batch_size * batch_length, 45)
                    C_train_reshape = C_train.reshape(batch_size * batch_length, 45)
                    C_mask_reshape = C_ordered_mask.reshape(batch_size * batch_length, 45)

                    time_steps_reshape = time_steps.reshape(batch_size * batch_length, 1)
                    stock_idxs_reshape = stock_idxs.reshape(batch_size * batch_length, 1)

                    returns_reshape = returns.reshape(batch_size * batch_length, 1)

                hidden_states = ar_model.get_hidden_state(model_input)
                for i in range(hidden_states.shape[0]):
                    ts_factor_maps[time_steps_reshape[i, 0].item()].append(hidden_states[i].detach().cpu().numpy())
                    ts_return_maps[time_steps_reshape[i, 0].item()].append(returns_reshape[i].detach().cpu().numpy())
        return ts_factor_maps, ts_return_maps
            

from collections import defaultdict
def ts_r2(ts_map):
    numerator, denominator = [], []
    
    for time_step, data in ts_map.items():
        data = np.array(data)
        preds = data[:,0]
        gts = data[:,1]
        numerator.append(np.mean(np.square(preds - gts)))
        denominator.append(np.mean(np.square(gts)))
    return 1 - np.sum(numerator) / np.sum(denominator)
        

def xs_r2(stock_map):
    numerator, denominator = [], []
    max_t = max([len(x) for x in stock_map.values()])
    for stock, data in stock_map.items():
        data = np.array(data)
        preds = data[:,0]
        gts = data[:,1]
        numerator.append((data.shape[0] / max_t)*np.mean(np.square(preds - gts)))
        denominator.append((data.shape[0] / max_t)*np.square(np.mean(gts)))
    return 1 - np.sum(numerator) / np.sum(denominator)
    
def eval_model_is(chars, expected_batch_size, model, ar_model, data_loader, return_metrics=False, expect_oos_data=False):
    preds = []
    gts = []
    train_masks = []
    ts_maps = defaultdict(list)
    xs_maps = defaultdict(list)
    pred_alphas = []
    pred_betas = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            if expect_oos_data:
                time_steps, stock_idxs, data, mask, ordered_mask, factors, _, _, returns = data
            else:
                time_steps, stock_idxs, data, mask, ordered_mask, factors, returns = data

            assert not torch.isnan(data).any()
        #         print(data, mask)
            assert not torch.logical_and(data <= 0, ~mask).any(), torch.min(data * mask)

            C_train = data.transpose_(0, 1).float().to(device)[:-1]
            time_steps = time_steps.transpose_(0, 1).float().to(device)[:-1]
            stock_idxs = stock_idxs.transpose_(0, 1).float().to(device)[:-1]
            if return_metrics:
                returns = returns.transpose_(0, 1).float().to(device)[:-1]
            C_mask = mask.float().to(device).transpose_(0, 1)[:-1]
            C_ordered_mask = ordered_mask.float().to(device).transpose_(0, 1)[:-1]

            factors = factors.transpose_(0, 1).float().to(device)[1:]

            train_input = torch.cat([C_train, C_mask, factors], axis=2)


            if train_input.shape[1] == expected_batch_size:
                if model is not None:
                    alpha_pred, beta_pred, hidden_out = model(train_input)

                    model_input = torch.cat([hidden_out[:-1].detach(), C_train[1:]], axis=2)
                    batch_size, batch_length, dim = model_input.shape
                    model_input = model_input.reshape(batch_size * batch_length, dim)

                    C_ordered_mask_reshape = C_ordered_mask[1:].reshape(batch_size * batch_length, 45)
                    C_mask_reshape = C_ordered_mask[1:].reshape(batch_size * batch_length, 45)
                    C_train_reshape = C_train[1:].reshape(batch_size * batch_length, 45)

                    time_steps_reshape = time_steps[1:].reshape(batch_size * batch_length, 1)
                    stock_idxs_reshape = stock_idxs[1:].reshape(batch_size * batch_length, 1)
                    if return_metrics:
                        returns_reshape = returns[1:].reshape(batch_size * batch_length, 1)
                else:
                    batch_size, batch_length, dim = C_train.shape
                    model_input = C_train.reshape(batch_size * batch_length, dim)

                    C_ordered_mask_reshape = C_ordered_mask.reshape(batch_size * batch_length, 45)
                    C_train_reshape = C_train.reshape(batch_size * batch_length, 45)
                    C_mask_reshape = C_ordered_mask.reshape(batch_size * batch_length, 45)

                    time_steps_reshape = time_steps.reshape(batch_size * batch_length, 1)
                    stock_idxs_reshape = stock_idxs.reshape(batch_size * batch_length, 1)
                    if return_metrics:
                        returns_reshape = returns.reshape(batch_size * batch_length, 1)


                if return_metrics:
                    pred, rhat = ar_model.impute(model_input)

                    for i in range(pred.shape[0]):
                        ts_maps[time_steps_reshape[i, 0].item()].append((rhat[i, 0].item(),
                                                                         returns_reshape[i, 0].item()))
                        xs_maps[stock_idxs_reshape[i, 0].item()].append((rhat[i, 0].item(),
                                                                         returns_reshape[i, 0].item()))

                else:
                    
                    alphas, betas = ar_model.impute_alphas_betas(model_input)
                    pred = alphas / (alphas + betas)
                    
                    pred_alphas.append(alphas.detach().cpu().numpy())
                    pred_betas.append(betas.detach().cpu().numpy())

                preds.append(pred.detach().cpu().numpy())
                gts.append(C_train_reshape.detach().cpu().numpy())
                train_masks.append(C_mask_reshape.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    train_masks = np.concatenate(train_masks, axis=0)
    gts[train_masks == 1] = np.nan
    mse = np.nanmean((preds - gts)**2, axis=0)
    
    if return_metrics:
        return list(zip(chars, np.sqrt(mse))), np.sqrt(np.mean(mse)), ts_r2(ts_maps), xs_r2(xs_maps),\
                    dec_portfolio_returns(ts_maps)
    
    alphas = np.concatenate(pred_alphas, axis=0)
    betas = np.concatenate(pred_betas, axis=0)
    log_likelihood = loss_functions.beta_log_lik_np(gts, alphas, betas, train_masks,
                                                 reduce_axis=1, ret_mus=None, ret_sigmas=None, return_gts=None).mean()
    return list(zip(chars, np.sqrt(mse))), np.sqrt(np.mean(mse)), log_likelihood

def eval_model_oos(chars, expected_batch_size, model, ar_model, data_loader):
    preds = []
    gts = []
    train_masks = []
    pred_alphas = []
    pred_betas = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            try:
                time_steps, stock_idxs, data, mask, ordered_mask, factors, oos_data, oos_mask  = data
            except:
                time_steps, stock_idxs, data, mask, ordered_mask, factors, oos_data, oos_mask, _ = data
            assert not torch.isnan(data).any()
        #         print(data, mask)
            assert not torch.logical_and(data <= 0, ~mask).any(), torch.min(data * mask)

            C_train = data.transpose_(0, 1).float().to(device)[:-1]
            C_test = oos_data.transpose_(0, 1).float().to(device)[:-1]
            C_mask = mask.float().to(device).transpose_(0, 1)[:-1]
            C_test_mask = oos_mask.float().to(device).transpose_(0, 1)[:-1]
            C_ordered_mask = ordered_mask.float().to(device).transpose_(0, 1)[:-1]

            factors = factors.transpose_(0, 1).float().to(device)[1:]

            train_input = torch.cat([C_train, C_mask, factors], axis=2)


            if train_input.shape[1] == expected_batch_size:
                if model is not None:

                    alpha_pred, beta_pred, hidden_out = model(train_input)

                    model_input = torch.cat([hidden_out[:-1].detach(), C_train[1:]], axis=2)
                    batch_size, batch_length, dim = model_input.shape
                    model_input = model_input.reshape(batch_size * batch_length, dim)

                    C_ordered_mask_reshape = C_ordered_mask[1:].reshape(batch_size * batch_length, 45)
                    C_mask_reshape = C_ordered_mask[1:].reshape(batch_size * batch_length, 45)
                    C_test_mask_reshape = C_test_mask[1:].reshape(batch_size * batch_length, 45)

                    C_train_reshape = C_train[1:].reshape(batch_size * batch_length, 45)
                    C_test_reshape = C_test[1:].reshape(batch_size * batch_length, 45)

                else:
                    batch_size, batch_length, dim = C_train.shape
                    model_input = C_train.reshape(batch_size * batch_length, dim)

                    C_ordered_mask_reshape = C_ordered_mask.reshape(batch_size * batch_length, 45)
                    C_mask_reshape = C_ordered_mask.reshape(batch_size * batch_length, 45)
                    C_test_mask_reshape = C_test_mask.reshape(batch_size * batch_length, 45)

                    C_train_reshape = C_train.reshape(batch_size * batch_length, 45)
                    C_test_reshape = C_test.reshape(batch_size * batch_length, 45)

                alphas, betas = ar_model.impute_alphas_betas(model_input)
                pred = alphas / (alphas + betas)
                
                pred_alphas.append(alphas.detach().cpu().numpy())
                pred_betas.append(betas.detach().cpu().numpy())
                preds.append(pred.detach().cpu().numpy())
                gts.append(C_test_reshape.detach().cpu().numpy())
                train_masks.append(C_test_mask_reshape.detach().cpu().numpy())
    #             print(preds[-1][train_masks[-1] ==1], gts[-1][train_masks[-1] ==1])
    #             print()
            
    preds = np.concatenate(preds, axis=0)
    alphas = np.concatenate(pred_alphas, axis=0)
    betas = np.concatenate(pred_betas, axis=0)
    gts = np.concatenate(gts, axis=0)
    train_masks = np.concatenate(train_masks, axis=0)
    gts[train_masks != 1] = np.nan
    mse = np.nanmean((preds - gts)**2, axis=0)
    log_likelihood = loss_functions.beta_log_lik_np(gts, alphas, betas, 1 - train_masks,
                                                 reduce_axis=1, ret_mus=None, ret_sigmas=None, return_gts=None).mean()
    return list(zip(chars, np.sqrt(mse))), np.sqrt(np.mean(mse)), log_likelihood


def eval_ts_model_is(chars, expected_batch_size, model, data_loader, plot_ts=False):
    preds = []
    gts = []
    train_masks = []
    pred_alphas = []
    pred_betas = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            try:
                time_steps, stock_idxs, data, mask, ordered_mask, factors, returns = data
            except:
                time_steps, stock_idxs, data, mask, ordered_mask, factors, oos_data, oos_mask, returns = data
            assert not torch.isnan(data).any()
        #         assert not (data <= 0).any(), torch.min(data)
        #         print(data.shape)
            C_train = data.transpose_(0, 1).float().to(device)[:-1]

            C_mask = mask.float().to(device).transpose_(0, 1)[:-1]
            C_ordered_mask = ordered_mask.float().to(device).transpose_(0, 1)[:-1]

            factors = factors.transpose_(0, 1).float().to(device)[1:]

            train_input = torch.cat([C_train, C_mask, factors], axis=2)       
            if train_input.shape[1] == expected_batch_size:
                alpha_pred, beta_pred,_ = model(train_input)

                pred = alpha_pred / (alpha_pred + beta_pred)
                preds.append(pred.detach().cpu().numpy())
                pred_alphas.append(alpha_pred.detach().cpu().numpy())
                pred_betas.append(beta_pred.detach().cpu().numpy())
                gts.append(C_train.detach().cpu().numpy())
                train_masks.append(C_mask.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    train_masks = np.concatenate(train_masks, axis=0)
    gts[train_masks == 1] = np.nan
    alphas = np.concatenate(pred_alphas, axis=0)
    betas = np.concatenate(pred_betas, axis=0)
    squared_error = (preds - gts)**2
    if plot_ts:
        t_len = preds.shape[1]
        for i in range(5):
            for j in range(i*9, i*9 + 9):
                plt.plot(np.arange(0, t_len), np.sqrt(np.nanmean(squared_error[:,:,j], axis=0)), label=chars[j])
            plt.legend()
            plt.xlabel("Window Size")
            plt.ylabel("RMMSE")
            plt.show()

    mse = np.nanmean(squared_error, axis=(0,1))
    log_likelihood = loss_functions.beta_log_lik_np(gts, alphas, betas, train_masks,
                                                 reduce_axis=1, ret_mus=None, ret_sigmas=None, return_gts=None).mean()
    return list(zip(chars, np.sqrt(mse))), np.mean(np.sqrt(mse)), log_likelihood

def eval_ts_model_oos(chars, expected_batch_size, model, data_loader):
    preds = []
    gts = []
    train_masks = []
    pred_alphas = []
    pred_betas = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            time_steps, stock_idxs, data, mask, ordered_mask, factors, oos_data, oos_mask, returns  = data
            assert not torch.isnan(data).any()
        #         assert not (data <= 0).any(), torch.min(data)
        #         print(data.shape)
            C_train = data.transpose_(0, 1).float().to(device)[:-1]
            C_test = oos_data.transpose_(0, 1).float().to(device)[:-1]
            C_mask = mask.float().to(device).transpose_(0, 1)[:-1]
            C_ordered_mask = ordered_mask.float().to(device).transpose_(0, 1)[:-1]
            C_test_mask = oos_mask.float().to(device).transpose_(0, 1)[:-1]

            factors = factors.transpose_(0, 1).float().to(device)[1:]

            train_input = torch.cat([C_train, C_mask, factors], axis=2)       
            if train_input.shape[1] == expected_batch_size:
                alpha_pred, beta_pred,_ = model(train_input)

                pred = alpha_pred / (alpha_pred + beta_pred)
                pred_alphas.append(alpha_pred.detach().cpu().numpy())
                pred_betas.append(beta_pred.detach().cpu().numpy())
                preds.append(pred.detach().cpu().numpy())
                gts.append(C_test.detach().cpu().numpy())
                train_masks.append(C_test_mask.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    train_masks = np.concatenate(train_masks, axis=0)
    gts[train_masks != 1] = np.nan
    alphas = np.concatenate(pred_alphas, axis=0)
    betas = np.concatenate(pred_betas, axis=0)
    mse = np.nanmean((preds - gts)**2, axis=(0,1))
    
    log_likelihood = loss_functions.beta_log_lik_np(gts, alphas, betas, 1 - train_masks,
                                                 reduce_axis=1, ret_mus=None, ret_sigmas=None, return_gts=None).mean()
    return list(zip(chars, np.sqrt(mse))), np.mean(np.sqrt(mse)), log_likelihood


def get_oos_imputed_panel(chars, expected_batch_size, model, ar_model, data_loader):
    preds = []
    gts = []
    ts_idxs = []
    company_idxs = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            try:
                time_steps, stock_idxs, data, mask, ordered_mask, factors, oos_data, oos_mask  = data
            except:
                time_steps, stock_idxs, data, mask, ordered_mask, factors, oos_data, oos_mask, _ = data
            assert not torch.isnan(data).any()
        #         print(data, mask)
            assert not torch.logical_and(data <= 0, ~mask).any(), torch.min(data * mask)

            C_train = data.transpose_(0, 1).float().to(device)[:-1]
            time_steps = time_steps.transpose_(0, 1).float()[:-1]
            stock_idxs = stock_idxs.transpose_(0, 1).float()[:-1]
            
            C_test = oos_data.transpose_(0, 1).float().to(device)[:-1]
            C_mask = mask.float().to(device).transpose_(0, 1)[:-1]
            C_test_mask = oos_mask.float().to(device).transpose_(0, 1)[:-1]
            C_ordered_mask = ordered_mask.float().to(device).transpose_(0, 1)[:-1]

            factors = factors.transpose_(0, 1).float().to(device)[1:]

            train_input = torch.cat([C_train, C_mask, factors], axis=2)


            if train_input.shape[1] == expected_batch_size:
                if model is not None:

                    alpha_pred, beta_pred, hidden_out = model(train_input)

                    model_input = torch.cat([hidden_out[:-1].detach(), C_train[1:]], axis=2)
                    batch_size, batch_length, dim = model_input.shape
                    model_input = model_input.reshape(batch_size * batch_length, dim)

                    C_ordered_mask_reshape = C_ordered_mask[1:].reshape(batch_size * batch_length, 45)
                    C_mask_reshape = C_ordered_mask[1:].reshape(batch_size * batch_length, 45)
                    C_test_mask_reshape = C_test_mask[1:].reshape(batch_size * batch_length, 45)

                    C_train_reshape = C_train[1:].reshape(batch_size * batch_length, 45)
                    C_test_reshape = C_test[1:].reshape(batch_size * batch_length, 45)
                    
                    time_steps_reshape = time_steps[1:].reshape(batch_size * batch_length, 1)
                    stock_idxs_reshape = stock_idxs[1:].reshape(batch_size * batch_length, 1)

                else:
                    batch_size, batch_length, dim = C_train.shape
                    model_input = C_train.reshape(batch_size * batch_length, dim)

                    C_ordered_mask_reshape = C_ordered_mask.reshape(batch_size * batch_length, 45)
                    C_mask_reshape = C_ordered_mask.reshape(batch_size * batch_length, 45)
                    C_test_mask_reshape = C_test_mask.reshape(batch_size * batch_length, 45)

                    C_train_reshape = C_train.reshape(batch_size * batch_length, 45)
                    C_test_reshape = C_test.reshape(batch_size * batch_length, 45)
                    
                    time_steps_reshape = time_steps.reshape(batch_size * batch_length, 1)
                    stock_idxs_reshape = stock_idxs.reshape(batch_size * batch_length, 1)

                alphas, betas = ar_model.impute_alphas_betas(model_input)
                pred = alphas / (alphas + betas)
                
                ts_idxs.append(time_steps_reshape.detach().cpu().numpy())
                company_idxs.append(stock_idxs_reshape.detach().cpu().numpy())
                
                preds.append(pred.detach().cpu().numpy())
            
    preds = np.concatenate(preds, axis=0)
    company_idxs = np.concatenate(company_idxs, axis=0)
    ts_idxs = np.concatenate(ts_idxs, axis=0)
    
    return preds, company_idxs, ts_idxs