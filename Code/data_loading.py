import numpy as np
from tqdm.notebook import tqdm
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, Dataset
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy


def percentile_rank(x, UNK=np.nan):
    mask = np.logical_not(np.isnan(x))
    x_copy = np.copy(x)
    x_mask = x_copy[mask]
    n = len(x_mask)
    if n > 1:
        temp = [(i, x_mask[i]) for i in range(n)]
        temp_sorted = sorted(temp, key=lambda t: t[1])
        idx = sorted([(temp_sorted[i][0], i) for i in range(n)], key=lambda t: t[0])
        x_copy[mask] = np.array([idx[i][1] for i in range(n)]) / (n - 1)
    elif n == 1:
        x_copy[mask] = 0.5
    return x_copy

def percentile_rank_panel(char_panel):
    ret_panel = np.zeros(char_panel.shape)
    ret_panel[:, :, :] = np.nan
    for t in tqdm(range(char_panel.shape[0])):
        for i in range(char_panel.shape[2]):
            ret_panel[t, :, i] = percentile_rank(char_panel[t, :, i])
        assert np.sum(np.isnan(ret_panel[t])) > 0, 'something fucky'
    return ret_panel

def get_data_dataframe(data_panel, return_panel, char_names, dates, permnos, monthly_updates, mask):
    T, N, C = data_panel.shape
    if mask is None:
        nonnan_returns = np.argwhere(np.logical_or(~np.isnan(return_panel), np.any(~np.isnan(data_panel),
                                                                                   axis=2)))
        num_nonnan_returns = np.sum(np.logical_or(~np.isnan(return_panel), np.any(~np.isnan(data_panel),
                                                                                   axis=2)))
    else:
        nonnan_returns = np.argwhere(mask)
        num_nonnan_returns = np.sum(mask)
    
    data_matrix = np.zeros((num_nonnan_returns, C+4))
    columns = np.append(char_names, ["return", "date", "permno", "monthly_update"])
    for i in range(nonnan_returns.shape[0]):
        nonnan_return = nonnan_returns[i]
        data_matrix[i,:C] = data_panel[nonnan_return[0], nonnan_return[1], :]
        data_matrix[i,C] = return_panel[nonnan_return[0], nonnan_return[1]]
        data_matrix[i,C+1] = dates[nonnan_return[0]]
        data_matrix[i,C+2] = permnos[nonnan_return[1]]
        data_matrix[i,C+3] = monthly_updates[nonnan_return[0], nonnan_return[1]]
    
    chars_and_returns_df = pd.DataFrame(data_matrix)
    chars_and_returns_df.columns = columns
    
    return chars_and_returns_df

def get_data_panel(path, rf_path, computstat_data_present_filter=True, financial_firm_filter=True, start_date=None):
    data = pd.read_feather(path)
    if start_date is not None:
        data = data.loc[data.date >= start_date]
    print(data.columns)
    dates = data['date'].unique()
    dates.sort()
    permnos = data['permno'].unique().astype(int)
    permnos.sort()
    rf_data = pd.io.parsers.read_csv(rf_path).to_numpy()

    date_vals = [int(date) for date in dates]
    chars = np.array(data.columns.tolist()[:-4])
    print(chars)
    chars.sort()

    char_data = np.zeros((len(date_vals), permnos.shape[0], len(chars)))
    monthly_updates = np.zeros((len(date_vals), permnos.shape[0]))
    char_data[:, :, :] = np.nan
    returns = np.zeros((len(date_vals), permnos.shape[0]))
    returns[:, :] = np.nan
    rfts = []

    permno_map = np.zeros(int(max(permnos)) + 1, dtype=int)
    for i, permno in enumerate(permnos):
        permno_map[permno] = i

    for i, date in enumerate(dates):
        date_data = data.loc[data['date'] == date].sort_values(by='permno')
        date_permnos = date_data['permno'].to_numpy().astype(int)
        permno_inds_for_date = permno_map[date_permnos]
        char_data[i, permno_inds_for_date, :] = date_data[chars].to_numpy()
        monthly_updates[i, permno_inds_for_date] = date_data["monthly_update"].to_numpy()
        returns[i, permno_inds_for_date] = date_data['return'].to_numpy()
        rft_idx = np.argwhere(rf_data[:,0] == str(int(date // 100)))[0][0]
        rfts.append(float(rf_data[rft_idx,1]) / 100)

    percentile_rank_chars = percentile_rank_panel(char_data) - 0.5
    
    assert np.all(np.isnan(percentile_rank_chars) == np.isnan(char_data))
    
    if computstat_data_present_filter:
        cstat_permnos = pd.read_csv("../Data/compustat_permnos.csv")["PERMNO"].to_numpy()
        permno_filter = np.isin(permnos, cstat_permnos)
        percentile_rank_chars = percentile_rank_chars[:,permno_filter,:]
        char_data = char_data[:,permno_filter,:]
        permnos = permnos[permno_filter]
        monthly_updates = monthly_updates[:,permno_filter]
        returns = returns[:,permno_filter]
        
    if financial_firm_filter:
        sic_fic = pd.read_csv("../data/sic_fic.csv")
        non_fininancial_permnos = ~np.isin(permnos, sic_fic.loc[sic_fic['sic']//1000 == 6]['LPERMNO'].unique())
        percentile_rank_chars = percentile_rank_chars[:,non_fininancial_permnos,:]
        char_data = char_data[:,non_fininancial_permnos,:]
        permnos = permnos[non_fininancial_permnos]
        monthly_updates = monthly_updates[:,non_fininancial_permnos]
        returns = returns[:,non_fininancial_permnos]
    
    return percentile_rank_chars, char_data, chars, date_vals, returns, permnos, rfts, monthly_updates



class NpDataset(Dataset):
    def __init__(self, data, target, idxs):
        self.data = data
        self.target = target
        self.idxs = idxs
    def __len__(self): 
        return self.data.shape[0]
    def __getitem__(self, i): 
        return self.data[i], self.target[i], self.idxs[i]
def calc_sharpe(r):
    return np.mean(r / r.std())



def get_train_val_test_loaders(train_data, val_data, test_data, test_partitions=None,
                              device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_data, val_data, test_data = get_premade_x_y_ind(train_data, val_data, test_data, test_partitions)
    
    char_panel_train, return_panel_train, ind_panel_train = train_data
    char_panel_val, return_panel_val, ind_panel_val = val_data
    
    train_dataset = NpDataset(char_panel_train, return_panel_train, ind_panel_train)
    train_loader = DataLoader(train_dataset, batch_size=10000000, shuffle=True)
    val_dataset = NpDataset(char_panel_val, return_panel_val, ind_panel_val)
    val_loader = DataLoader(val_dataset, batch_size=1000000, shuffle=False)  

    X,Y,ind = None,None,None
    for i,data in enumerate(train_loader):
        X, Y, ind = data
#         X = X.to(device)
#         Y = Y.to(device)
#         ind = ind.to(device)

    X_val, Y_val, ind_val = None,None,None
    for i,data in enumerate(val_loader):
        X_val, Y_val, ind_val = data
#         X_val = X_val.to(device)
#         Y_val = Y_val.to(device)
#         ind_val = ind_val.to(device)
    
    if test_partitions is None:
        char_panel_test, return_panel_test, ind_panel_test = test_data
        test_dataset = NpDataset(char_panel_test, return_panel_test, ind_panel_test)
        test_loader = DataLoader(test_dataset, batch_size=100000000, shuffle=False)
        X_test, Y_test, ind_test = None,None,None
        for i,data in enumerate(test_loader):
            X_test, Y_test, ind_test = data
#             X_test = X_test.to(device)
#             Y_test = Y_test.to(device)
#             ind_test = ind_test.to(device)

        return [X, Y, ind], [X_val, Y_val, ind_val], [X_test, Y_test, ind_test]
    else:
        test_retvals = []
        prev_min = -1
        for char_panel_test, return_panel_test, ind_panel_test in test_data:
            test_dataset = NpDataset(char_panel_test, return_panel_test, ind_panel_test)
            test_loader = DataLoader(test_dataset, batch_size=100000000, shuffle=False)
            X_test, Y_test, ind_test = None,None,None
            for i,data in enumerate(test_loader):
                X_test, Y_test, ind_test = data
#                 X_test = X_test.to(device)
#                 Y_test = Y_test.to(device)
#                 ind_test = ind_test.to(device)
            test_retvals.append((X_test, Y_test, ind_test))
            
        
        return [X, Y, ind], [X_val, Y_val, ind_val], test_retvals


def get_premade_x_y_ind(train_data, val_data, test_data, test_partitions=None):
    
    train_individualFeature, train_return, train_mask = train_data
    train_data, train_returns, train_ind = get_gkx_x_y(train_individualFeature, train_return,
                                                       train_mask, train_individualFeature.shape[0],
                                                       0, None, None,None)
    
    val_individualFeature, val_return, val_mask = val_data
    val_data, val_returns, val_ind = get_gkx_x_y(val_individualFeature, val_return,
                                                 val_mask, val_individualFeature.shape[0], 0, None, None, None)
    
    if test_partitions is None:        
        test_individualFeature, test_return, test_mask = test_data
        test_data, test_returns, test_ind = get_gkx_x_y(test_individualFeature, test_return,
                                                        test_mask, test_individualFeature.shape[0], 0, None, None, None)
        test_retval = [test_data, test_returns, test_ind]
    else:
        test_retval = []
        test_individualFeature, test_return, test_mask, test_missing_counts = test_data
        for p in test_partitions:
            partition_test_mask = np.logical_and(np.logical_and(test_missing_counts <= p[1], 
                                                               test_missing_counts >= p[0]),
                                                 test_mask)
            test_data, test_returns, test_ind, _,_,_, _,_,_ = get_gkx_x_y(test_individualFeature, test_return,
                                                                            partition_test_mask,
                                                                          test_individualFeature.shape[0], 0, None, None,
                                                                          None)
            prev_min = p
            test_retval.append((test_data, test_returns, test_ind))
    
    return [train_data, train_returns, train_ind], [val_data, val_returns, val_ind], test_retval

def get_gkx_x_y(char_panel, return_panel, masks, num_months_train, num_months_val, dates, macro_data, extra_macros):
    
    M = char_panel.shape[2]
    kelly_char_panel_train = np.zeros((np.sum(masks[:num_months_train]), M))
    kelly_return_panel_train = np.zeros((np.sum(masks[:num_months_train]), 1))
    kelly_ind_panel_train = np.zeros((np.sum(masks[:num_months_train]), 3))
        
    ids = np.arange(char_panel.shape[1])
    curr = 0
    for t in range(num_months_train):
        
        t_count = np.sum(masks[t]) + curr
        kelly_char_panel_train[curr:t_count, :M] = char_panel[t][masks[t], :]
        
        kelly_return_panel_train[curr:t_count] = np.expand_dims(return_panel[t, masks[t]], axis=1)
        kelly_ind_panel_train[curr:t_count, 0] = ids[masks[t]]
        kelly_ind_panel_train[curr:t_count, 1] = t
        kelly_ind_panel_train[curr:t_count, 2] = t_count - curr
        curr = t_count
    

    return kelly_char_panel_train, kelly_return_panel_train, kelly_ind_panel_train


def get_data(chars, min_chars_present, gamma_ts=None, fill_val=-1, ordered=False,
            oos_mask=None, oos_eval_data=None, return_panel=None, t_start=0, min_length=20):
    if ordered:
        in_sample = np.logical_or(np.all(~np.isnan(chars), axis=2),
                                  np.argmax(np.isnan(chars), axis=2) >= min_chars_present)
#         in_sample = np.sum(~np.isnan(chars), axis=2) >= min_chars_present
        print(np.sum(in_sample))
    else:
        in_sample = np.sum(~np.isnan(chars), axis=2) >= min_chars_present
    if return_panel is not None:
        in_sample = np.logical_and(in_sample, ~np.isnan(return_panel))
    if gamma_ts is not None:
        in_sample = np.logical_and(in_sample, np.all(~np.isnan(gamma_ts), axis=2))
        
    starts = np.zeros(in_sample.shape[1], dtype=int)
    in_curr_sample = np.zeros(in_sample.shape[1], dtype=bool)
    samples = []
    for t in tqdm(range(chars.shape[0])):
        next_in_sample = np.copy(in_sample[t])
        to_add = np.logical_and(in_curr_sample, ~next_in_sample)
        for idx in np.argwhere(to_add):
            idx = idx[0]
            if t-starts[idx] >= min_length:
                sample = np.copy(chars[starts[idx]:t, idx, :])*.99 + 0.5
                if return_panel is not None:
                    returns = return_panel[starts[idx]:t, idx]
                else:
                    returns = None
                if oos_mask is None:
                    oos_sample, oos_sample_mask = None, None
                else:
                    oos_sample = np.copy(oos_eval_data[starts[idx]:t, idx, :])*.99 + 0.5
                    oos_sample_mask = oos_mask[starts[idx]:t, idx, :]
                mask = np.isnan(sample)
                
                ordered_mask = np.copy(mask)
                for t_2 in range(mask.shape[0]):
                    if np.any(mask[t_2]):
                        assert (not ordered) or np.argmax(mask[t_2]) >= min_chars_present, (np.argmax(mask[t_2]), min_chars_present,
                                                                        np.sum(~mask[t_2]),
                                                                        )
                        ordered_mask[t_2,np.argmax(mask[t_2]):] = 1
                
                sample[np.isnan(sample)] = fill_val
                if np.sum(ordered_mask) > 0:
                    assert (not ordered) or np.min(sample[~ordered_mask]) > 0
                if gamma_ts is not None:
                    factors = gamma_ts[starts[idx]:t, idx, :]
                    samples.append((starts[idx] + t_start, t+t_start, idx, sample, mask, 
                                    ordered_mask, factors, oos_sample, oos_sample_mask,
                                   returns))
                else:
                    samples.append((starts[idx] + t_start, t+t_start, idx, sample, mask, 
                                    ordered_mask, oos_sample, oos_sample_mask, returns))
        starts[np.logical_and(~in_curr_sample, next_in_sample)] = t
        in_curr_sample = next_in_sample
    return samples


def get_singleton_data(chars, min_chars_present, gamma_ts=None, fill_val=-1, triangular_mask=False):
    in_sample = np.sum(~np.isnan(chars), axis=2) >= min_chars_present
    samples = []
    for t in tqdm(range(chars.shape[0])):
        to_add = np.copy(in_sample[t])
        for idx in np.argwhere(to_add):
            idx = idx[0]
            sample = np.copy(chars[t, idx, :])
            mask = np.isnan(sample)
            if triangular_mask and np.any(mask):
                first_nan = np.argmax(mask)
                mask[first_nan:] = 1
            if np.any(~mask):
                sample[np.isnan(sample)] = fill_val
                if gamma_ts is not None:
                    factors = gamma_ts[starts[idx]:t, idx, :]
                    samples.append((t, t, idx, sample, mask, factors))
                else:
                    samples.append((t, t, idx, sample, mask))
    return samples


class ListDatasetWithFactors(Dataset):
    def __init__(self, data, batch_size=None, return_oos_data=False, return_returns=False):
        self.return_oos_data = return_oos_data
        self.return_returns = return_returns
        if batch_size is None:            
            self.data = [x[3] for x in data]
            self.masks = [x[4] for x in data]
            self.ordered_mask = [x[5] for x in data]
            self.factors = [x[6] for x in data]
            self.oos_data = [x[7] for x in data]
            self.oos_data_masks = [x[8] for x in data]
            self.returns = [x[9] for x in data]
            self.time_idx = [np.array(list(range(x[0], x[1]))) for x in data]
            self.stock_idx = [np.array([x[2]]*len(x[9])) for x in data]
                             
        else:
            self.data = []
            self.masks = []
            self.ordered_mask = []
            self.factors = []
            self.oos_data = []
            self.oos_data_masks = []
            self.returns = []
            self.time_idx = []
            self.stock_idx = []
            for start_t, end_t, stock_idx, data, masks, ordered_mask, factors, oos_data, oos_mask, returns in data:
                idx = 0
                while idx + batch_size <= min([data.shape[0], masks.shape[0], ordered_mask.shape[0], factors.shape[0]]):
                    self.data.append(data[idx:idx+batch_size])
                    self.masks.append(masks[idx:idx+batch_size])
                    self.ordered_mask.append(ordered_mask[idx:idx+batch_size])
                    self.factors.append(factors[idx:idx+batch_size])
                    self.time_idx.append(np.array(list(range(0, batch_size))) + start_t + idx)
                    self.stock_idx.append(np.array([stock_idx]*batch_size))
                    if returns is not None:
                        self.returns.append(returns[idx:idx+batch_size])
                    else:
                        self.returns.append(returns)
                    if self.return_oos_data:
                        self.oos_data.append(oos_data[idx:idx+batch_size])
                        self.oos_data_masks.append(oos_mask[idx:idx+batch_size])
                    idx += batch_size
    def __len__(self): 
        return len(self.data)
    def __getitem__(self, i): 
#         print(self.data[i].shape, self.masks[i].shape, self.ordered_mask[i].shape, self.factors[i].shape)
        if self.return_oos_data and self.return_returns:
            return self.time_idx[i], self.stock_idx[i], self.data[i], self.masks[i], \
                    self.ordered_mask[i], self.factors[i], self.oos_data[i], \
                        self.oos_data_masks[i], self.returns[i]
        if self.return_oos_data:
            return self.time_idx[i], self.stock_idx[i], self.data[i], self.masks[i], \
                    self.ordered_mask[i], self.factors[i], self.oos_data[i], self.oos_data_masks[i]
        elif self.return_returns:
            return self.time_idx[i], self.stock_idx[i], self.data[i], self.masks[i],\
                    self.ordered_mask[i], self.factors[i], self.returns[i]
        return self.time_idx[i], self.stock_idx[i], self.data[i], self.masks[i], \
                    self.ordered_mask[i], self.factors[i]

    
class ListDataset(Dataset):
    def __init__(self, data, batch_size=None):
        if batch_size is None:            
            self.data = [x[3] for x in data]
            self.masks = [x[4] for x in data]
        else:
            self.data = []
            self.masks = []
            for _, _, _, data, masks in data:
                idx = 0
                while idx + batch_size <= len(data):
                    self.data.append(data[idx:idx+batch_size])
                    self.masks.append(masks[idx:idx+batch_size])
                    idx += batch_size
    def __len__(self): 
        return len(self.data)
    def __getitem__(self, i): 
        return self.data[i], self.masks[i]
