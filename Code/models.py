import numpy as np
import csv
import pandas as pd
import torch
import torch.nn as nn



class CharLSTM(nn.Module):

    #input dim: dimension size of hidden state
    #num_layer: number of lstms in stack
    def __init__(self, input_dim=45, hidden_dim=6, batch_size=1,
                    num_layers=1, output_dim=45, **kwargs):
        super(CharLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.alpha_lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout=0.1)
        self.alpha_out = torch.nn.Linear(self.hidden_dim, output_dim, bias=True)
        
        self.alpha_scale = 10
        self.beta_scale = 10
    
    def get_blank_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device),
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device))
    
    #assume that entire match is fed in in each input, initialize hidden state and propogate match 
    def forward(self, input):
        hidden = self.get_blank_hidden()
        params_out, (h_n, c_n) = self.alpha_lstm(input.view(len(input), self.batch_size, -1), hidden)
        print(params_out.shape, h_n.shape, c_n.shape)
        params_out = torch.sigmoid(self.alpha_out(params_out))
        alpha_out = params_out[:,:,:45]
        beta_out = params_out[:,:,45:]
#         print(params_out.shape)
        return alpha_out * self.alpha_scale, beta_out * self.beta_scale


class XSCharLSTM(nn.Module):

    def __init__(self, input_dim=45+6, hidden_dims=[6], batch_size=1,
                    num_layers=1, **kwargs):
        super(XSCharLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        input_dims = [input_dim] + hidden_dims[:-1]
        self.alpha_lstms = torch.nn.ModuleList([nn.LSTM(i_d, h_d, 1, dropout=0.01) 
                                                   for i_d, h_d in zip(input_dims, hidden_dims)])
        self.blank_hiddens = torch.nn.ParameterList([self.get_blank_hidden(h) for h in hidden_dims])
        self.blank_cells = torch.nn.ParameterList([self.get_blank_hidden(h) for h in hidden_dims])
        self.alpha_out = torch.nn.Linear(self.hidden_dims[-1], 45*2, bias=True)
        nn.init.xavier_normal_(self.alpha_out.weight)
        
        self.alpha_scale = 100
        self.beta_scale = 100
    
    def get_blank_hidden(self, h):
        return nn.Parameter(torch.zeros(1, self.batch_size, h, requires_grad=True).to(self.device))
    
    def forward(self, input):
        hidden_out = input.view(len(input), self.batch_size, -1)
        for i, alpha_lstm in enumerate(self.alpha_lstms):
            hidden = self.blank_hiddens[i]
            cell = self.blank_cells[i]
            hidden_out, (h_n, c_n) = alpha_lstm(hidden_out, (hidden, cell)) 
        params_out = torch.sigmoid(self.alpha_out(hidden_out))
        alpha_out = params_out[:,:,:45]
        beta_out = params_out[:,:,45:]
        return alpha_out * self.alpha_scale, beta_out * self.beta_scale, hidden_out



# Credit: Ameya Daigavane - https://github.com/ameya98/DeepGenerativeModels/blob/master/deepgenmodels

import torch
import torch.nn as nn
import torch.distributions as dist

class Beta_NADE(nn.Module):

    def __init__(self, inp_dimensions=45, latent_dimensions=5, hidden_state_dim=0, predict_returns=False):
        
        super(Beta_NADE, self).__init__()
        self.predict_returns = predict_returns
        if predict_returns:
            return_add = 1
            self.return_means = nn.Linear(latent_dimensions, 1)
            self.return_vars = nn.Linear(latent_dimensions, 1)
        else:
            return_add = 0
        
        self.inp_dimensions = inp_dimensions + hidden_state_dim
        self.latent_dimensions = latent_dimensions
        self.hidden_state_dim = hidden_state_dim
        self.output_dmensions = inp_dimensions
        
        self.hidden_beta = nn.Linear(inp_dimensions + hidden_state_dim, latent_dimensions)
        nn.init.xavier_normal_(self.hidden_beta.weight)
        
        self.alpha_weights_beta = torch.nn.parameter.Parameter(torch.rand(inp_dimensions, latent_dimensions, requires_grad=True))
        nn.init.xavier_normal_(self.alpha_weights_beta)
        self.alpha_bias_beta = torch.nn.parameter.Parameter(torch.rand(inp_dimensions, requires_grad=True))
        self.beta_scale = 100
        
        self.alpha_weights_alpha = torch.nn.parameter.Parameter(torch.rand(inp_dimensions, latent_dimensions, requires_grad=True))
        nn.init.xavier_normal_(self.alpha_weights_alpha)
        self.alpha_bias_alpha = torch.nn.parameter.Parameter(torch.rand(inp_dimensions, requires_grad=True))
        self.alpha_scale = 100
        

        self.sum_matrix = torch.ones(inp_dimensions + return_add,
                                     inp_dimensions + hidden_state_dim,
                                     requires_grad=False)
        
        for rownum, row in enumerate(self.sum_matrix):
            row[rownum+hidden_state_dim:] = 0
        
        self.sum_matrix = torch.nn.parameter.Parameter(self.sum_matrix)
        print(self.sum_matrix, self.sum_matrix.shape)
    
    def shape_vectors(self, x, get_hidden_state=False):
        x_diag = torch.stack([torch.diag(x_j) for x_j in x])

        beta_dot_products = self.hidden_beta(x_diag)
        hidden_beta_activations = torch.sigmoid(torch.matmul(self.sum_matrix, beta_dot_products))
        if get_hidden_state:
            return hidden_beta_activations[:,-1,:]
        beta_out= torch.sigmoid(torch.sum(torch.mul(hidden_beta_activations[:,:self.output_dmensions,:],
                                                    self.alpha_weights_beta), dim=2)
                             + self.alpha_bias_beta)
        
        alpha_out = torch.sigmoid(torch.sum(torch.mul(hidden_beta_activations[:,:self.output_dmensions,:],
                                                      self.alpha_weights_alpha), dim=2)
                             + self.alpha_bias_alpha)
        
        if self.predict_returns:
            return_activations = hidden_beta_activations[:,-1,:]
            return_means = self.return_means(return_activations)
            return_vars = torch.exp(self.return_vars(return_activations))
            return alpha_out * self.alpha_scale, beta_out * self.beta_scale, return_means, return_vars
        
        return alpha_out * self.alpha_scale, beta_out * self.beta_scale

    
    def forward(self, x):
        return self.shape_vectors(x)
    
    
    def impute_alphas_betas(self, x):
        x_copy = torch.clone(x)
        assert torch.sum(x_copy == -1) > 0
        imputed_mask = x_copy[:,self.hidden_state_dim:] == -1
        num_missing = torch.max(torch.sum(imputed_mask, axis=1))

        for i in range(num_missing):
            if self.predict_returns:
                alphas, betas, ret_mean, ret_sigma = self.shape_vectors(x_copy)
            else:
                alphas, betas = self.shape_vectors(x_copy)
            pred = alphas / (alphas + betas)
            x_copy[:,self.hidden_state_dim:][imputed_mask] = pred[imputed_mask]

        assert torch.sum(x_copy[:,self.hidden_state_dim:] == -1) == 0
        if self.predict_returns:
            alphas, betas, ret_mean, ret_sigma = self.shape_vectors(x_copy)
            return alphas, betas, ret_mean
        else:
            alphas, betas = self.shape_vectors(x_copy)
            return alphas, betas
        
    def impute(self, x):
        if self.predict_returns:
            alphas, betas, ret_mean = self.impute_alphas_betas(x)
            preds = alphas / (alphas + betas)
            return preds, ret_mean
        else:
            alphas, betas = self.impute_alphas_betas(x_copy)
            preds = alphas / (alphas + betas)
            return preds
    
    def get_hidden_state(self, x):
        x_copy = torch.clone(x)
        assert torch.sum(x_copy == -1) > 0
        imputed_mask = x_copy[:,self.hidden_state_dim:] == -1
        num_missing = torch.max(torch.sum(imputed_mask, axis=1))

        for i in range(num_missing):
            if self.predict_returns:
                alphas, betas, ret_mean, ret_sigma = self.shape_vectors(x_copy)
            else:
                alphas, betas = self.shape_vectors(x_copy)
            pred = alphas / (alphas + betas)
            x_copy[:,self.hidden_state_dim:][imputed_mask] = pred[imputed_mask]
            
        return self.shape_vectors(x_copy, get_hidden_state=True)