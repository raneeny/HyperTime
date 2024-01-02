# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 10:51:45 2023

@author: Raneen_new
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class HypernodeEmbeddingLayer(nn.Module):
    def __init__(self, time_series_len, time_series_dim, device=None):
        super(HypernodeEmbeddingLayer, self).__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.theta = nn.Parameter(torch.randn(time_series_dim, time_series_len)).to(self.device)

    def forward(self, x):
        x = x.to(self.device)  # Ensure input is on the correct device

        # Compute attention weights
        attention_scores = torch.einsum('ijklm,ml->ijkl', x, self.theta)
        alpha = F.softmax(attention_scores, dim=-1)

        # Calculate the weighted sum
        weighted_sum = torch.einsum('ijklm,ijkl->ijkm', x, alpha)

        return weighted_sum