# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 10:52:33 2023

@author: Raneen_new
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperedgeEmbeddingLayer(nn.Module):
    def __init__(self, time_series_dim, lstm_hidden_dim, attention_dim, device=None):
        super(HyperedgeEmbeddingLayer, self).__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize layers and parameters on the device
        self.lstm = nn.LSTM(input_size=time_series_dim, hidden_size=lstm_hidden_dim, batch_first=True).to(self.device)
        self.attention_weights = nn.Parameter(torch.randn(attention_dim, lstm_hidden_dim)).to(self.device)

    def forward(self, x):
        x = x.to(self.device)  # Ensure input tensor is on the correct device

        batch_size, num_edges, time_series_len, _ = x.shape
    
        # LSTM and attention operations
        lstm_out, _ = self.lstm(x)  # lstm_out shape: [batch_size*num_edges, time_series_len, lstm_hidden_dim]
        attention_scores = torch.matmul(lstm_out, self.attention_weights.T)
        attention_weights = F.softmax(attention_scores, dim=1)
        weighted_lstm_out = torch.einsum('bti,btj->bti', attention_weights, lstm_out)

        # Post-processing to get edge vectors
        edge_vectors = torch.mean(weighted_lstm_out, dim=1)
        edge_vectors = edge_vectors.view(batch_size, num_edges, -1)

        return edge_vectors