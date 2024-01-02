# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 10:50:23 2023

@author: Raneen_new
"""

import torch
import torch.nn as nn
from hypernodeembedding import HypernodeEmbeddingLayer
from hyperedgeembedding import HyperedgeEmbeddingLayer
from hypernodeconvolustion import HyperNodeConvolution
from hyperedgeconvolustion import HyperedgeConvolution
from classifier import Classifier
class HypergraphNeuralNetwork(nn.Module):
    def __init__(self, time_series_len, time_series_dim, lstm_hidden_dim, attention_dim, output_dim,new_feature_dim, num_classes,device):
        super(HypergraphNeuralNetwork, self).__init__()
        # Initialize all the layers
        self.hypernode_embedding_layer = HypernodeEmbeddingLayer(time_series_len, time_series_dim,device)
        self.hyperedge_embedding_layer = HyperedgeEmbeddingLayer(time_series_dim, lstm_hidden_dim, attention_dim,device)
        self.hypernode_con_layer = HyperNodeConvolution(time_series_dim, output_dim,device)
        self.hyperedge_conv = HyperedgeConvolution(new_feature_dim, lstm_hidden_dim, new_feature_dim,num_classes,device)
        #self.classifier = Classifier(num_classes)

    def forward(self, x):
        # Forward pass through each layer
        node_embed = self.hypernode_embedding_layer(x)
        edge_embed = self.hyperedge_embedding_layer(node_embed)
        node_conv = self.hypernode_con_layer(x, edge_embed)
        output = self.hyperedge_conv(node_conv)
        #output = self.classifier(edge_conv)
        return output