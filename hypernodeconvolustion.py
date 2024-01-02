# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 10:53:07 2023

@author: Raneen_new
"""

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class HyperNodeConvolution(nn.Module):
    def __init__(self, timeseries_dim, output_dim,device):
        super(HyperNodeConvolution, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=timeseries_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=timeseries_dim, out_channels=1, kernel_size=3, padding=1)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=output_dim, kernel_size=1).to(self.device)
        self.batch_norm_1 = nn.BatchNorm1d(timeseries_dim).to(self.device)
        self.batch_norm_2 = nn.BatchNorm1d(1).to(self.device)
        self.relu = nn.ReLU().to(self.device)
        self.output_dim = output_dim

    def forward(self, node_features, edge_features):
        # node_features shape: [batch, edges, nodes, timeseries_len, timeseries_dim]
        # edge_features shape: [batch, edges, timeseries_len]
        node_features = node_features.to(self.device)
        edge_features = edge_features.to(self.device)
        batch_size, edges, nodes, timeseries_len, _ = node_features.size()
        output_dim = self.conv3.out_channels

        # Placeholder for the updated node features
        updated_node_features = torch.zeros(batch_size, edges, nodes, self.output_dim,edge_features.shape[2])

        for batch in range(batch_size):
            for edge in range(edges):
                for node in range(nodes):
                    # Find all edges in the current batch sample that share the same node
                    connected_edges = self.find_connected_edges(batch, edge, node, node_features)

                    # Process through convolutional layers
                    x = self.conv1(connected_edges)
                    x = self.batch_norm_1(x)
                    x = self.relu(x)

                    x = self.conv2(x)
                    x = self.batch_norm_2(x)
                    x = self.relu(x)

                    x = self.conv3(x)
                    # Sum over the T_i to get the new representation for the node
                    #x = self.conv3(aggregated_edge_features)
                    x_n = x.sum(dim=0)
                    # Update the representation for the node
                    updated_node_features[batch, edge, node, :] = x_n

        # Reshape to match the desired output shape
        return updated_node_features

    def find_connected_edges(self, batch,edge, target_node, node_features):
        # node_features shape: [batch, edges, nodes, timeseries_len, timeseries_dim]
        edges = node_features.size(1)
        nodes = node_features.size(2)
        # Get the feature vector for the target node
        target_node_feature = []
        # Initialize a list to store indices of connected edges
        connected_edges = []

        # Iterate over all edges and nodes to find matches
        for edge in range(edges):
            for node in range(nodes):
                # Compare feature vector of the current node with the target node
                if torch.equal(target_node_feature, node_features[batch, edge, node, :, :]):
                    connected_edges.append(edge)
                    break  # Break if a match is found in the current edge
        return connected_edges


    def aggregate_edge_features(self, batch, connected_edges, edge_features):
        # Initialize aggregated features with zeros
        timeseries_len = edge_features.size(2)
        aggregated_features = torch.zeros(timeseries_len, dtype=edge_features.dtype, device=self.device)

        # Sum the features of the connected edges
        for edge in connected_edges:
            aggregated_features += edge_features[batch, edge]

        # Reshape to make it suitable for the convolution operation
        aggregated_features = aggregated_features.view(1, -1, timeseries_len)

        return aggregated_features
