# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 10:53:47 2023

@author: Raneen_new
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperedgeConvolution(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_classes, device=None):
        super(HyperedgeConvolution, self).__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize layers on the device
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1).to(self.device)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1).to(self.device)
        self.conv3 = nn.Conv1d(hidden_dim, output_dim, kernel_size=3, padding=1).to(self.device)
        self.classifier = nn.Linear(output_dim, num_classes).to(self.device)
        self.num_classes = num_classes

    def forward(self, x):
        x = x.to(self.device) 
        # x shape: [batch_size, edge_number, node_number, timeseries, dimension]
        batch_size, edge_number, node_number, timeseries, _ = x.shape
        output = []

        # Iterate over each edge
        for edge in range(edge_number):
            # Process nodes of each edge
            edge_data = x[:, edge, :, :, :]  # shape: [batch_size, node_number, timeseries, dimension]
            
            # Apply Conv1D layers
            edge_data = self.conv1(edge_data)
            edge_data = torch.relu(edge_data)
            edge_data = self.conv2(edge_data)
            edge_data = torch.relu(edge_data)
            edge_data = self.conv3(edge_data)
            edge_data = torch.relu(edge_data)

            # Flatten the output
            edge_data = edge_data.view(batch_size, node_number, -1)
            edge_data = edge_data.mean(dim=1)  # Example of pooling

            # Append the result for this edge
            output.append(edge_data)

        # Concatenate output of all edges
        output = torch.cat(output, dim=1)
        # Flatten the output
        output = output.view(batch_size, -1)

        if self.classifier.in_features != output.size(1):
            self.classifier = nn.Linear(output.size(1), self.num_classes).to(self.device)

        output = self.classifier(output)

        return output