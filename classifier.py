# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 10:54:55 2023

@author: Raneen_new
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(1, num_classes)  # The input feature size is 1 after summing

    def forward(self, x):
        # Sum over both the edge and feature dimensions
        x_summed = torch.sum(x, dim=(1, 2), keepdim=True)

        # Apply the linear layer
        logits = self.fc(x_summed)

        # Apply softmax to obtain probabilities
        #probabilities = F.softmax(logits, dim=1)

        return logits