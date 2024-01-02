# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 17:00:46 2023

@author: Raneen_new
"""
import numpy as np
import time
import numpy as np
import sys, getopt
import pandas
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import collections
import time
import pickle
import networkx as nx
import pandas as pd
#from pyts.transformation import ShapeletTransform
import bisect
import hypernetx as hnx
import copy
np.random.seed(0)
"""
Seg_data:This function is useful in certain types of data analysis where you want to study patterns or features 
that occur over multiple scales (both within windows and across segments). For example, in time series analysis or signal 
processing, you might want to look at how certain features or patterns evolve over time both on short timescales 
(within a window) and on longer timescales (across multiple windows or segments).
"""
def Seg_data(data,s,w):
    """
    Segments a ND numpy array of data into 's' segments, each of which contains 'w' windows.

    Parameters:
    data: A ND numpy array.
    s: The desired number of segments per data sample.
    w: The desired number of windows per segment.

    The function first calculates the window length and total number of windows 
    based on the input parameters. Then, for each data sample, it divides the data 
    into windows of the calculated length. It further segments this windowed data 
    into 's' segments, each of which contains 'w' windows. The segmented data for 
    each sample is added to a list.

    Returns:
    A list where each element corresponds to a data sample. Each element is a list 
    of segments, each of which is a list of windows. 
    """
    ##split the data into n segments, and for each segment split to w window
    ##for each data sample return s segment [[[],[],...,w],.....,s] s segments of size w
    window_len = int((data.shape[1]/s)/w)
    if(window_len ==0):
        window_len = 5
    window_num= int(data.shape[1]/window_len)
    segmant = []
    for d in range(len(data)):
        x = 0
        wind = []
        while(x < (data.shape[1])):
            dd =[]
            if(x+window_len<(data.shape[1])):
                dd = data[d][x:x+window_len:]
                wind.append(dd)
            x+=window_len 
                    
        wind = np.asarray(wind)
        seg = []
        idx = 0
        seg_data = []
        for wi in wind:
            if(idx == int(len(wind)/s)):
                idx = 0
                seg_data.append(seg)
                seg = []
            seg.append(wi)
            idx +=1
        segmant.append(seg_data)
    return segmant

def calc_correlation(actual, predic):
    """
    Calculate and return the Pearson correlation coefficient between two arrays (windows).

    The Pearson correlation coefficient measures the linear relationship between two datasets.
    The coefficient ranges from -1 to 1. A value of 1 implies a perfect positive correlation,
    a value of -1 implies a perfect negative correlation, and a value of 0 implies no correlation.

    Parameters:
    actual: A numpy array for the actual values.
    predic: A numpy array for the predicted values.

    Returns:
    A float number representing the Pearson correlation coefficient between actual and predic.
    """
    a_diff = actual - np.mean(actual)
    p_diff = predic - np.mean(predic)
    numerator = np.sum(a_diff * p_diff)
    denominator = np.sqrt(np.sum(a_diff ** 2)) * np.sqrt(np.sum(p_diff ** 2))
    return numerator / denominator

def calc_correlation_nodeslist(actual, node_dict):
    """
    Calculate the maximum absolute correlation between 'actual' window and the nodes in 'node_dict'.
    This function calls 'calc_correlation' to find the correlation between 'actual' and each array 
    in 'node_dict', and tracks the maximum absolute correlation value.

    Parameters:
    actual: A numpy array for the actual values.
    node_dict: A dictionary where the key is the node identifier, and the value is a numpy array 
               representing node data.

    Returns:
    max_value: The maximum absolute correlation value found among the nodes.
    selected_key: The key of the node that has the maximum absolute correlation with the 'actual' array.
    """
    max_value = 0
    selected_key = ''
    for node_key, node_value in node_dict.items():
        value = calc_correlation(actual, node_value)
        abs_value = abs(value)
        if abs_value > max_value:
            max_value = abs_value
            selected_key = node_key
    return max_value, selected_key

def snapshot_hypergraph_samples(segment, T, return_connected_graph=True):
    """
    Transforms the given time-series data into a set of hypergraphs and a set of connected hypergraphs.
    
    Parameters:
    segment: A nested list where each sample is divided into segments and each segment contains multiple windows.
    T: A threshold value for the correlation coefficient.
    return_connected_graph: If True, the function will also return the set of connected hypergraphs.
    
    Returns:
    hypergraphs: A list of hypergraphs. Each hypergraph is a dictionary where keys are hyperedge names 
                 and values are lists of node names.
    nodes_dict: A dictionary where keys are node names and values are the corresponding windows of data.
    connected_hypergraphs: A list of connected hypergraphs (only returned if return_connected_graph is True).
                           Each connected hypergraph is a dictionary where keys are hyperedge names and values 
                           are lists of connected node names.
    """
    num_samples = len(segment)  # Number of data samples
    num_segments = len(segment[0])  # Number of segments per sample
    hypergraphs = []  # List to store hypergraphs for each data sample
    connected_hypergraphs = []  # List to store connected hypergraphs
    
    nodes_dict = {}  # Dictionary to store node names and content
    ind = 0
    for sample_data in segment:
        seprated_hyper_graph = {}
        edge_index = 0
        connected_hyper_graph = {}
        last_node = ''
        #loop through the segments in data sample
        for s in range(num_segments):
            window_nodes = []
            first_node = ''
            i = 0
            #loop through the windows in data sample segment [s]
            for window in sample_data[s]:
                node_name = ''
                # Create first node in graph
                if not nodes_dict:
                    node_name = f"n0"
                    nodes_dict[node_name] = window
                    window_nodes.append(node_name)
                else:
                    # Find nodes similarities through correlation
                    value, node = calc_correlation_nodeslist(window, nodes_dict)
                    if value < T:
                        node_index = len(nodes_dict)
                        node_name = f"n{node_index}"
                        nodes_dict[node_name] = window
                    else:
                        node_name = node
                    window_nodes.append(node_name)
                
                if i == 0:
                    first_node = node_name
                i += 1
            
            #seprated_hyper_graph.append((window_nodes))
            edge_name = f"e{edge_index}"
            seprated_hyper_graph[edge_name] = window_nodes
            if return_connected_graph:
                if last_node != '':
                    #connected_hyper_graph.append([last_node, first_node])
                    connected_hyper_graph[edge_name] = [last_node, first_node]
            last_node = node_name
            #connected_hyper_graph.append((window_nodes))
            connected_hyper_graph[edge_name] = window_nodes
            edge_index +=1
        hypergraphs.append(seprated_hyper_graph)
        connected_hypergraphs.append(connected_hyper_graph)
        print(ind)
        ind += 1

    if return_connected_graph:
        return hypergraphs, nodes_dict, connected_hypergraphs
    else:
        return hypergraphs, nodes_dict
    
def adjacency_matrix(H):
    """
    This function takes a graph 'H' as an input, where 'H' is represented as a network graph object. 
    It constructs and returns an adjacency matrix of this graph. The adjacency matrix is a square matrix used to represent 
    a finite graph. 
    The elements of the matrix indicate whether pairs of vertices are adjacent or not in the graph. 
    In this function, the adjacency matrix is represented as a pandas DataFrame, where the index and columns are the nodes 
    (vertices) of the graph 'H'. If there is an edge between two vertices, the corresponding cell in the DataFrame is set to 1, 
    otherwise, it is 0.
    
    Args:
    H: NetworkX graph object
    
    Returns:
    adj_matrix: DataFrame representing the adjacency matrix of the input graph
    """
    vertices = list(H.nodes)
    adj_matrix = pd.DataFrame(0, index=vertices, columns=vertices)
    for e in H.edges:
        for v1 in H.edges[e]:
            for v2 in H.edges[e]:
                if v1 != v2:
                    adj_matrix.loc[v1, v2] = 1
    return adj_matrix

def incidence_matrix(H):
    """
    This function constructs and returns the incidence matrix for a given graph 'H', 
    where 'H' is represented as a network graph object. The incidence matrix is a binary matrix that describes the 
    relationship between two classes of objects: vertices and edges. In this function, the incidence matrix is represented 
    as a pandas DataFrame, where the index is the nodes (vertices) of the graph and the columns are the edges. 
    If a vertex 'v' is part of an edge 'e', the corresponding cell in the DataFrame is set to 1, otherwise, it's 0.
    
    Args:
    H: NetworkX graph object
    
    Returns:
    inc_matrix: DataFrame representing the incidence matrix of the input graph
    """
    vertices = list(H.nodes)  # Define 'vertices' here as well
    edges = list(H.edges)
    inc_matrix = pd.DataFrame(0, index=vertices, columns=edges)  # Now 'vertices' is defined
    for e in H.edges:
        for v in H.edges[e]:
            inc_matrix.loc[v, e] = 1
    return inc_matrix

    vertices = list(H.nodes)
    edges = list(H.edges)
    inc_matrix = pd.DataFrame(0, index=vertices, columns=edges)
    
def data_adjacency_incidence_matrix(hypergraphs):
    """
    This function takes as input a list of hypergraphs, and for each hypergraph in the list, it computes both the 
    adjacency matrix and the incidence matrix. The function appends these matrices to respective lists and returns them.

    Args:
    hypergraphs: List of hypergraphs to be processed

    Returns:
    Tuple containing two lists:
    adj_matrix: List of adjacency matrices computed for each hypergraph
    inc_matrix: List of incidence matrices computed for each hypergraph
    """
    adj_matrix = []
    inc_matrix = []
    for i in hypergraphs:
        H = hnx.Hypergraph(i)
        adj_matrix.append(adjacency_matrix(H))
        inc_matrix.append(incidence_matrix(H))
    return adj_matrix,inc_matrix

def map_hypernode_name_vector(hypergraphs,nodes_name):
    hypergraphs_cp =  copy.deepcopy(hypergraphs)
    hypergraphs_new = []
    for hyperedges in hypergraphs_cp:
        for edge in hyperedges:
            hyperedges[edge] = [nodes_name[node] for node in hyperedges[edge]]
        hypergraphs_new.append(hyperedges)
    return hypergraphs_new