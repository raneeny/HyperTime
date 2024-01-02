#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""
Created on Mon Oct 16 20:23:30 2023

@author: Raneen_new
"""

from Data_Preprocessing import ReadData
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import time
import csv
#from pyts.transformation import ShapeletTransform
np.random.seed(0)
from Hypergraph_Construction import Seg_data,snapshot_hypergraph_samples,data_adjacency_incidence_matrix,map_hypernode_name_vector
from hyperneuralnetwork import HypergraphNeuralNetwork
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from aeon.datasets import load_classification
from sklearn.preprocessing import OneHotEncoder
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def readData(data_name,dir_name):
    """
    Reads in the dataset with the provided name from the specified directory. 
    The dataset can be in one of three formats: 'HAR', 'PAMAP2', or other.

    Parameters:
    data_name: A string, the name of the dataset to be read.
    dir_name: A string, the name of the directory where the dataset is stored.

    Returns:
    x_training: The training samples.
    x_validation: The validation samples.
    x_test: The test samples.
    y_training: The training labels.
    y_validation: The validation labels.
    y_true: The original (non-one-hot-encoded) test labels.
    y_test: The one-hot-encoded test labels.
    input_shape: The shape of the input data samples.
    nb_classes: The number of classes in the labels.
    """
    dir_path = dir_name + data_name+'/'
    dataset_path = dir_path + data_name +'.mat'
    
    ##read data and process it
    prepare_data = ReadData()
    if(data_name == "HAR"):
        dataset_path = dir_name + data_name +'/train.pt'
        x_training = torch.load(dataset_path)
        x_train = x_training['samples']
        y_train = x_training['labels']
        dataset_path = dir_name + data_name +'/train.pt'
        x_testing = torch.load(dataset_path)
        x_test = x_testing['samples']
        y_test = x_testing['labels']
        x_train = x_train.cpu().detach().numpy()
        y_train = y_train.cpu().detach().numpy()
        x_test = x_test.cpu().detach().numpy()
        y_test = y_test.cpu().detach().numpy()
        #reshape array(num_sample,ts_len,dim)
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[2], x_train.shape[1])
        x_test = x_test.reshape(x_test.shape[0],x_test.shape[2], x_test.shape[1])
    elif(data_name == "PAMAP2"):
        dataset_path = dir_name + data_name +'/PTdict_list.npy'
        x_data = np.load(dataset_path)
        dataset_path = dir_name + data_name +'/arr_outcomes.npy'
        y_data = np.load(dataset_path)
        split_len = int(len(x_data)*0.9)
        x_train,x_test  = x_data[:split_len,:], x_data[split_len:,:]
        y_train,y_test  = y_data[:split_len,:], y_data[split_len:,:]
        
    else:
        prepare_data.data_preparation(dataset_path, dir_path)
        datasets_dict = prepare_data.read_dataset(dir_path,data_name)
        x_train = datasets_dict[data_name][0]
        y_train = datasets_dict[data_name][1]
        x_test = datasets_dict[data_name][2]
        y_test = datasets_dict[data_name][3]
        x_train, x_test = prepare_data.z_norm(x_train, x_test)
    
    nb_classes = prepare_data.num_classes(y_train,y_test)
    y_train, y_test, y_true = prepare_data.on_hot_encode(y_train,y_test)
    x_train, x_test, input_shape = prepare_data.reshape_x(x_train,x_test)
    x_training = x_train
    y_training = y_train
 
    x_new1 = np.concatenate((x_train, x_test), axis=0)
    y_new1 = np.concatenate((y_train, y_test), axis=0)
    x_training, x_validation, y_training, y_validation = train_test_split(x_new1, y_new1, test_size=0.20,shuffle=True)
    x_validation,x_test,y_validation,y_test = train_test_split(x_validation, y_validation, test_size=0.50,shuffle=True)
    print(x_training.shape)
    print(x_validation.shape)
    print(x_test.shape)
    return x_training, x_validation, x_test, y_training, y_validation, y_true,y_test, input_shape,nb_classes

def train_loop(dataloader, val_dataloader,batch_size, nb_classes, num_epochs, use_case,lr,device):
    time_series_len = 0
    time_series_dim = 0
    lstm_hidden_dim = 50
    attention_dim = 20
    output_dim = 64
    new_feature_dim = 64

    # Define a dictionary mapping use_case numbers to respective classes
    model_classes = {
        0: HypergraphNeuralNetwork,
        #1: HypergraphNeuralNetwork1,
        #2: HypergraphNeuralNetwork2,
        # Add more mappings as needed...
    }

    # Initialize the model based on the use_case
    if use_case in model_classes:
        model_class = model_classes[use_case]
    else:
        model_class = model_classes[0]

    for data in dataloader:
        time_series_len = data[0].shape[3]
        time_series_dim = data[0].shape[4]

    # Create an instance of the selected model class
    model = model_class(time_series_len, time_series_dim, lstm_hidden_dim, attention_dim, output_dim, new_feature_dim, nb_classes,device)
    #model = HypergraphNeuralNetwork(time_series_len, time_series_dim, lstm_hidden_dim, attention_dim, output_dim,new_feature_dim, num_classes)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss, total, correct = 0, 0, 0
    
        for data in dataloader:
            inputs = data[0].to(dtype=torch.float32, device=device)
            labels = data[1].to(device=device)
        
            # Assuming data[1] is your label tensor
            labels = torch.argmax(labels, dim=1) if labels.ndim > 1 else labels
        
            outputs = model(inputs)
            loss = criterion(outputs, labels)
    
            # Compute predictions and update correct/total counts
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        total_loss /= len(dataloader)
        accuracy = 100 * correct / total  # Calculate training accuracy
    
        # Validation loop
        #model.eval()  # Set model to evaluation mode
        val_loss, val_total, val_correct = 0, 0, 0
        
        with torch.no_grad():
            for data in val_dataloader:
                inputs = data[0].to(dtype=torch.float32, device=device)  # Move inputs to the correct device
                labels = data[1].to(device=device)  # Move labels to the correct device
        
                # Adjust labels format if necessary
                labels = torch.argmax(labels, dim=1) if labels.ndim > 1 else labels
        
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_loss += loss.item()
    
        val_loss /= len(val_dataloader)
        val_accuracy = 100 * val_correct / val_total
    
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%, Validation Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')


    return model

def test_loop(dataloader, model, device, data_name, results_file='results_multi_variate.csv'):
    #model.eval()  # Set the model to evaluation mode
    y_pred = []
    y_true = []

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data[0].to(device, dtype=torch.float32), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')

    # Write results to a CSV file
    with open(results_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Check if the file is empty to write headers
        if file.tell() == 0:
            writer.writerow(['Dataset Name', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
        writer.writerow([data_name, accuracy, precision, recall, f1])

        
def run(data_name, univariate, read_data, window_num, edge_num, num_epochs, batch_size,use_case,lr):
    """
    if(HAR or PAM):
        #dir_name = '/home/younis/Dynamic_HyperGraph/data/mtsdata/'
        dir_name = '../../../../MTS2Graph/MTS2Graph/data/mtsdata/'
        x_training, x_validation, x_test, y_training, y_validation, y_true,y_test, input_shape,nb_classes = readData(data_name,dir_name)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(data_name)
    start_time = time.time()
    x_training, y_training,_ = load_classification(data_name)
    x_training = x_training.reshape(x_training.shape[0], x_training.shape[2], x_training.shape[1])
    nb_classes = len(np.unique(y_training, axis=0))
    ###convert OneHotEncoder
    enc = OneHotEncoder()
    enc.fit(y_training.reshape(-1, 1))
    y_training = enc.transform(y_training.reshape(-1, 1)).toarray()
  
    print("Read data--- %s seconds ---" % (time.time() - start_time))
    #merge the data as one set and apply the segmentation and the hypergraph to the whole dataset
    # Assuming x_training, x_validation, and x_test are your data arrays
    # Concatenate along the first axis (usually the batch axis)
    #x_total = np.concatenate((x_training, x_validation, x_test), axis=0)
    #x_total=x_training
    # Similarly for y_training, y_validation, and y_test
    start_time = time.time()
    #here x_total
    #Seg_data(data,s,w) 's' segments, each of which contains 'w' windows
    segment = Seg_data(x_training,edge_num,window_num)
    print("Segment data--- %s seconds ---" % (time.time() - start_time))
    hypergraphs = []
    nodes_name = []
    #inc_matrix  = []
    if(read_data):
        infile = '../models/'+data_name+'/'+'hypergraphs'+'.npy'
        hypergraphs = np.load(infile,allow_pickle=True)
        infile = '../models/'+data_name+'/'+'nodes_name'+'.npy'
        nodes_name = np.load(infile,allow_pickle=True).item()
        infile = '../models/'+data_name+'/'+'connected_hypergraphs'+'.npy'
        connected_hypergraphs = np.load(infile,allow_pickle=True)
        #infile = 'models/'+data_name+'/'+'inc_matrix_hypergraphs'+'.npy'
        #inc_matrix = np.load(infile, allow_pickle=True)
    else:   
        start_time = time.time()
        hypergraphs,nodes_name,connected_hypergraphs=snapshot_hypergraph_samples(segment,0.01,True)
        #hypergraphs,nodes_name,connected_hypergraphs=snapshot_hypergraph_DTW(segment,0.01,True)
        #outfile = 'models/'+data_name+'/'+'hypergraphs'+'.npy'
        #np.save(outfile, hypergraphs)
        #outfile = 'models/'+data_name+'/'+'nodes_name'+'.npy'
        #np.save(outfile, nodes_name)
        #outfile = 'models/'+data_name+'/'+'connected_hypergraphs'+'.npy'
        #np.save(outfile, connected_hypergraphs)
        #adj_matrix,inc_matrix = data_adjacency_incidence_matrix(hypergraphs)
        #outfile = 'models/'+data_name+'/'+'inc_matrix_hypergraphs'+'.npy'
        #np.save(outfile, inc_matrix)
    
    hypergraphs_new =  map_hypernode_name_vector(connected_hypergraphs,nodes_name)
    hypergraphs_new_list = [list(d.values()) for d in hypergraphs_new]
    
    x_trainig, x_test, y_trainig, y_test = train_test_split(hypergraphs_new_list, y_training, test_size=0.1, random_state=42)
    x_trainig, x_val, y_trainig, y_val = train_test_split(x_trainig, y_trainig, test_size=0.1, random_state=42)
    
    tensor_data = [torch.tensor(sequence) for sequence in x_trainig]
    #tensor_data = [torch.tensor(sequence) for sequence in hypergraphs_new_list]
    # Convert list of tensors to a single 3D tensor for DataLoader
    tensor_data = torch.stack(tensor_data)
    #labels = np.concatenate((y_training, y_validation, y_test), axis=0)
    labels = y_trainig
    tensor_labels = torch.tensor(labels)
    print(tensor_data.shape)
    print(tensor_labels.shape)
    dataset = TensorDataset(tensor_data, tensor_labels)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    
    ##validation loader 
    tensor_data_test = [torch.tensor(sequence) for sequence in x_val]
    # Convert list of tensors to a single 3D tensor for DataLoader
    tensor_data_test = torch.stack(tensor_data_test)
    #labels = np.concatenate((y_training, y_validation, y_test), axis=0)
    labels_test = y_val
    tensor_labels_test = torch.tensor(labels_test)
    tensor_labels_test = torch.argmax(tensor_labels_test, dim=1) if tensor_labels_test.ndim > 1 else tensor_labels_test
    dataset_test = TensorDataset(tensor_data_test, tensor_labels_test)
    val_dataloader = DataLoader(dataset_test, batch_size, shuffle=True)
    ##train  the model
    model=train_loop(dataloader,val_dataloader,batch_size,nb_classes,num_epochs,use_case,lr,device)
    ##get the test phase
    #segment_test = Seg_data(x_test,edge_num,window_num)
    #hypergraphs_test,nodes_name_test,connected_hypergraphs_test=snapshot_hypergraph_samples(segment_test,0.01,True)
    #hypergraphs_new_test =  map_hypernode_name_vector(connected_hypergraphs_test,nodes_name_test)
    #hypergraphs_test = [list(d.values()) for d in hypergraphs_new_test]
    tensor_data_test = [torch.tensor(sequence) for sequence in x_test]
    # Convert list of tensors to a single 3D tensor for DataLoader
    tensor_data_test = torch.stack(tensor_data_test)
    #labels = np.concatenate((y_training, y_validation, y_test), axis=0)
    labels_test = y_test
    tensor_labels_test = torch.tensor(labels_test)
    tensor_labels_test = torch.argmax(tensor_labels_test, dim=1) if tensor_labels_test.ndim > 1 else tensor_labels_test
    dataset_test = TensorDataset(tensor_data_test, tensor_labels_test)
    test_dataloader = DataLoader(dataset_test, batch_size, shuffle=True)
    test_loop(test_dataloader, model, device,data_name)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the model with specified parameters.')
    parser.add_argument('--data_name', type=str, default='ECG', help='Name of the dataset')
    parser.add_argument('--univariate', action='store_true', help='Flag for univariate data')
    parser.add_argument('--read_data', action='store_true', help='Flag for reading data')
    parser.add_argument('--window_num', type=int, default=3, help='Number of windows')
    parser.add_argument('--edge_num', type=int, default=7, help='Number of edges')
    parser.add_argument('--num_epochs', type=int, default=600, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--use_case', type=int, default=1, help='Use case')
    parser.add_argument('--lr', type=float, default=0.001, help='learning  rate')

    args = parser.parse_args()
    univariate = {
        "ACSF1",
        "Adiac",
        "AllGestureWiimoteX",
        "AllGestureWiimoteY",
        "AllGestureWiimoteZ",
        "ArrowHead",
        "Beef",
        "BeetleFly",
        "BirdChicken",
        "BME",
        "Car",
        "CBF",
        "Chinatown",
        "ChlorineConcentration",
        "CinCECGTorso",
        "Coffee",
        "Computers",
        "CricketX",
        "CricketY",
        "CricketZ",
        "Crop",
        "DiatomSizeReduction",
        "DistalPhalanxOutlineAgeGroup",
        "DistalPhalanxOutlineCorrect",
        "DistalPhalanxTW",
        "DodgerLoopDay",
        "DodgerLoopGame",
        "DodgerLoopWeekend",
        "Earthquakes",
        "ECG200",
        "ECG5000",
        "ECGFiveDays",
        "ElectricDevices",
        "EOGHorizontalSignal",
        "EOGVerticalSignal",
        "EthanolLevel",
        "FaceAll",
        "FaceFour",
        "FacesUCR",
        "FiftyWords",
        "Fish",
        "FordA",
        "FordB",
        "FreezerRegularTrain",
        "FreezerSmallTrain",
        "Fungi",
        "GestureMidAirD1",
        "GestureMidAirD2",
        "GestureMidAirD3",
        "GesturePebbleZ1",
        "GesturePebbleZ2",
        "GunPoint",
        "GunPointAgeSpan",
        "GunPointMaleVersusFemale",
        "GunPointOldVersusYoung",
        "Ham",
        "HandOutlines",
        "Haptics",
        "Herring",
        "HouseTwenty",
        "InlineSkate",
        "InsectEPGRegularTrain",
        "InsectEPGSmallTrain",
        "InsectWingbeatSound",
        "ItalyPowerDemand",
        "LargeKitchenAppliances",
        "Lightning2",
        "Lightning7",
        "Mallat",
        "Meat",
        "MedicalImages",
        "MelbournePedestrian",
        "MiddlePhalanxOutlineCorrect",
        "MiddlePhalanxOutlineAgeGroup",
        "MiddlePhalanxTW",
        "MixedShapesRegularTrain",
        "MixedShapesSmallTrain",
        "MoteStrain",
        "NonInvasiveFetalECGThorax1",
        "NonInvasiveFetalECGThorax2",
        "OliveOil",
        "OSULeaf",
        "PhalangesOutlinesCorrect",
        "Phoneme",
        "PickupGestureWiimoteZ",
        "PigAirwayPressure",
        "PigArtPressure",
        "PigCVP",
        "PLAID",
        "Plane",
        "PowerCons",
        "ProximalPhalanxOutlineCorrect",
        "ProximalPhalanxOutlineAgeGroup",
        "ProximalPhalanxTW",
        "RefrigerationDevices",
        "Rock",
        "ScreenType",
        "SemgHandGenderCh2",
        "SemgHandMovementCh2",
        "SemgHandSubjectCh2",
        "ShakeGestureWiimoteZ",
        "ShapeletSim",
        "ShapesAll",
        "SmallKitchenAppliances",
        "SmoothSubspace",
        "SonyAIBORobotSurface1",
        "SonyAIBORobotSurface2",
        "StarLightCurves",
        "Strawberry",
        "SwedishLeaf",
        "Symbols",
        "SyntheticControl",
        "ToeSegmentation1",
        "ToeSegmentation2",
        "Trace",
        "TwoLeadECG",
        "TwoPatterns",
        "UMD",
        "UWaveGestureLibraryAll",
        "UWaveGestureLibraryX",
        "UWaveGestureLibraryY",
        "UWaveGestureLibraryZ",
        "Wafer",
        "Wine",
        "WordSynonyms",
        "Worms",
        "WormsTwoClass",
        "Yoga",
    }
    multivariate = {
        "EthanolConcentration",
    }
    # Loop over each dataset in the univariate list
    
    #if(args.univariate): 
    #     for dataset in univariate:
    #         run(data_name=dataset, univariate=args.univariate, read_data=args.read_data, 
    #             window_num=args.window_num, edge_num=args.edge_num, num_epochs=args.num_epochs, 
    #             batch_size=args.batch_size, use_case=args.use_case, lr=args.lr)
    #else:
    for dataset in multivariate:
        run(data_name=dataset, univariate=args.univariate, read_data=args.read_data, 
               window_num=args.window_num, edge_num=args.edge_num, num_epochs=args.num_epochs, 
               batch_size=args.batch_size, use_case=args.use_case, lr=args.lr)