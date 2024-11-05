# Standard library imports
from pathlib import Path
import os
import sys
import math
import itertools
from typing import List, Union, Sequence


# Third-party library imports
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from numpy import ndarray
from tqdm import tqdm
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
from torch_geometric.data import Data
from torch_geometric.nn import ChebConv
from torch_geometric.nn.conv import GeneralConv


def train_generator(train_data, num_epochs, generator, optimizer, window=8, delay=0, horizon=1, stride=1):
    """
    Trains the given model using the provided training data.

    Args:
        train_data (Dataset): The dataset containing the training data.
        num_epochs (int): The number of epochs to train the model.
        model (nn.Module): The model to be trained.
        optimizer (Optimizer): The optimizer used for training the model.
        window (int, optional): The size of the input sequence window. Defaults to 8.
        delay (int, optional): The delay between the input sequence and the target sequence. Defaults to 0.
        horizon (int, optional): The prediction horizon. Defaults to 1.
        stride (int, optional): The stride for iterating over the training data. Defaults to 1.

    Returns:
        None
    """
    total_timesteps = train_data.snapshot_count
    sample_span = window + delay + horizon

    generator.train()
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch+1}/{num_epochs}')
        epoch_cost = 0
        for start in tqdm(range(0, total_timesteps - sample_span + 1, stride), desc='Training'):
            input_seq = train_data[start:start + window]
            target_seq = train_data[start + window + delay: start + window + delay + horizon]
            predictions = generator(input_seq, None, None)
            predictions = torch.stack(predictions, dim=0)
            target_seq = torch.stack([target_seq[i].x for i in range(target_seq.snapshot_count)], dim=0)
            cost = torch.mean((predictions - target_seq) ** 2)
            epoch_cost += cost.item()
            cost.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f'Cost after epoch {epoch+1}: {epoch_cost}')
        
def train(train_data, num_epochs, model, optimizer, window=8, delay=0, horizon=1, stride=1):
    """
    Trains the given model using the provided training data.

    Args:
        train_data (Dataset): The dataset containing the training data.
        num_epochs (int): The number of epochs to train the model.
        model (nn.Module): The model to be trained.
        optimizer (Optimizer): The optimizer used for training the model.
        window (int, optional): The size of the input sequence window. Defaults to 8.
        delay (int, optional): The delay between the input sequence and the target sequence. Defaults to 0.
        horizon (int, optional): The prediction horizon. Defaults to 1.
        stride (int, optional): The stride for iterating over the training data. Defaults to 1.

    Returns:
        None
    """
    total_timesteps = train_data.snapshot_count
    sample_span = window + delay + horizon

    model.train()
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch+1}/{num_epochs}')
        epoch_cost = 0 
        for start in tqdm(range(0, total_timesteps - sample_span + 1, stride), desc='Training'):
            input_seq = train_data[start:start + window]
            target_seq = train_data[start + window + delay: start + window + delay + horizon]

            h_t_prev = None
            for i in range(input_seq.snapshot_count):
                snapshot = input_seq[i]
                y_hat, h_t = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h_t_prev)
                h_t_prev = h_t

            cost = torch.mean((y_hat - target_seq[0].x) ** 2)
            epoch_cost += cost.item()
            #print('Loss', cost.item())
            cost.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f'Cost after epoch {epoch+1}: {epoch_cost}')
        

    
