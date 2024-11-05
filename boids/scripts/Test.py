
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

# Project-specific import
from .utils import plot_predictions


def test_generator_plot(test_data, generator, window=8, delay=0, horizon=1, stride=1, boid_indices=[0,1,2,3, 4, 5]):
    """
    Tests the given generator model using the provided test data and plots the trajectories.

    Args:
        test_data (Dataset): The dataset containing the test data.
        generator (nn.Module): The generator model to be tested.
        window (int, optional): The size of the input sequence window. Defaults to 8.
        delay (int, optional): The delay between the input sequence and the target sequence. Defaults to 0.
        horizon (int, optional): The prediction horizon. Defaults to 1.
        stride (int, optional): The stride for iterating over the test data. Defaults to 1.
        boid_indices (list, optional): List of boid indices to plot. Defaults to [0, 1, 2].

    Returns:
        None
    """
    total_timesteps = test_data.snapshot_count
    sample_span = window + delay + horizon

    generator.eval()
    with torch.no_grad():
        for start in tqdm(range(0, total_timesteps - sample_span + 1, stride), desc='Testing'):
            input_seq = test_data[start:start + window]
            target_seq = test_data[start + window + delay: start + window + delay + horizon]
            predictions = generator(input_seq, None, None)
            predictions = torch.stack(predictions, dim=0)
            target_seq = torch.stack([target_seq[i].x for i in range(target_seq.snapshot_count)], dim=0)
    
            boids_history = []
            boids_future = []
            boids_pred_future = []
            for boid_idx in boid_indices:
                boid_idx_actual_x = [target_seq[i, boid_idx, 0].item() for i in range(target_seq.shape[0])]
                boid_idx_actual_y = [target_seq[i, boid_idx, 1].item() for i in range(target_seq.shape[0])]
                boid_idx_pred_x = [pred[boid_idx, 0].item() for pred in predictions]
                boid_idx_pred_y = [pred[boid_idx, 1].item() for pred in predictions]
                boid_idx_hist_x = [input_seq[i].x[boid_idx, 0].item() for i in range(input_seq.snapshot_count)]
                boid_idx_hist_y = [input_seq[i].x[boid_idx, 1].item() for i in range(input_seq.snapshot_count)]

                boids_history.append((boid_idx_hist_x, boid_idx_hist_y))
                boids_future.append((boid_idx_actual_x, boid_idx_actual_y))
                boids_pred_future.append((boid_idx_pred_x, boid_idx_pred_y))

            def plot_boid_trajectories(boid_indices, boids_history, boids_future, boids_pred_future):
                """
                Plots the actual paths vs the predicted paths for the given boid indices.

                Args:
                    boid_indices (list): List of boid indices to plot.
                    boids_history (list): List of tuples containing historical x and y coordinates for each boid.
                    boids_future (list): List of tuples containing actual future x and y coordinates for each boid.
                    boids_pred_future (list): List of tuples containing predicted future x and y coordinates for each boid.

                Returns:
                    None
                """
                plt.figure(figsize=(10, 5))
                for i, boid_idx in enumerate(boid_indices):
                    hist_x, hist_y = boids_history[i]
                    actual_x, actual_y = boids_future[i]
                    pred_x, pred_y = boids_pred_future[i]

                    plt.plot(hist_x, hist_y, label=f'Boid {boid_idx} History', linestyle='-', marker='o', alpha=0.7)
                    plt.plot(actual_x, actual_y, label=f'Boid {boid_idx} Actual Future', linestyle='-', marker='o', color='blue', alpha=0.7)
                    plt.plot(pred_x, pred_y, label=f'Boid {boid_idx} Predicted Future', linestyle='-', marker='^', color='red', alpha=0.7)

                plt.xlabel('x')
                plt.ylabel('y')
                plt.title('Boid Trajectories')
                plt.show()

            # Call the function to plot the trajectories
            plot_boid_trajectories(boid_indices, boids_history, boids_future, boids_pred_future)

def test_generator(test_data, generator, window=8, delay=0, horizon=1, stride=1):
    """
    Tests the given generator model using the provided test data.

    Args:
        test_data (Dataset): The dataset containing the test data.
        generator (nn.Module): The generator model to be tested.
        window (int, optional): The size of the input sequence window. Defaults to 8.
        delay (int, optional): The delay between the input sequence and the target sequence. Defaults to 0.
        horizon (int, optional): The prediction horizon. Defaults to 1.
        stride (int, optional): The stride for iterating over the test data. Defaults to 1.

    Returns:
        float: The average loss over the test dataset.
    """
    total_timesteps = test_data.snapshot_count
    sample_span = window + delay + horizon

    generator.eval()
    total_loss = 0
    with torch.no_grad():
        for start in tqdm(range(0, total_timesteps - sample_span + 1, stride), desc='Testing'):
            input_seq = test_data[start:start + window]
            target_seq = test_data[start + window + delay: start + window + delay + horizon]
            predictions = generator(input_seq, None, None)
            predictions = torch.stack(predictions, dim=0)
            target_seq = torch.stack([target_seq[i].x for i in range(target_seq.snapshot_count)], dim=0)
            cost = torch.mean((predictions - target_seq) ** 2)
            total_loss += cost.item()

    average_loss = total_loss / (total_timesteps - sample_span + 1)
    print('Testing Loss: ', average_loss)
    return average_loss

def test(test_data, model, window=8, delay=0, horizon=1, stride=1, output_dir='../plots',loader=None):
    """
    Tests the given model using the provided test data and saves the plots as images.

    Args:
        test_data (Dataset): The dataset containing the test data.
        model (nn.Module): The model to be tested.
        window (int, optional): The size of the input sequence window. Defaults to 8.
        delay (int, optional): The delay between the input sequence and the target sequence. Defaults to 0.
        horizon (int, optional): The prediction horizon. Defaults to 1.
        stride (int, optional): The stride for iterating over the test data. Defaults to 1.
        output_dir (str, optional): The directory to save the plots. Defaults to 'plots'.

    Returns:
        float: The average loss over the test dataset.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_timesteps = test_data.snapshot_count
    sample_span = window + delay + horizon

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for start in tqdm(range(0, total_timesteps - sample_span + 1, stride)):
            input_seq = test_data[start:start + window]
            target_seq = test_data[start + window + delay: start + window + delay + horizon]

            h_t_prev = None
            for i in range(input_seq.snapshot_count):
                snapshot = input_seq[i]
                y_hat, h_t = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h_t_prev)
                h_t_prev = h_t

            cost = torch.mean((y_hat - target_seq[0].x) ** 2)
            total_loss += cost.item()

            # Save the plot
            filename = os.path.join(output_dir, f'plot_{start}.png')
            plot_predictions(y_hat, target_seq, filename,loader)

    average_loss = total_loss / (total_timesteps - sample_span + 1)
    return average_loss


