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
        
# Again from https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/signal/train_test_split.py
def temporal_signal_split(data_iterator, train_ratio=0.8):
    train_snapshots = int(data_iterator.snapshot_count * train_ratio)
    train_iterator = data_iterator[0:train_snapshots]
    test_iterator = data_iterator[train_snapshots:]
    return train_iterator, test_iterator

def plot_predictions(y_hat, target_seq, filename,loader):
    
    """
    Plots the predicted points and the actual points, and saves the plot as an image file.

    Args:
        y_hat (torch.Tensor): The predicted points.
        target_seq (torch_geometric.data.Data): The actual points.
        filename (str): The filename to save the plot.
    """
    
    y_hat_np = y_hat.detach().cpu().numpy()
    target_np = target_seq[0].x.detach().cpu().numpy()

    loader.min_features[0:2], loader.max_features[0:2]

    plt.figure(figsize=(10, 5))

    # Plot predicted points
    plt.scatter(y_hat_np[:, 0] * (loader.max_features[0] - loader.min_features[0]) + loader.min_features[0], 
                y_hat_np[:, 1] * (loader.max_features[1] - loader.min_features[1]) + loader.min_features[1], color='r', label='Predicted', alpha=0.6)

    # Plot actual points
    plt.scatter(target_np[:, 0] * (loader.max_features[0] - loader.min_features[0]) + loader.min_features[0], 
                target_np[:, 1] * (loader.max_features[1] - loader.min_features[1]) + loader.min_features[1], color='b', label='Actual', alpha=0.6)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim((0,1000))
    plt.ylim((0,1000))
    plt.legend()
    plt.title('Predicted vs Actual Points')
    plt.savefig(filename)
    plt.close()

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

def create_video_from_images(image_folder, output_video_path, fps=30):
    # Get list of images in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    def extract_number(filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else -1
    images.sort(key=extract_number)  # Ensure the images are in the correct order

    # Read the first image to get the dimensions
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width = frame.shape[:2]

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs as well
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # Release the video writer object
    video.release()







