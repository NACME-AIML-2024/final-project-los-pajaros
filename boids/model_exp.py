import os
import torch
from tqdm import tqdm
from dataset import BoidDatasetLoader
from models import GraphSeqGenerator
from utils import temporal_signal_split
import matplotlib.pyplot as plt


loader = BoidDatasetLoader()
dataset = loader.get_dataset()
dataset.snapshot_count

train_dataset, test_dataset = temporal_signal_split(dataset)

generator = generator = GraphSeqGenerator(node_feat_dim=4,
                              enc_hidden_dim=32,
                              enc_latent_dim=16,
                              dec_hidden_dim=32,
                              pred_horizon=8,
                              min_max_x=(loader.min_features[0], loader.max_features[0]),
                              min_max_y=(loader.min_features[1], loader.max_features[1]),
                              min_max_edge_weight=(loader.min_edge_weight, loader.max_edge_weight),
                              visualRange=75,
                            )

device = torch.device('cpu')

generator.to(device)

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

def test_generator_plot(test_data, generator, window=8, delay=0, horizon=1, stride=1, boid_indices=[0,1,2,3], output_dir='../generator_plots'):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_timesteps = test_data.snapshot_count
    sample_span = window + delay + horizon

    generator.eval()
    with torch.no_grad():
        for start in tqdm(range(0, total_timesteps - sample_span + 1, stride), desc='Plotting Testing Data'):
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

            def plot_boid_trajectories(boid_indices, boids_history, boids_future, boids_pred_future, filename):
                
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
                plt.savefig(filename)
                plt.close()

            filename = os.path.join(output_dir, f'plot_{start}.png')
            
            # Call the function to plot the trajectories
            plot_boid_trajectories(boid_indices, boids_history, boids_future, boids_pred_future, filename)
            
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
        if (epoch+1) % 10 == 0:
            test_generator(test_dataset, generator, horizon=generator.out_steps)
            test_generator_plot(test_dataset, generator, horizon=generator.out_steps, output_dir=f'../generator_test_plots_at_epoch_{epoch+1}')
            


optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)

train_generator(train_dataset, 50, generator, optimizer, horizon=generator.out_steps)