# Standard library imports
from typing import Union, Sequence

# Third-party library imports
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from tqdm import tqdm

class DynamicGraphTemporalSignal(object):
    r"""
    Pulled from: 
    https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/signal/dynamic_graph_temporal_signal.py
    A data iterator object to contain a dynamic graph with a
    changing edge set and weights . The feature set and node labels
    (target) are also dynamic. The iterator returns a single discrete temporal
    snapshot for a time period (e.g. day or week). This single snapshot is a
    Pytorch Geometric Data object. Between two temporal snapshots the edges,
    edge weights, target matrices and optionally passed attributes might change.

    Args:
        edge_indices (Sequence of Numpy arrays): Sequence of edge index tensors.
        edge_weights (Sequence of Numpy arrays): Sequence of edge weight tensors.
        features (Sequence of Numpy arrays): Sequence of node feature tensors.
        targets (Sequence of Numpy arrays): Sequence of node label (target) tensors.
        **kwargs (optional Sequence of Numpy arrays): Sequence of additional attributes.
    """

    def __init__(
        self,
        edge_indices: Sequence[Union[np.ndarray, None]],
        edge_weights: Sequence[Union[np.ndarray, None]],
        features: Sequence[Union[np.ndarray, None]],
        targets: Sequence[Union[np.ndarray, None]],
        **kwargs: Sequence[np.ndarray]
    ):
        self.edge_indices = edge_indices
        self.edge_weights = edge_weights
        self.features = features
        self.targets = targets
        self.additional_feature_keys = []
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.additional_feature_keys.append(key)
        self._check_temporal_consistency()
        self._set_snapshot_count()

    def _check_temporal_consistency(self):
        assert len(self.features) == len(
            self.targets
        ), "Temporal dimension inconsistency."
        assert len(self.edge_indices) == len(
            self.edge_weights
        ), "Temporal dimension inconsistency."
        assert len(self.features) == len(
            self.edge_weights
        ), "Temporal dimension inconsistency."
        for key in self.additional_feature_keys:
            assert len(self.targets) == len(
                getattr(self, key)
            ), "Temporal dimension inconsistency."

    def _set_snapshot_count(self):
        self.snapshot_count = len(self.features)

    def _get_edge_index(self, time_index: int):
        if self.edge_indices[time_index] is None:
            return self.edge_indices[time_index]
        else:
            return torch.LongTensor(self.edge_indices[time_index])

    def _get_edge_weight(self, time_index: int):
        if self.edge_weights[time_index] is None:
            return self.edge_weights[time_index]
        else:
            return torch.FloatTensor(self.edge_weights[time_index])

    def _get_features(self, time_index: int):
        if self.features[time_index] is None:
            return self.features[time_index]
        else:
            return torch.FloatTensor(self.features[time_index])

    def _get_target(self, time_index: int):
        if self.targets[time_index] is None:
            return self.targets[time_index]
        else:
            if self.targets[time_index].dtype.kind == "i":
                return torch.LongTensor(self.targets[time_index])
            elif self.targets[time_index].dtype.kind == "f":
                return torch.FloatTensor(self.targets[time_index])

    def _get_additional_feature(self, time_index: int, feature_key: str):
        feature = getattr(self, feature_key)[time_index]
        if feature.dtype.kind == "i":
            return torch.LongTensor(feature)
        elif feature.dtype.kind == "f":
            return torch.FloatTensor(feature)

    def _get_additional_features(self, time_index: int):
        additional_features = {
            key: self._get_additional_feature(time_index, key)
            for key in self.additional_feature_keys
        }
        return additional_features

    def __getitem__(self, time_index: Union[int, slice]):
        if isinstance(time_index, slice):
            snapshot = DynamicGraphTemporalSignal(
                self.edge_indices[time_index],
                self.edge_weights[time_index],
                self.features[time_index],
                self.targets[time_index],
                **{key: getattr(self, key)[time_index] for key in self.additional_feature_keys}
            )
        else:
            x = self._get_features(time_index)
            edge_index = self._get_edge_index(time_index)
            edge_weight = self._get_edge_weight(time_index)
            y = self._get_target(time_index)
            additional_features = self._get_additional_features(time_index)

            snapshot = Data(x=x, edge_index=edge_index, edge_attr=edge_weight,
                            y=y, **additional_features)
        return snapshot

    def __next__(self):
        if self.t < len(self.features):
            snapshot = self[self.t]
            self.t = self.t + 1
            return snapshot
        else:
            self.t = 0
            raise StopIteration

    def __iter__(self):
        self.t = 0
        return self

class BoidDatasetLoader(object):
    """
    The BoidDatasetLoader class is designed to load and process the Boid Dataset.
    It reads data from CSV files, processes it, and prepares it for further analysis or modeling. The class performs the
    following key functions:

    1. Initialization (__init__ method):
       - Calls the _read_data method to load and preprocess the data.

    2. Data Reading and Preprocessing (_read_data method):
       - Reads simulation data from 'simulation.csv' and 'simulation_edges.csv'.
       - Drops the 'Simulation' column from both dataframes.
       - Renames columns to standardize the naming convention (e.g., 'Boids' to 'BoidID', 'Boid_i' to 'BoidID_i').
       - Stores the cleaned dataframes in the _dataset attribute.
       - Calls the _process_dataset method to further process the data and extract features, edges, and edge weights.

    3. Dataset Processing (_process_dataset method):
       - Groups the simulation data by 'Timestep'.
       - Initializes lists to store edge indices, node features, and distances.
       - Iterates over each timestep to extract relevant data for that timestep.
       - Converts the dataframes to NumPy arrays for efficient computation.
       - Creates a dictionary to map BoidID to coordinates.
       - Uses vectorized operations to calculate distances between boids with edges.

    4. Getting Features and Edge Weights (_get_edge_weights() and _get_features() methods):
        - Normalized features and _edge_weights using min-max normalization
        - Attributes are now normalized when called and returned

    5. Getting Target (_get_target() method):
        - If t is the current index of our dataset, then t+1 is the target
        - Contains the node features of the graph at t+1
        - Will probably not used this as wouldnt work entirely for more than 1 timestep prediction

    Attributes:
        _dataset: A tuple containing the cleaned simulation data and edge data.
        features: Node features extracted from the simulation data.
        _edges: Edge indices representing connections between boids.
        _edge_weights: Weights of the edges, which could represent distances or other metrics.

    Methods:
        __init__(): Initializes the class and reads the data.
        _read_data(): Reads and preprocesses the data from CSV files.
        _process_dataset(sim_df, sim_edges_df): Processes the dataset to extract features, edges, and edge weights.
    """
    def __init__(self):
        self._read_data()

    def _read_data(self):
        path_to_sim = '../data/simulation.csv'
        path_to_sim_edges = '../data/simulation_edges.csv'
        sim_df = pd.read_csv(path_to_sim)
        sim_edges_df = pd.read_csv(path_to_sim_edges)

        sim_df.drop(columns='Simulation', inplace=True)
        sim_df.rename(columns={'Boids':'BoidID'}, inplace=True)

        sim_edges_df.drop(columns='Simulation', inplace=True)
        sim_edges_df.rename(columns={'Boid_i':'BoidID_i', 'Boid_j':'BoidID_j'}, inplace=True)
        
        self._dataset = (sim_df, sim_edges_df)
        self.features, self._edges, self._edge_weights = self._process_dataset(self._dataset[0], self._dataset[1])

    def _process_dataset(self, sim_df, sim_edges_df):
        # Group the dataframes by 'Timestep'
        sim_grouped = sim_df.groupby('Timestep')
        edges_grouped = sim_edges_df.groupby('Timestep')
        
        # Initialize lists to store edge indices and node features
        edge_indices = []
        node_features = []
        
        distances = []

        # Iterate over each group
        for timestep, _ in sim_grouped:
            # Extract relevant columns for the current timestep
            timestep_df = sim_grouped.get_group(timestep)[['x', 'y', 'dx', 'dy', 'BoidID']]
            timestep_edges_df = edges_grouped.get_group(timestep)[['BoidID_i', 'BoidID_j']]
            
            # Convert dataframes to numpy arrays
            node_array = timestep_df[['x', 'y', 'dx', 'dy']].to_numpy()
            edge_array = timestep_edges_df.to_numpy().T

            # Create a dictionary to map BoidID to coordinates
            boid_coords = {boid_id: coords for boid_id, coords in zip(timestep_df['BoidID'], timestep_df[['x', 'y']].values)}
            
            # Get coordinates for boids involved in edges
            boid_i_coords = np.array([boid_coords[boidid_i] for boidid_i in edge_array[0]])
            boid_j_coords = np.array([boid_coords[boidid_j] for boidid_j in edge_array[1]])
            
            # Calculate distances using vectorized operations
            timestep_distances = np.linalg.norm(boid_i_coords - boid_j_coords, axis=1)
            
            distances.append(timestep_distances)

            # Append the numpy arrays to the respective lists
            edge_indices.append(edge_array)
            node_features.append(node_array)

        # Return the lists of edge indices and node features
        return node_features, edge_indices, distances

    def _compute_feature_min_max(self, feature_list):
        """
        Compute the minimum and maximum of features across all node features.

        Parameters:
        all_node_features (list of np.ndarray): List of numpy arrays containing node features.

        Returns:
        tuple: A tuple containing two numpy arrays: (final_min, final_max).
        """
        if not feature_list:
            raise ValueError("The input list 'all_node_features' is empty.")
        
        # Initialize final_min and final_max with appropriate dimensions
        if len(feature_list[0].shape) == 0:
            raise ValueError('The input list is missing node features') 
        
        if len(feature_list[0].shape) == 1:
            feature_dim = 1
            axis_val = None
            final_min = float('inf')
            final_max = float('-inf')
        else:
            feature_dim = feature_list[0].shape[1]
            axis_val = 0
            final_min = np.array([float('inf')] * feature_dim)
            final_max = np.array([float('-inf')] * feature_dim)
        
        # Iterate through all node features to compute final_min and final_max
        for features in feature_list:
            curr_max = np.max(features, axis=axis_val)
            curr_min = np.min(features, axis=axis_val)
            final_max = np.max(np.array([final_max, curr_max]), axis=0)
            final_min = np.min(np.array([final_min, curr_min]), axis=0)
        
        return final_min, final_max
    
    def _minmax_scale(self, feature_list, final_min, final_max):
        normalized = []
        for i in range(len(feature_list)):
            X = feature_list[i]
            X_std = (X - final_min) / (final_max - final_min)
            normalized.append(X_std)
        return normalized

    def undo_minmax_scale(self, normalized_feature_list, final_min, final_max):
        unnormalized = []
        for i in range(len(normalized_feature_list)):
            X_std = normalized_feature_list[i]
            X_scaled = X_std * (final_max - final_min) + final_min
            unnormalized.append(X_scaled)
        return unnormalized
    

    def _get_edge_weights(self):
        self.min_edge_weight, self.max_edge_weight = self._compute_feature_min_max(self._edge_weights)
        self._edge_weights = self._minmax_scale(self._edge_weights, self.min_edge_weight, self.max_edge_weight)

    def _get_features(self):
        self.min_features, self.max_features = self._compute_feature_min_max(self.features)
        self.features = self._minmax_scale(self.features, self.min_features, self.max_features)
    def _get_targets(self):
        self.targets = [self.features[i] for i in range(1, len(self.features))]


    def get_dataset(self) -> DynamicGraphTemporalSignal:

        self._get_edge_weights()
        self._get_features()
        self._get_targets()
        dataset = DynamicGraphTemporalSignal(
            self._edges[:len(self.features)-1], 
            self._edge_weights[:len(self.features)-1], 
            self.features[:len(self.features)-1], 
            self.targets
        )
        return dataset 
    
def make_batch(list_of_sequences, seq_length):
    batch_dataset = []
    # Look at the first item in the sequence, and then second and so on
    for t in range(seq_length):
        batch_list = []
        # Iterate through list_of_sequences and add the the t-th item from each of them to the batch_list
        for i in range(len(list_of_sequences)):
            batch_list.append(list_of_sequences[i][t])
        # Create a Batch Object from batch_list and add that to batch_dataset
        batch_dataset.append(Batch.from_data_list(batch_list))
    return batch_dataset

def dataLoader(dataset, window=8, delay=0, horizon=1, stride=1, batch_size=32):
    sample_span = window + delay + horizon
    total_timesteps = dataset.snapshot_count
    # First get list of observed sequences and their corresponding target sequences
    obs_seq_list = []
    target_seq_list = []
    for start in tqdm(range(0, total_timesteps - sample_span + 1, stride), desc='Getting Observation and Target Sequences'):
        obs_seq = dataset[start:start + window]
        obs_seq_list.append(obs_seq)
        target_seq = dataset[start + window + delay: start + window + delay + horizon]
        target_seq_list.append(target_seq)
    # Second need to break up the list into bactches of size BATCH_SIZE
    batch_list = []
    for i in tqdm(range(0, len(obs_seq_list), batch_size), desc="Making Fresh Batches"):
        # Get the start and end index for the batch
        start, end = i, min(i+batch_size, len(obs_seq_list)) # Doing min here in case I don't have enough in my list to make a complete batch
        # Grab a sample of both list to create a batch
        sample_obs_seq_list = obs_seq_list[start:end]
        sample_target_seq_list = target_seq_list[start:end]

        # Calling helper function make_batch to create a batch from a list of sequences
        curr_obs_batch = make_batch(sample_obs_seq_list, window)
        curr_target_batch = make_batch(sample_target_seq_list, horizon)
        batch_list.append((curr_obs_batch, curr_target_batch))

    return batch_list