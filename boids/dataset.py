
import numpy as np
import pandas as pd

from utils import DynamicGraphTemporalSignal


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