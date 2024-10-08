
import numpy as np
import torch

from utils import GConvGRU
import torch.nn.functional as F

# Basic Graphical Recurrent Neural Network
class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, filters):
        super(RecurrentGCN, self).__init__()
        self.recurrent = GConvGRU(node_features, filters, 2)
        self.linear = torch.nn.Linear(filters, 4)

    def forward(self, x, edge_index, edge_weight, H=None):
        h = self.recurrent(x, edge_index, edge_weight, H)
        x = F.relu(h)
        x = self.linear(x)
        return x, h
    
# Code For Generator 
class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, k=2):
        super(Encoder, self).__init__()
        self.recurrent = GConvGRU(input_dim, hidden_dim, k)
        self.linear = torch.nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x, edge_index, edge_weight, h):
        h_0 = self.recurrent(x, edge_index, edge_weight, h)
        h = F.relu(h_0)
        h = self.linear(h)
        return h, h_0 # Output = (latent matrix, hidden state for encoder)

class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, k=2):
        super(Decoder, self).__init__()
        self.recurrent = GConvGRU(latent_dim, hidden_dim, k)
        self.linear = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, z, edge_index, edge_weight, h):
        h_0 = self.recurrent(z, edge_index, edge_weight, h)
        h = F.relu(h_0)
        h = self.linear(h)
        return h, h_0 # Output = (Final Output, hidden state for decoder)

class GraphSeqGenerator(torch.nn.Module):
    def __init__(self, node_feat_dim, enc_hidden_dim, enc_latent_dim, dec_hidden_dim, pred_horizon, min_max_x, min_max_y, min_max_edge_weight, visualRange):
        super(GraphSeqGenerator, self).__init__()
        self.encoder = Encoder(node_feat_dim, enc_hidden_dim, enc_latent_dim)
        self.decoder = Decoder(enc_latent_dim, dec_hidden_dim, node_feat_dim)
        self.out_steps = pred_horizon
        self.min_x, self.max_x = min_max_x
        self.min_y, self.max_y = min_max_y
        self.min_edge_weight, self.max_edge_weight = min_max_edge_weight
        self.visualRange = visualRange

    def _compute_edge_index_and_weight(self, y_hat):
        # Not designed for batches :/
        # Grab x and y features
        y_hat_x = y_hat[:, 0].detach().numpy()
        y_hat_y = y_hat[:, 1].detach().numpy()

        # Undo normalization
        y_hat_x = y_hat_x * (self.max_x - self.min_x) + self.min_x
        y_hat_y = y_hat_y * (self.max_y - self.min_y) + self.min_y

        # Compute the distance of all points and include that edge if its less than visualRange
        coords = np.stack((y_hat_x, y_hat_y), axis=1)
        dist_matrix = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=2)
        
        # Get indices where distance is less than visualRange
        edge_indices = np.where((dist_matrix < self.visualRange) & (dist_matrix > 0))
        
        # Create edge_index and edge_attr
        edge_index = np.vstack((edge_indices[0], edge_indices[1]))
        edge_weight = dist_matrix[edge_indices]

        #Normalize edge_weight
        edge_weight = (edge_weight - self.min_edge_weight) / (self.max_edge_weight - self.min_edge_weight)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)
        return edge_index, edge_weight


        
    def forward(self, sequence, h_enc, h_dec):
        # Warmup Section
        for i in range(sequence.snapshot_count):
            snapshot = sequence[i]
            z, h_enc_0 = self.encoder(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h_enc)
            y_hat, h_dec_0 = self.decoder(z, snapshot.edge_index, snapshot.edge_weight, h_dec)

            h_enc = h_enc_0
            h_dec = h_dec_0

        predictions = []
        predictions.append(y_hat)

        # Prediction Section
        for _ in range(self.out_steps-1):
            # TODO: Compute edge index and edge_attr of y_hat :()
            y_hat_edge_index, y_hat_edge_attr = self._compute_edge_index_and_weight(y_hat)

            z, h_enc_0 = self.encoder(y_hat, y_hat_edge_index, y_hat_edge_attr, h_enc)
            y_hat, h_dec_0 = self.decoder(z, y_hat_edge_index, y_hat_edge_attr, h_dec)

            predictions.append(y_hat)
        return predictions
    
# STILL UNDER PRODUCTION...
class GraphSeqDiscriminator(torch.nn.Module):
    def __init__(self, node_feat_dim, enc_hidden_dim, enc_latent_dim):
        super(GraphSeqDiscriminator, self).__init__()

        self.encoder = Encoder(node_feat_dim, enc_hidden_dim, enc_latent_dim)
        self.linear = torch.nn.Linear(enc_latent_dim, 1)

    def forward(self, x, edge_index, edge_weight, h):
        z, h_enc_0 = self.encoder(x, edge_index, edge_weight, h)
        z = F.relu(z)

        # Apply global mean pooling across the node dimension (dim=0) to aggregate node features
        z_pooled = z.mean(dim=0)
        out = self.linear(z_pooled)
        out = torch.sigmoid(z_pooled)
        return out, h_enc_0