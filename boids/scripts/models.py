# Standard library imports
from pathlib import Path
from typing import List, Union, Sequence


# Third-party library imports
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

class GConvGRU(torch.nn.Module):
    r"""An implementation of the Chebyshev Graph Convolutional Gated Recurrent Unit
    Cell. For details see this paper: `"Structured Sequence Modeling with Graph
    Convolutional Recurrent Networks." <https://arxiv.org/abs/1612.07659>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        normalization: str = "sym",
        bias: bool = True,
    ):
        super(GConvGRU, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalization = normalization
        self.bias = bias
        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):

        self.conv_x_z = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_z = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

    def _create_reset_gate_parameters_and_layers(self):

        self.conv_x_r = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_r = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

    def _create_candidate_state_parameters_and_layers(self):

        self.conv_x_h = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_h = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
            #print('You none')
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H, lambda_max):
        #print(X.shape, edge_index.shape, edge_weight.shape, H.shape, self.out_channels)
        Z = self.conv_x_z(X, edge_index, edge_weight, lambda_max=lambda_max)
        Z = Z + self.conv_h_z(H, edge_index, edge_weight, lambda_max=lambda_max)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H, lambda_max):
        R = self.conv_x_r(X, edge_index, edge_weight, lambda_max=lambda_max)
        R = R + self.conv_h_r(H, edge_index, edge_weight, lambda_max=lambda_max)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R, lambda_max):
        H_tilde = self.conv_x_h(X, edge_index, edge_weight, lambda_max=lambda_max)
        H_tilde = H_tilde + self.conv_h_h(H * R, edge_index, edge_weight, lambda_max=lambda_max)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H
    
    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
        lambda_max: torch.Tensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
            * **lambda_max** *(PyTorch Tensor, optional but mandatory if normalization is not sym)* - Largest eigenvalue of Laplacian.


        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H, lambda_max)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H, lambda_max)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R, lambda_max)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H

#  Basic Graphical Recurrent Neural Network
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

class GraphSeqGenerator(torch.nn.Module):
    def __init__(self, node_feat_dim, enc_hidden_dim, enc_latent_dim, dec_hidden_dim, pred_horizon, min_max_x, min_max_y, min_max_edge_weight, visualRange,device):
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