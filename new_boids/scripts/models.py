# Standard library imports
from pathlib import Path
from typing import List, Union, Sequence


# Third-party library imports
import numpy as np
import torch
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from torch_geometric.data import Data, Batch

#-----BELOW CODE FOR GAN-----#
class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, recurrent_model='gru', k=2):
        super(Encoder, self).__init__()
        if recurrent_model == 'gru':
            self.recurrent = GConvGRU(input_dim, hidden_dim, k)
            self.recurrent_model = 'gru'
        else:
            self.recurrent = GConvLSTM(input_dim, hidden_dim, k)
            self.recurrent_model = 'lstm'
        self.linear = torch.nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x, edge_index, edge_weight, h, c=None):
        if self.recurrent_model == 'gru':
            h_0 = self.recurrent(x, edge_index, edge_weight, h)
        else:
            h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
        h = F.relu(h_0)
        h = self.linear(h)
        if self.recurrent_model == 'gru':
            return h, h_0 # Output = (latent matrix, hidden state for encoder)
        else:
            return h, h_0, c_0
class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, recurrent_model='gru', k=2):
        super(Decoder, self).__init__()
        if recurrent_model == 'gru':
            self.recurrent = GConvGRU(latent_dim, hidden_dim, k)
            self.recurrent_model = 'gru'
        else:
            self.recurrent = GConvLSTM(latent_dim, hidden_dim, k)
            self.recurrent_model = 'lstm'
        self.linear = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, z, edge_index, edge_weight, h, c):
        if self.recurrent_model == 'gru':
            h_0 = self.recurrent(z, edge_index, edge_weight, h)
        else:
            h_0, c_0 = self.recurrent(z, edge_index, edge_weight, h, c)
        h = F.relu(h_0)
        h = self.linear(h)
        if self.recurrent_model == 'gru':
            return h, h_0 # Output = (Final Output, hidden state for decoder)
        else:
            return h, h_0, c_0
    
class GraphSeqDiscriminator(torch.nn.Module):
    def __init__(self, node_feat_dim, enc_hidden_dim, enc_latent_dim, obs_len, target_len, num_boids, mode, recurrent_model):
        super(GraphSeqDiscriminator, self).__init__()
        self.encoder = Encoder(node_feat_dim, enc_hidden_dim, enc_latent_dim, recurrent_model)
        self.recurrent_model = recurrent_model
        self.linear = torch.nn.Linear(enc_latent_dim * num_boids, 1)
        self.obs_len = obs_len
        self.target_len = target_len
        self.num_boids = num_boids
        self.mode = mode 
    
    def forward(self, batch_obs_seq, batch_target_seq, h_enc=None, c_enc=None):
        def process_batch_seq(batch_seq, seq_len, h_enc, c_enc, recurrent_model):
            for i in range(seq_len):
                snapshot_batch = batch_seq[i]
                if self.recurrent_model == 'gru':
                    z_batch, h_enc_0 = self.encoder(snapshot_batch.x, 
                                                    snapshot_batch.edge_index, 
                                                    snapshot_batch.edge_attr, 
                                                    h_enc
                                                   )
                    h_enc = h_enc_0
                else:
                    z_batch, h_enc_0, c_enc_0 = self.encoder(snapshot_batch.x, 
                                                    snapshot_batch.edge_index, 
                                                    snapshot_batch.edge_attr, 
                                                    h_enc,
                                                    c_enc
                                                   )
                    h_enc = h_enc_0
                    c_enc = c_enc_0
                    
            if self.recurrent_model == 'gru':
                return z_batch, h_enc
            else:
                return z_batch, h_enc, c_enc
            
        
        if self.recurrent_model == 'gru':
            # Process Observation Sequence (X)
            _, h_enc = process_batch_seq(batch_obs_seq, self.obs_len, h_enc, c_enc, self.recurrent_model)
            # Process Target Sequence (Y or Y_hat)
            z_batch, h_enc = process_batch_seq(batch_target_seq, self.target_len, h_enc, c_enc, self.recurrent_model)
        else:
            # Process Observation Sequence (X)
            _, h_enc, c_enc = process_batch_seq(batch_obs_seq, self.obs_len, h_enc, c_enc, self.recurrent_model)
            # Process Target Sequence (Y or Y_hat)
            z_batch, h_enc, c_enc = process_batch_seq(batch_target_seq, self.target_len, h_enc, c_enc, self.recurrent_model)
            
        
        z_batch = F.leaky_relu(z_batch)
        # Reshaping z_batch from dim (num_boids*batch_size, num_latent) to dim (batch_size, enc_latent_dim*num_boids)
        z_batch = torch.reshape(z_batch, (-1, self.num_boids*z_batch.shape[1])) 
        out_batch = self.linear(z_batch)
        if self.mode == 'gan':
            out_batch = torch.sigmoid(out_batch)
            
        return out_batch

def get_noise(shape, noise_type):
    if noise_type == 'gaussian':
        return torch.randn(*shape).cuda()
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)
    
class GraphSeqGenerator(torch.nn.Module):
    def __init__(self, node_feat_dim, enc_hidden_dim, enc_latent_dim, dec_hidden_dim, obs_len, target_len, num_boids, batch_size, min_max_x, min_max_y, min_max_edge_weight, visual_range, device, recurrent_model):
        super(GraphSeqGenerator, self).__init__()
        self.encoder = Encoder(node_feat_dim, enc_hidden_dim, enc_latent_dim-1, recurrent_model)
        self.decoder = Decoder(enc_latent_dim, dec_hidden_dim, node_feat_dim, recurrent_model)
        self.target_len = target_len
        self.obs_len = obs_len
        self.num_boids = num_boids
        self.batch_size = batch_size
        self.node_feat_dim = node_feat_dim
        self.min_x, self.max_x = min_max_x
        self.min_y, self.max_y = min_max_y
        self.min_edge_weight, self.max_edge_weight = min_max_edge_weight
        self.visual_range = visual_range
        self.device = device
        self.recurrent_model = recurrent_model

    def _compute_edge_index_and_weight(self, y_hat_batch):
        # Grab x and y features
        y_hat_x_batch = y_hat_batch[:, 0] 
        y_hat_y_batch = y_hat_batch[:, 1]

        # Undo normalization
        y_hat_x_batch = y_hat_x_batch * (self.max_x - self.min_x) + self.min_x
        y_hat_y_batch = y_hat_y_batch * (self.max_y - self.min_y) + self.min_y

        coords = torch.stack((y_hat_x_batch, y_hat_y_batch), dim=1)
        
        coords = torch.reshape(coords, (-1, self.num_boids, 2)) # A single dimension may be -1, in which case it’s inferred from the remaining dimensions and the number of elements in input
        distances = torch.cdist(coords, coords, p=2)

        edge_index_list = []
        edge_weight_list = []
        for i in range(distances.shape[0]):
            curr_dist = distances[i]

            indices = torch.where((curr_dist < self.visual_range) & (curr_dist > 0))
            edge_index = torch.vstack((indices[0], indices[1]))
            edge_weight = curr_dist[indices]

            edge_index_list.append(edge_index)
            edge_weight_list.append(edge_weight)

        return edge_index_list, edge_weight_list
    
    def _convertToDataBatch(self, y_hat_batch, y_hat_edge_index_batch, y_hat_edge_attr_batch):
        y_hat_batch = torch.reshape(y_hat_batch, (-1, self.num_boids, self.node_feat_dim))

        data_list = [Data(x=y_hat_batch[i], edge_index=y_hat_edge_index_batch[i], edge_attr=y_hat_edge_attr_batch[i]) for i in range(y_hat_batch.shape[0])]
        return Batch.from_data_list(data_list)
        
    def _add_noise(self, input, user_noise='gaussian'):
        noise_vector = get_noise(shape=(input.shape[0], 1), noise_type=user_noise).to(self.device)
        return torch.cat((input, noise_vector), dim=1)

    def forward(self, obs_seq_batch, h_enc=None, h_dec=None, c_enc=None, c_dec=None):
        # Warmup Section
        for i in range(self.obs_len):
            snapshot_batch = obs_seq_batch[i]
            
            if self.recurrent_model == 'gru':
                z_batch, h_enc_0 = self.encoder(snapshot_batch.x, snapshot_batch.edge_index, snapshot_batch.edge_attr, h_enc, c_enc)
                z_noisy_batch = self._add_noise(z_batch, user_noise='gaussian')
                y_hat_batch, h_dec_0 = self.decoder(z_noisy_batch, snapshot_batch.edge_index, snapshot_batch.edge_attr, h_dec, c_dec)
                h_enc = h_enc_0
                h_dec = h_dec_0
            else:
                z_batch, h_enc_0, c_enc_0 = self.encoder(snapshot_batch.x, snapshot_batch.edge_index, snapshot_batch.edge_attr, h_enc, c_enc)
                z_noisy_batch = self._add_noise(z_batch, user_noise='gaussian')
                y_hat_batch, h_dec_0, c_dec_0 = self.decoder(z_noisy_batch, snapshot_batch.edge_index, snapshot_batch.edge_attr, h_dec, c_dec)
                h_enc = h_enc_0
                h_dec = h_dec_0
                c_enc = c_enc_0
                c_dec = c_dec_0

            

        y_hat_edge_index_batch, y_hat_edge_attr_batch = self._compute_edge_index_and_weight(y_hat_batch)

        pred_snapshot_batch_list = []

        pred_snapshot_batch = self._convertToDataBatch(y_hat_batch, y_hat_edge_index_batch, y_hat_edge_attr_batch)

        pred_snapshot_batch_list.append(pred_snapshot_batch)
        
        
        # Prediction Section
        for _ in range(self.target_len-1):
            y_hat_batch = pred_snapshot_batch.x
            y_hat_edge_index_batch = pred_snapshot_batch.edge_index
            y_hat_edge_attr_batch = pred_snapshot_batch.edge_attr
            if self.recurrent_model == 'gru':
                z_batch, h_enc_0 = self.encoder(y_hat_batch, y_hat_edge_index_batch, y_hat_edge_attr_batch, h_enc, c_enc)
                z_noisy_batch = self._add_noise(z_batch, user_noise='gaussian')
                y_hat_batch, h_dec_0 = self.decoder(z_noisy_batch, y_hat_edge_index_batch, y_hat_edge_attr_batch, h_dec, c_dec)
                h_enc = h_enc_0
                h_dec = h_dec_0
            else:
                z_batch, h_enc_0, c_enc_0 = self.encoder(y_hat_batch, y_hat_edge_index_batch, y_hat_edge_attr_batch, h_enc, c_enc)
                z_noisy_batch = self._add_noise(z_batch, user_noise='gaussian')
                y_hat_batch, h_dec_0, c_dec_0 = self.decoder(z_noisy_batch, y_hat_edge_index_batch, y_hat_edge_attr_batch, h_dec, c_dec)
                h_enc = h_enc_0
                h_dec = h_dec_0
                c_enc = c_enc_0
                c_dec = c_dec_0

            y_hat_edge_index_batch, y_hat_edge_attr_batch = self._compute_edge_index_and_weight(y_hat_batch)

            pred_snapshot_batch = self._convertToDataBatch(y_hat_batch, y_hat_edge_index_batch, y_hat_edge_attr_batch)
            pred_snapshot_batch_list.append(pred_snapshot_batch)

        return pred_snapshot_batch_list





#-----BELOW CODE FROM PYTORCH GEOMETRIC TEMPORAL-----#
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


class GConvLSTM(torch.nn.Module):
    r"""An implementation of the Chebyshev Graph Convolutional Long Short Term Memory
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
        super(GConvLSTM, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalization = normalization
        self.bias = bias
        self._create_parameters_and_layers()
        self._set_parameters()

    def _create_input_gate_parameters_and_layers(self):

        self.conv_x_i = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_i = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.w_c_i = Parameter(torch.Tensor(1, self.out_channels))
        self.b_i = Parameter(torch.Tensor(1, self.out_channels))

    def _create_forget_gate_parameters_and_layers(self):

        self.conv_x_f = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_f = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.w_c_f = Parameter(torch.Tensor(1, self.out_channels))
        self.b_f = Parameter(torch.Tensor(1, self.out_channels))

    def _create_cell_state_parameters_and_layers(self):

        self.conv_x_c = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_c = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.b_c = Parameter(torch.Tensor(1, self.out_channels))

    def _create_output_gate_parameters_and_layers(self):

        self.conv_x_o = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_o = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.w_c_o = Parameter(torch.Tensor(1, self.out_channels))
        self.b_o = Parameter(torch.Tensor(1, self.out_channels))

    def _create_parameters_and_layers(self):
        self._create_input_gate_parameters_and_layers()
        self._create_forget_gate_parameters_and_layers()
        self._create_cell_state_parameters_and_layers()
        self._create_output_gate_parameters_and_layers()

    def _set_parameters(self):
        glorot(self.w_c_i)
        glorot(self.w_c_f)
        glorot(self.w_c_o)
        zeros(self.b_i)
        zeros(self.b_f)
        zeros(self.b_c)
        zeros(self.b_o)

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return C

    def _calculate_input_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        I = self.conv_x_i(X, edge_index, edge_weight, lambda_max=lambda_max)
        I = I + self.conv_h_i(H, edge_index, edge_weight, lambda_max=lambda_max)
        I = I + (self.w_c_i * C)
        I = I + self.b_i
        I = torch.sigmoid(I)
        return I

    def _calculate_forget_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        F = self.conv_x_f(X, edge_index, edge_weight, lambda_max=lambda_max)
        F = F + self.conv_h_f(H, edge_index, edge_weight, lambda_max=lambda_max)
        F = F + (self.w_c_f * C)
        F = F + self.b_f
        F = torch.sigmoid(F)
        return F

    def _calculate_cell_state(self, X, edge_index, edge_weight, H, C, I, F, lambda_max):
        T = self.conv_x_c(X, edge_index, edge_weight, lambda_max=lambda_max)
        T = T + self.conv_h_c(H, edge_index, edge_weight, lambda_max=lambda_max)
        T = T + self.b_c
        T = torch.tanh(T)
        C = F * C + I * T
        return C

    def _calculate_output_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        O = self.conv_x_o(X, edge_index, edge_weight, lambda_max=lambda_max)
        O = O + self.conv_h_o(H, edge_index, edge_weight, lambda_max=lambda_max)
        O = O + (self.w_c_o * C)
        O = O + self.b_o
        O = torch.sigmoid(O)
        return O

    def _calculate_hidden_state(self, O, C):
        H = O * torch.tanh(C)
        return H

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
        C: torch.FloatTensor = None,
        lambda_max: torch.Tensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state and cell state
        matrices are not present when the forward pass is called these are
        initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor, optional)* - Cell state matrix for all nodes.
            * **lambda_max** *(PyTorch Tensor, optional but mandatory if normalization is not sym)* - Largest eigenvalue of Laplacian.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor)* - Cell state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        C = self._set_cell_state(X, C)
        I = self._calculate_input_gate(X, edge_index, edge_weight, H, C, lambda_max)
        F = self._calculate_forget_gate(X, edge_index, edge_weight, H, C, lambda_max)
        C = self._calculate_cell_state(X, edge_index, edge_weight, H, C, I, F, lambda_max)
        O = self._calculate_output_gate(X, edge_index, edge_weight, H, C, lambda_max)
        H = self._calculate_hidden_state(O, C)
        return H, C