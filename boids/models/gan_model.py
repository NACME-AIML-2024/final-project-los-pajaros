

import os
import torch
import json
import random
import matplotlib.pyplot as plt
from scripts.dataset import BoidDatasetLoader, dataLoader
from scripts.utils import temporal_signal_split
from scripts.models import GraphSeqGenerator, GraphSeqDiscriminator
from scripts.train import train_gan
from scripts.test import test_gan

loader = BoidDatasetLoader()
dataset = loader.get_dataset()

train_dataset, test_dataset = temporal_signal_split(dataset)
# Utility function to create directories
def create_dirs(directory_structure):
    for path in directory_structure:
        os.makedirs(path, exist_ok=True)

class GanModel:
    def __init__(self, hyperparams, save_dir="results"):
        self.required_hyperparams = [
            "obs_len", "target_len", "batch_size", "lr_generator", "lr_discriminator", 
            "num_epochs", "diversity_k", "device", "label_smoothing"
        ]
        self.hyperparams = hyperparams

        self._check_hyperparams()

        self.num_boids = 100
        self.visual_range = 75
        self.train_batches = dataLoader(train_dataset,
                                        window=hyperparams['obs_len']['value'],
                                        horizon=hyperparams['target_len']['value'],
                                        batch_size=hyperparams['batch_size']['value']
                                        )
        
        self.device = torch.device(self.hyperparams['device']['value'])
        
        # Initialize models
        self.generator = GraphSeqGenerator(node_feat_dim=4,
                                           enc_hidden_dim=32,
                                           enc_latent_dim=16,
                                           dec_hidden_dim=32,
                                           obs_len=self.hyperparams['obs_len']['value'],
                                           target_len=self.hyperparams['target_len']['value'],
                                           num_boids=self.num_boids,
                                           batch_size=self.hyperparams['batch_size']['value'],
                                           min_max_x=(loader.min_features[0], loader.max_features[0]),
                                           min_max_y=(loader.min_features[1], loader.max_features[1]),
                                           min_max_edge_weight=(loader.min_edge_weight, loader.max_edge_weight),
                                           visual_range=self.visual_range,
                                           device=self.device,
                                           )
        
        self.discriminator = GraphSeqDiscriminator(node_feat_dim=4,
                                                   enc_hidden_dim=32,
                                                   enc_latent_dim=16,
                                                   obs_len=self.hyperparams['obs_len']['value'],
                                                   target_len=self.hyperparams['target_len']['value'],
                                                   num_boids=self.num_boids,
                                                   )
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        # Set up save directories
        self.save_dir = save_dir
        self.train_dir = os.path.join(save_dir, "train")
        self.train_plots_dir = os.path.join(self.train_dir, "plots")
        self.models_dir = os.path.join(self.train_dir, "models")
        self.train_losses_dir = os.path.join(self.train_dir, "losses")

        self.test_dir = os.path.join(self.save_dir, "test")
        self.test_plots_dir = os.path.join(self.test_dir, "plots")
        self.test_losses_dir = os.path.join(self.test_dir, "losses")


        create_dirs([self.save_dir, self.train_dir, self.train_plots_dir, self.models_dir, self.train_losses_dir, self.test_dir, self.test_plots_dir, self.test_losses_dir])

    def _check_hyperparameters(self):
        missing_params = [param for param in self.required_hyperparams if param not in self.hyperparams]
        if missing_params:
            raise ValueError(f"Missing hyperparameters: {', '.join(missing_params)}")
        
    def save_loss(self, losses, loss_type, save_in_train_dir=True):
        """Helper method to save losses as JSON."""
        if save_in_train_dir:
            loss_path = os.path.join(self.train_losses_dir, f"{loss_type}_losses.json")
        else:
            loss_path = os.path.join(self.test_losses_dir, f"{loss_type}_losses.json")
        with open(loss_path, "w") as outfile:
            json.dump(losses, outfile)

    def plot_loss(self, losses, title, xlabel, ylabel, labels, filename, save_in_train_dir=True):
        """General method for plotting and saving loss plots."""
        plt.figure(figsize=(10, 5))
        plt.title(title)
        for label in labels:
            plt.plot(losses[label], label=label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        if save_in_train_dir:
            plt.savefig(os.path.join(self.train_plots_dir, filename))
        else:
            plt.savefig(os.path.join(self.test_plots_dir, filename))
        plt.close()

    def train(self):
        criterion = torch.nn.BCELoss()
        optimizerG = torch.optim.Adam(self.generator.parameters(), lr=self.hyperparams['lr_generator']['value'])
        optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr=self.hyperparams['lr_discriminator']['value'])

        train_losses = train_gan(train_batches=self.train_batches,
                                 num_epochs=self.hyperparams['num_epochs']['value'],
                                 generator=self.generator,
                                 discriminator=self.discriminator,
                                 criterion=criterion,
                                 optimizer_gen=optimizerG,
                                 optimizer_disc=optimizerD,
                                 batch_size=self.hyperparams['batch_size']['value'],
                                 device=self.device,
                                 k=self.hyperparams['diversity_k']['value'],
                                 )

        # Save training losses and model
        self.save_loss(train_losses, "train")
        torch.save(self.generator, os.path.join(self.models_dir, 'generator.pt'))
        torch.save(self.discriminator, os.path.join(self.models_dir, 'discriminator.pt'))

        # Plot loss curves
        self.plot_loss(train_losses, "Generator and Discriminator Loss During Training", 
                       'Iterations', 'Loss', ['Generator', 'Discriminator'], 'gen_and_disc_train_loss.png')
        
        self.plot_loss(train_losses, "D(x) and D(G(z)) During Training", 
                       'Iterations', 'D(·)', ['D(x)', 'D(G(z)) Before Discriminator Is Updated', 'D(G(z)) After Discriminator Is Updated'], 'disc_output_train.png')
        
        self.plot_loss(train_losses, "Variety Loss During Training", 
                       'Iterations', 'Loss', ['Variety'], 'variety_train_loss.png')

    def test(self):
        test_batches = dataLoader(test_dataset, 
                                  window=self.hyperparams['obs_len']['value'], 
                                  horizon=self.hyperparams['target_len']['value'], 
                                  batch_size=self.hyperparams['batch_size']['value']
                                  )

        test_losses = test_gan(test_batches=test_batches,
                               generator=self.generator,
                               discriminator=self.discriminator,
                               criterion=self.criterion,
                               device=self.device,
                               batch_size=self.hyperparams['batch_size']['value'],
                               k=self.hyperparams['diversity_k']['value'],
                               )

        # Save test losses
        self.save_loss(test_losses, "test", save_in_train_dir=False)

        # Save test result plot
        self._save_test_plot(test_batches)

    def _save_test_plot(self, test_batches):
        # Logic to plot and save test predictions vs actual values.
        single_batch = test_batches[random.randint(0, len(test_batches)-1)]
        self.generator.eval()  

        with torch.no_grad():
            obs_seq_batch = single_batch[0]
            target_seq_batch = single_batch[1]

            obs_seq_batch = [obs_batch.to(self.device) for obs_batch in obs_seq_batch]
            target_seq_batch = [target_batch.to(self.device) for target_batch in target_seq_batch]

            y_hat_seq_batch = self.generator(obs_seq_batch, None, None)

            obs_seq_batch = [obs_batch.to('cpu') for obs_batch in obs_seq_batch]
            target_seq_batch = [target_batch.to('cpu') for target_batch in target_seq_batch]
            y_hat_seq_batch = [y_hat_batch.to('cpu') for y_hat_batch in y_hat_seq_batch]
            
            y_hat_seq_xy_batch = [torch.reshape(y_hat_batch.x[:, 0:2], (-1, self.num_boids, 2)) for y_hat_batch in y_hat_seq_batch]
            target_seq_xy_batch = [torch.reshape(target_batch.x[:, 0:2], (-1, self.num_boids, 2)) for target_batch in target_seq_batch]
            obs_seq_xy_batch = [torch.reshape(obs_batch.x[:, 0:2], (-1, self.num_boids, 2)) for obs_batch in obs_seq_batch]

            random_i = random.randint(0, y_hat_seq_xy_batch[0].shape[0]-1)
            random_y_hat_seq_xy = [y_hat_xy_batch[random_i] for y_hat_xy_batch in y_hat_seq_xy_batch]
            random_target_seq_xy = [target_xy_batch[random_i] for target_xy_batch in target_seq_xy_batch]
            random_obs_seq_xy = [obs_xy_batch[random_i] for obs_xy_batch in obs_seq_xy_batch]
            random_boids = [random.randint(0, self.num_boids-1) for _ in range(5)]

            plt.title('Predicted Vs. Actual Paths')
            for boid_id in random_boids:
                plt.plot([random_obs_seq_xy[i][boid_id][0] for i in range(len(random_obs_seq_xy))], 
                         [random_obs_seq_xy[i][boid_id][1] for i in range(len(random_obs_seq_xy))], marker='^', 
                         label=f'Observed Path for Boid #{boid_id}')
                plt.plot([random_target_seq_xy[i][boid_id][0] for i in range(len(random_target_seq_xy))], 
                         [random_target_seq_xy[i][boid_id][1] for i in range(len(random_target_seq_xy))], marker='.', 
                         label=f'Actual Future Path for Boid #{boid_id}')
                plt.plot([random_y_hat_seq_xy[i][boid_id][0] for i in range(len(random_y_hat_seq_xy))], 
                         [random_y_hat_seq_xy[i][boid_id][1] for i in range(len(random_y_hat_seq_xy))], marker='x', 
                         label=f'Predicted Future Path for Boid #{boid_id}')
            plt.legend(loc=(1.04, 0))
            plt.savefig(os.path.join(self.test_plots_dir, 'predicted_vs_actual_paths.png'))
            plt.close()
