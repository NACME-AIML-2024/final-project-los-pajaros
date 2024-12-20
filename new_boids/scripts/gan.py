

import os
import torch
import json
import random
import matplotlib.pyplot as plt
from scripts.boid_dataset import BoidDatasetLoader, dataLoader
from scripts.utils import temporal_signal_split, create_dirs, HandlerTupleVertical
from scripts.models import GraphSeqGenerator, GraphSeqDiscriminator
from scripts.train import train_step
from scripts.test import test_step

class GanModel:
    def __init__(self, hyperparams, save_dir="results"):
        self.required_hyperparams = [
            "obs_len", "target_len", "batch_size", "lr_generator", "lr_discriminator", 
            "num_epochs", "diversity_k", "device", "label_smoothing", "mode", "scheduler", 
            "gen_enc_hidden_dim", "gen_enc_latent_dim", "gen_dec_hidden_dim", 
            "disc_enc_latent_dim", "recurrent_model"
        ]
        self.hyperparams = hyperparams

        self._check_hyperparameters()

        self.num_boids = 100
        self.visual_range = 75
        self.loader = BoidDatasetLoader()
        self.train_dataset, self.test_dataset = temporal_signal_split(self.loader.get_dataset())
        self.val_dataset, self.test_dataset = temporal_signal_split(self.test_dataset, train_ratio=0.5)
        self.train_batches = dataLoader(self.train_dataset,
                                        window=hyperparams['obs_len']['value'],
                                        horizon=hyperparams['target_len']['value'],
                                        batch_size=hyperparams['batch_size']['value']
                                        )
        self.test_batches = dataLoader(self.test_dataset,
                                      window=hyperparams['obs_len']['value'],
                                      horizon=hyperparams['target_len']['value'],
                                      batch_size=hyperparams['batch_size']['value']
                                     )
        self.val_batches = dataLoader(self.val_dataset,
                                      window=hyperparams['obs_len']['value'],
                                      horizon=hyperparams['target_len']['value'],
                                      batch_size=hyperparams['batch_size']['value']
                                     )
        self.device = torch.device(self.hyperparams['device']['value'])
        
        
        # Initialize models
        self.generator = GraphSeqGenerator(node_feat_dim=4,
                                           enc_hidden_dim=self.hyperparams['gen_enc_hidden_dim']['value'],
                                           enc_latent_dim=self.hyperparams['gen_enc_latent_dim']['value'],
                                           dec_hidden_dim=self.hyperparams['gen_dec_hidden_dim']['value'],
                                           obs_len=self.hyperparams['obs_len']['value'],
                                           target_len=self.hyperparams['target_len']['value'],
                                           num_boids=self.num_boids,
                                           batch_size=self.hyperparams['batch_size']['value'],
                                           min_max_x=(self.loader.min_features[0], self.loader.max_features[0]),
                                           min_max_y=(self.loader.min_features[1], self.loader.max_features[1]),
                                           min_max_edge_weight=(self.loader.min_edge_weight, self.loader.max_edge_weight),
                                           visual_range=self.visual_range,
                                           device=self.device,
                                           recurrent_model=self.hyperparams['recurrent_model']['value']
                                           )
        
        self.discriminator = GraphSeqDiscriminator(node_feat_dim=4,
                                                   enc_hidden_dim=self.hyperparams['gen_dec_hidden_dim']['value'],
                                                   enc_latent_dim=self.hyperparams['disc_enc_latent_dim']['value'],
                                                   obs_len=self.hyperparams['obs_len']['value'],
                                                   target_len=self.hyperparams['target_len']['value'],
                                                   num_boids=self.num_boids,
                                                   mode=self.hyperparams['mode']['value'],
                                                   recurrent_model=self.hyperparams['recurrent_model']['value']
                                                   )
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        # Set up save directories
        self.save_dir = save_dir
        self.train_dir = os.path.join(save_dir, "train")
        self.train_plots_dir = os.path.join(self.train_dir, "plots")
        self.models_dir = os.path.join(self.train_dir, "models")
        self.train_losses_dir = os.path.join(self.train_dir, "losses")
        
        self.val_dir = os.path.join(save_dir, "validation")
        self.checkpoint_dir = os.path.join(self.val_dir, "checkpoints")
        self.val_plots_dir = os.path.join(self.val_dir, "val_plots")

        self.test_dir = os.path.join(self.save_dir, "test")
        self.test_plots_dir = os.path.join(self.test_dir, "plots")
        self.test_losses_dir = os.path.join(self.test_dir, "losses")


        create_dirs([self.save_dir, 
                     self.train_dir, self.train_plots_dir, self.models_dir, self.train_losses_dir,
                     self.val_dir, self.checkpoint_dir, self.val_plots_dir,
                     self.test_dir, self.test_plots_dir, self.test_losses_dir]
                   )

    def _check_hyperparameters(self):
        missing_params = [param for param in self.required_hyperparams if param not in self.hyperparams]
        if missing_params:
            raise ValueError(f"Missing hyperparameters: {', '.join(missing_params)}")
        #Check if implemented the mode they requested
        if self.hyperparams["mode"]["value"] not in set(['gan', 'lsgan']):
            raise ValueError(f'Mode: {self.hyperparams["mode"]["value"]} not available')
        
    def save_metric(self, metrics, metric_type, save_in_train_dir=True):
        """Helper method to save losses as JSON."""
        if save_in_train_dir:
            metric_path = os.path.join(self.train_losses_dir, f"{metric_type}_metrics.json")
        else:
            metric_path = os.path.join(self.test_losses_dir, f"{metric_type}_metrics.json")
        with open(metric_path, "w") as outfile:
            json.dump(metrics, outfile)

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
        if self.hyperparams['mode']['value'] == 'gan':
            criterion = torch.nn.BCELoss()
        if self.hyperparams['mode']['value'] == 'lsgan':
            criterion = torch.nn.MSELoss()
            
        optimizerG = torch.optim.Adam(self.generator.parameters(), 
                                      lr=self.hyperparams['lr_generator']['value']
                                     )
        optimizerD = torch.optim.Adam(self.discriminator.parameters(), 
                                      lr=self.hyperparams['lr_discriminator']['value']
                                     )
        if self.hyperparams['scheduler']['value']:
            scheduler_gen = torch.optim.lr_scheduler.StepLR(optimizer=optimizerG,
                                                            step_size=25,
                                                            gamma=0.1
                                                           )
            scheduler_disc = torch.optim.lr_scheduler.StepLR(optimizer=optimizerD,
                                                             step_size=25,
                                                             gamma=0.1
                                                            )
        else:
            scheduler_gen = torch.optim.lr_scheduler.StepLR(optimizer=optimizerG,
                                                            step_size=25,
                                                            gamma=1
                                                           )
            
            scheduler_disc = torch.optim.lr_scheduler.StepLR(optimizer=optimizerD,
                                    step_size=25,
                                    gamma=1
                                    )
        
        NUM_EPOCHS = self.hyperparams['num_epochs']['value']

        metrics = ["Generator Loss", "Discriminator Loss", "Variety Loss"]
        if self.hyperparams['mode']['value'] == "gan":
            metrics.extend(["D(x)", "D(G(z)) Before Discriminator Is Updated", 
                            "D(G(z)) After Discriminator Is Updated"]
                          )
        metrics_over_steps = {metric: [] for metric in metrics}
        
        for epoch in range(NUM_EPOCHS):
            print(f'Epoch: {epoch+1}/{NUM_EPOCHS}')
            train_metrics = train_step(train_batches=self.train_batches,
                                       generator=self.generator,
                                       discriminator=self.discriminator,
                                       criterion=criterion,
                                       optimizer_gen=optimizerG,
                                       optimizer_disc=optimizerD,
                                       scheduler_gen=scheduler_gen,
                                       scheduler_disc=scheduler_disc,
                                       batch_size=self.hyperparams['batch_size']['value'],
                                       device=self.device,
                                       label_smoothing=self.hyperparams['label_smoothing']['value'],
                                       k=self.hyperparams['diversity_k']['value'],
                                       mode=self.hyperparams['mode']['value']
                                      )
            # Save training metrics in metrics_over_steps
            for key, value in train_metrics.items():
                metrics_over_steps[key].extend(value)
            # Checkpoint for model
            if epoch % 10 == 0:
                #Save test plot 
                gen_loss, disc_loss, variety_loss = test_step(test_batches=self.val_batches, 
                     generator=self.generator, 
                     discriminator=self.discriminator, 
                     criterion=criterion, 
                     device=self.device, 
                     batch_size=self.hyperparams['batch_size']['value'], 
                     num_boids=self.num_boids, 
                     test_plot_dir=os.path.join(self.val_plots_dir, f"predicted_vs_ground_truth_at_epoch_{epoch}.png"),
                     k=self.hyperparams['diversity_k']['value'], 
                     mode=self.hyperparams['mode']['value']
                    )
                #Save checkpoint
                torch.save({'epoch': epoch,
                            'gen_state_dict': self.generator.state_dict(),
                            'optimizer_gen_state_dict': optimizerG.state_dict(),
                            'disc_state_dict': self.discriminator.state_dict(),
                            'optimizer_disc_state_dict': optimizerD.state_dict(),
                            'generator val loss': gen_loss,
                            'discriminator val loss': disc_loss,
                            'variety val loss': variety_loss,
                            }, os.path.join(self.checkpoint_dir, f"model_at_epoch_{epoch}.pth"))

        # Save training losses and model
        self.save_metric(metrics_over_steps, "train")
        torch.save(self.generator, os.path.join(self.models_dir, 'generator.pt'))
        torch.save(self.discriminator, os.path.join(self.models_dir, 'discriminator.pt'))

        # Plot loss curves
        self.plot_loss(metrics_over_steps, "Generator and Discriminator Loss During Training", 
                       'Iterations', 'Loss', ['Generator Loss', 'Discriminator Loss'], 'gen_and_disc_train_loss.png')
        if self.hyperparams['mode']['value'] == 'gan':
            self.plot_loss(metrics_over_steps, "D(x) and D(G(z)) During Training", 
                           'Iterations', 'D(·)', ['D(x)', 'D(G(z)) Before Discriminator Is Updated', 'D(G(z)) After Discriminator Is Updated'], 'disc_output_train.png')
        
        self.plot_loss(metrics_over_steps, "Variety Loss During Training", 
                       'Iterations', 'Loss', ['Variety Loss'], 'variety_train_loss.png')

    def test(self):
        NUM_EPOCHS = self.hyperparams['num_epochs']['value']
        if self.hyperparams['mode']['value'] == 'gan':
            criterion = torch.nn.BCELoss()
        if self.hyperparams['mode']['value'] == 'lsgan':
            criterion = torch.nn.MSELoss()
        gen_loss, disc_loss, variety_loss = test_step(test_batches=self.test_batches, 
                                                      generator=self.generator, 
                                                      discriminator=self.discriminator, 
                                                      criterion=criterion,
                                                      device=self.device,
                                                      batch_size=self.hyperparams['batch_size']['value'],
                                                      num_boids=self.num_boids,
                                                      test_plot_dir=os.path.join(self.test_plots_dir, f"predicted_vs_ground_truth_at_epoch_{NUM_EPOCHS}.png"),
                                                      k=self.hyperparams['diversity_k']['value'],
                                                      mode=self.hyperparams['mode']['value']
                                                     )
        
        
