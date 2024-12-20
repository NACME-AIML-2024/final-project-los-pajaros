import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from scripts.utils import HandlerTupleVertical
import os

def test_step(test_batches, generator, discriminator, criterion, device, batch_size, num_boids, test_plot_dir, k=1, mode="gan"):
    generator.eval()  
    discriminator.eval()  

    metrics = ["Generator Loss", 
               "Discriminator Loss", 
               "Variety Loss"
              ]
    if mode == "gan":
        metrics.extend(["D(x)", "D(G(z)) Before Discriminator Is Updated", "D(G(z)) After Discriminator Is Updated"])
        
    metrics_over_steps = {metric: [] for metric in metrics}

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for batch in tqdm(test_batches, desc='Test'):
            obs_seq_batch = batch[0]
            target_seq_batch = batch[1]

            obs_seq_batch = [obs_batch.to(device) for obs_batch in obs_seq_batch]
            target_seq_batch = [target_batch.to(device) for target_batch in target_seq_batch]

            # Generate predictions
            y_hat_seq_batch = generator(obs_seq_batch)

            # Evaluate the discriminator on real data
            output_real_batch = discriminator(obs_seq_batch, target_seq_batch)
            real_labels = torch.ones(min(batch_size, output_real_batch.shape[0]), device=device)

            # Ensure the dimensions of `output_real_batch` and `real_labels` match
            if output_real_batch.dim() == 1:
                output_real_batch = output_real_batch.unsqueeze(0)
            if real_labels.dim() == 1:
                real_labels = real_labels.unsqueeze(0)

            err_disc_real_batch = criterion(output_real_batch.squeeze(), real_labels.squeeze())
            D_x = output_real_batch.mean().item()

            # Evaluate the discriminator on fake data
            output_fake_batch = discriminator(obs_seq_batch, y_hat_seq_batch)
            fake_labels = torch.zeros(min(batch_size, output_fake_batch.shape[0]), device=device)

            # Ensure the dimensions of `output_fake_batch` and `fake_labels` match
            if output_fake_batch.dim() == 1:
                output_fake_batch = output_fake_batch.unsqueeze(0)
            if fake_labels.dim() == 1:
                fake_labels = fake_labels.unsqueeze(0)

            err_disc_fake_batch = criterion(output_fake_batch.squeeze(), fake_labels.squeeze())
            D_G_z1 = output_fake_batch.mean().item()  # Before any updates
            err_disc = err_disc_fake_batch + err_disc_real_batch

            # Calculate variety loss
            k_L2_Losses = []
            for _ in range(k):
                y_hat_seq_k_batch = generator(obs_seq_batch, None, None)
                curr_L2_loss = sum(
                    [F.mse_loss(y_hat_seq_k_batch[i].x, target_seq_batch[i].x) for i in range(len(y_hat_seq_k_batch))]
                )
                k_L2_Losses.append(curr_L2_loss)
            min_L2_loss = min(k_L2_Losses)

            # Calculate generator loss based on the discriminator's output
            output = criterion(output_fake_batch.squeeze(), real_labels.squeeze())
            D_G_z2 = output_fake_batch.mean().item()  # After D is evaluated
            err_gen = min_L2_loss + output

            # Store evaluation metrics
            metrics_over_steps['Generator Loss'].append(err_gen.cpu().item())
            metrics_over_steps['Discriminator Loss'].append(err_disc.cpu().item())
            metrics_over_steps['Variety Loss'].append(min_L2_loss.cpu().item())
            if mode == "gan":
                metrics_over_steps['D(x)'].append(D_x)
                metrics_over_steps['D(G(z)) Before Discriminator Is Updated'].append(D_G_z1)
                metrics_over_steps['D(G(z)) After Discriminator Is Updated'].append(D_G_z2)
                
        gen_loss = (1/len(metrics_over_steps['Generator Loss'])) * sum(metrics_over_steps['Generator Loss'])
        disc_loss = (1/len(metrics_over_steps['Discriminator Loss'])) * sum(metrics_over_steps['Discriminator Loss'])
        variety_loss = (1/len(metrics_over_steps['Variety Loss'])) * sum(metrics_over_steps['Variety Loss'])
        
    print(f"Generator Loss: {gen_loss} | Discriminator Loss: {disc_loss} | Variety Loss: {variety_loss}")
    save_test_plot(generator, test_batches, device, test_plot_dir, num_boids)
    return gen_loss, disc_loss, variety_loss

def save_test_plot(generator, test_batches, device, test_plot_dir, num_boids):
    # Logic to plot and save test predictions vs actual values.
    single_batch = test_batches[random.randint(0, len(test_batches)-1)]
    generator.eval()  

    with torch.no_grad():
        obs_seq_batch = single_batch[0]
        target_seq_batch = single_batch[1]

        obs_seq_batch = [obs_batch.to(device) for obs_batch in obs_seq_batch]
        target_seq_batch = [target_batch.to(device) for target_batch in target_seq_batch]

        y_hat_seq_batch = generator(obs_seq_batch, None, None)

        obs_seq_batch = [obs_batch.to('cpu') for obs_batch in obs_seq_batch]
        target_seq_batch = [target_batch.to('cpu') for target_batch in target_seq_batch]
        y_hat_seq_batch = [y_hat_batch.to('cpu') for y_hat_batch in y_hat_seq_batch]
        
        y_hat_seq_xy_batch = [torch.reshape(y_hat_batch.x[:, 0:2], (-1, num_boids, 2)) for y_hat_batch in y_hat_seq_batch]
        target_seq_xy_batch = [torch.reshape(target_batch.x[:, 0:2], (-1, num_boids, 2)) for target_batch in target_seq_batch]
        obs_seq_xy_batch = [torch.reshape(obs_batch.x[:, 0:2], (-1, num_boids, 2)) for obs_batch in obs_seq_batch]

        rows = 4
        cols = 4
        fig, axs = plt.subplots(rows, cols, figsize=(16, 16), layout='constrained')
        fig.suptitle('Predicted Vs. Ground Truth', y=1.05, fontsize=16)
        fig.supxlabel('x')
        fig.supylabel('y')
        for i in range(rows):
            for j in range(cols):
                random_i = random.randint(0, y_hat_seq_xy_batch[0].shape[0]-1)
                random_y_hat_seq_xy = [y_hat_xy_batch[random_i] for y_hat_xy_batch in y_hat_seq_xy_batch]
                random_target_seq_xy = [target_xy_batch[random_i] for target_xy_batch in target_seq_xy_batch]
                random_obs_seq_xy = [obs_xy_batch[random_i] for obs_xy_batch in obs_seq_xy_batch]
                random_boids = [random.randint(0, num_boids-1) for _ in range(4)]
            
                observed_lines = []
                pred_lines = []
                ground_truth = []
                for boid_id in random_boids:
                    line1, = axs[i,j].plot([random_obs_seq_xy[i][boid_id][0] for i in range(len(random_obs_seq_xy))], 
                                     [random_obs_seq_xy[i][boid_id][1] for i in range(len(random_obs_seq_xy))], linestyle='dashed',
                                     label='Observed Trajectory'
                                    )
                    prev_color = line1.get_color()
                    line2,  = axs[i,j].plot([random_y_hat_seq_xy[i][boid_id][0] for i in range(len(random_y_hat_seq_xy))], 
                             [random_y_hat_seq_xy[i][boid_id][1] for i in range(len(random_y_hat_seq_xy))], 
                             label='Predicted Trajectory', 
                             color=prev_color
                            )
                    line3,  = axs[i,j].plot([random_target_seq_xy[i][boid_id][0] for i in range(len(random_target_seq_xy))], 
                             [random_target_seq_xy[i][boid_id][1] for i in range(len(random_target_seq_xy))], linestyle='dashed', 
                             label='Ground Truth',
                             color='blue')
                    observed_lines.append(line1)
                    pred_lines.append(line2)
                    ground_truth.append(line3)
        fig.legend([tuple(observed_lines), tuple(pred_lines), tuple([ground_truth[0]])], 
                   ['Observed Trajectory', 'Predicted Trajectory', 'Ground Truth'], 
                   handler_map = {tuple : HandlerTupleVertical()}, loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=3)
        fig.savefig(test_plot_dir, bbox_inches='tight')
        plt.close(fig)