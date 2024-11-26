import torch
from tqdm import tqdm
import torch.nn.functional as F

def test_gan(test_batches, generator, discriminator, criterion, device, batch_size, k=1):
    
    generator.eval()  
    discriminator.eval()  

    evaluation_metrics = {
        'Generator': [],
        'Discriminator': [],
        'Dx': [],
        'D_G_z1': [],
        'D_G_z2': [],
        'Variety': [],
    }

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for i, batch in enumerate(tqdm(test_batches, desc='Testing')):
            obs_seq_batch = batch[0]
            target_seq_batch = batch[1]

            obs_seq_batch = [obs_batch.to(device) for obs_batch in obs_seq_batch]
            target_seq_batch = [target_batch.to(device) for target_batch in target_seq_batch]

            # Generate predictions
            y_hat_seq_batch = generator(obs_seq_batch, None, None)

            # Evaluate the discriminator on real data
            output_real_batch, _ = discriminator(target_seq_batch, None)
            real_labels = torch.ones(min(batch_size, output_real_batch.shape[0]), device=device)

            # Ensure the dimensions of `output_real_batch` and `real_labels` match
            if output_real_batch.dim() == 1:
                output_real_batch = output_real_batch.unsqueeze(0)
            if real_labels.dim() == 1:
                real_labels = real_labels.unsqueeze(0)

            err_disc_real_batch = criterion(output_real_batch.squeeze(), real_labels.squeeze())
            D_x = output_real_batch.mean().item()

            # Evaluate the discriminator on fake data
            output_fake_batch, _ = discriminator(y_hat_seq_batch, None)
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
            evaluation_metrics['Generator'].append(err_gen.item())
            evaluation_metrics['Discriminator'].append(err_disc.item())
            evaluation_metrics['Dx'].append(D_x)
            evaluation_metrics['D_G_z1'].append(D_G_z1)
            evaluation_metrics['D_G_z2'].append(D_G_z2)
            evaluation_metrics['Variety'].append(min_L2_loss.item())

            if i % 20 == 0:
                print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f | %.4f \tLoss_variety: %.4f'
                      % (i, len(test_batches), err_disc.item(), err_gen.item(), D_x, D_G_z1, D_G_z2, min_L2_loss.item()))

    return evaluation_metrics



