import torch
import torch.nn.functional as F
from tqdm import tqdm


# Range btw [0.7, 1.0]
def smooth_positive_labels(y, device):
    return torch.sub(y, 0.3) + torch.mul(torch.rand(size=y.size(), device=device), 0.3)
# Range btw [0, 0.3]        
def smooth_negative_labels(y, device):                                 
    return y + torch.mul(torch.rand(size=y.size(), device=device), 0.3)
def train_gan(train_batches, num_epochs, generator, discriminator, criterion, optimizer_gen, optimizer_disc, batch_size, device, k=1):
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
    

    generator.train()
    discriminator.train()
    
    losses_over_iterations = {
        'Generator': [],
        'Discriminator': [],
        'Dx': [],
        'D_G_z1': [],
        'D_G_z2': [],
        'Variety': [],
    }

    for epoch in range(num_epochs):
        print(f'Epoch: {epoch+1}/{num_epochs}')
        for i, batch in enumerate(tqdm(train_batches, desc='Training')):
            obs_seq_batch = batch[0]
            target_seq_batch = batch[1]

            obs_seq_batch = [obs_batch.to(device) for obs_batch in obs_seq_batch]
            target_seq_batch = [target_batch.to(device) for target_batch in target_seq_batch]

            y_hat_seq_batch = generator(obs_seq_batch, None, None)

            # Discriminator Step
            discriminator.zero_grad()

            output_real_batch, _ = discriminator(target_seq_batch, None)
            smooth_pos_labels = smooth_positive_labels(torch.ones(min(batch_size, output_real_batch.shape[0]), device=device), device=device)
            err_disc_real_batch = criterion(output_real_batch.squeeze(), smooth_pos_labels)
            err_disc_real_batch.backward()
            D_x = output_real_batch.mean().item()

            # Need to detach the y_hat_seq_batch
            # More details here: https://community.deeplearning.ai/t/why-should-we-detach-the-discriminators-input/53220
            output_fake_batch, _ = discriminator([y_hat_seq_batch[i].detach() for i in range(len(y_hat_seq_batch))], None)
            smooth_neg_labels = smooth_negative_labels(torch.zeros(min(batch_size, output_fake_batch.shape[0]), device=device), device=device)
            err_disc_fake_batch = criterion(output_fake_batch.squeeze(), smooth_neg_labels)
            err_disc_fake_batch.backward()

            D_G_z1 = output_fake_batch.mean().item() #  Before D is updated | Side Note: Assigned it to wrong thing beforehand, now its good
            err_disc = err_disc_fake_batch + err_disc_real_batch
            optimizer_disc.step()

            # Generator Step
            generator.zero_grad()
            err_gen = 0

            # Calculate Variety Loss
            k_L2_Losses = []
            for _ in range(k):
                y_hat_seq_k_batch = generator(obs_seq_batch, None, None)
                curr_L2_loss = sum([F.mse_loss(y_hat_seq_k_batch[i].x, target_seq_batch[i].x) for i in range(len(y_hat_seq_k_batch))])
                k_L2_Losses.append(curr_L2_loss)
            min_L2_loss = min(k_L2_Losses)
            err_gen += min_L2_loss

            # Forward pass through the discriminator with the generated data
            output_fake_batch, _ = discriminator(y_hat_seq_batch, None)
            # Calculate the generator's loss based on how well it fooled the discriminator
            #print(output_fake_batch)
            output = criterion(output_fake_batch.squeeze(), torch.ones(min(batch_size, output_fake_batch.shape[0]), device=device))
            err_gen += output
            # Backward pass for generator
            err_gen.backward()
            D_G_z2 = output_fake_batch.mean().item() # After D is updated
            optimizer_gen.step()

            # if i % 10 == 0:
            #     print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f | %.4f \tLoss_variety: %.4f'
            #       % (epoch, num_epochs, i, len(train_batches),\
            #          err_disc.item(), err_gen.item(), D_x, D_G_z1, D_G_z2, min_L2_loss.item()))
                
            losses_over_iterations['Generator'].append(err_gen.cpu().item())
            losses_over_iterations['Discriminator'].append(err_disc.cpu().item())
            losses_over_iterations['Dx'].append(D_x)
            losses_over_iterations['D_G_z1'].append(D_G_z1)
            losses_over_iterations['D_G_z2'].append(D_G_z2)
            losses_over_iterations['Variety'].append(min_L2_loss.cpu().item())

    return losses_over_iterations


        
        

    
