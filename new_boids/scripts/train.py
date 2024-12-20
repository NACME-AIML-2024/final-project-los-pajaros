# This should have different modes depending on the type of model
# List of modes: "gan", "lsgan", "wsgan", "wsgan_gp"
# Main differences between these modes are the objective functions
# How to handle these differences?
# Output of the generator should be fine
# Output of the discriminator is different, specifically the activation function
# What I can do is apply the activation function after the function
import torch
from tqdm import tqdm
import torch.nn.functional as F

def smooth_positive_labels(y, device):
    return torch.sub(y, 0.3) + torch.mul(torch.rand(size=y.size(), device=device), 0.3)

def generator_step(generator, discriminator, optimizer_gen, criterion, obs_seq_batch, target_seq_batch, target_hat_seq_batch, k, batch_size, device, mode="gan"):
    generator.zero_grad()
    err_gen = 0
    metrics = []
    # 1. Calculate Variety Loss
    k_L2_Losses = []
    for _ in range(k):
        y_hat_seq_k_batch = generator(obs_seq_batch)
        curr_L2_loss = sum([F.mse_loss(y_hat_seq_k_batch[i].x, target_seq_batch[i].x) for i in range(len(y_hat_seq_k_batch))])
        k_L2_Losses.append(curr_L2_loss)
    min_L2_loss = min(k_L2_Losses)
    err_gen += min_L2_loss
    
    # 2. Calculate Generator Loss
    # Forward pass through the discriminator with the generated data
    if (mode == "gan") or (mode == "lsgan"):
        output_fake_batch = discriminator(obs_seq_batch, target_hat_seq_batch)
        real_labels = torch.ones(min(batch_size, output_fake_batch.shape[0]), device=device)
        output = criterion(output_fake_batch.squeeze(), real_labels)
    # TODO: if (mode == "wgan") or (mode == "wgan-gp):
    err_gen += output
    # Backward pass for generator
    err_gen.backward()
    # Save generator loss
    metrics.append(err_gen.cpu().item())
    # Save other metrics
    metrics.append(min_L2_loss.cpu().item())
    if mode == "gan":
        D_G_z2 = output_fake_batch.mean().item() # After D is updated
        metrics.append(D_G_z2)
    optimizer_gen.step()
    
    return metrics
    

def discriminator_step(discriminator, optimizer_disc, criterion, obs_seq_batch, target_seq_batch, target_hat_seq_batch, batch_size, device, mode="gan", label_smoothing=False):
    discriminator.zero_grad()
    metrics = []
    output_real_batch = discriminator(obs_seq_batch, target_seq_batch)

    positive_labels = torch.ones(min(batch_size, output_real_batch.shape[0]), device=device)
    if label_smoothing: # Only apply label smoothing on real labels for the discriminator step
        positive_labels = smooth_positive_labels(positive_labels, device=device)
    err_disc_real_batch = criterion(output_real_batch.squeeze(), positive_labels)
    err_disc_real_batch.backward()

    # Need to detach the target_hat_seq_batch
    # More details here: https://community.deeplearning.ai/t/why-should-we-detach-the-discriminators-input/53220
    output_fake_batch = discriminator(obs_seq_batch, [target_hat_seq_batch[i].detach() for i in range(len(target_hat_seq_batch))])

    negative_labels = torch.zeros(min(batch_size, output_fake_batch.shape[0]), device=device)
    err_disc_fake_batch = criterion(output_fake_batch.squeeze(), negative_labels)

    err_disc_fake_batch.backward()
    # TODO: if (mode == "wgan") or (mode == "wgan-gp):
    
    err_disc = err_disc_fake_batch + err_disc_real_batch
    metrics.append(err_disc.cpu().item())
    if mode == "gan":
        D_x = output_real_batch.mean().item()
        metrics.append(D_x)
        D_G_z1 = output_fake_batch.mean().item()
        metrics.append(D_G_z1)
    optimizer_disc.step()
    return metrics
    

def train_step(train_batches, generator, discriminator, criterion, optimizer_gen, optimizer_disc, scheduler_gen, scheduler_disc, batch_size, device, label_smoothing, k=1, mode="gan"):
    
    generator.train()
    discriminator.train()
    metrics = ["Generator Loss", "Discriminator Loss", "Variety Loss"]
    if mode == "gan":
        metrics.extend(["D(x)", "D(G(z)) Before Discriminator Is Updated", "D(G(z)) After Discriminator Is Updated"])
    metrics_over_steps = {metric: [] for metric in metrics}
    
    for batch in tqdm(train_batches, desc='Train'):
        
        obs_seq_batch = batch[0]
        target_seq_batch = batch[1]

        obs_seq_batch = [obs_batch.to(device) for obs_batch in obs_seq_batch]
        target_seq_batch = [target_batch.to(device) for target_batch in target_seq_batch]

        target_hat_seq_batch = generator(obs_seq_batch)
        
        # Discriminator Step
        disc_metrics = discriminator_step(discriminator=discriminator, 
                                          optimizer_disc=optimizer_disc, 
                                          criterion=criterion, 
                                          obs_seq_batch=obs_seq_batch, 
                                          target_seq_batch=target_seq_batch, 
                                          target_hat_seq_batch=target_hat_seq_batch, 
                                          batch_size=batch_size, 
                                          device=device, 
                                          mode=mode, 
                                          label_smoothing=label_smoothing
                                         )
            
        # Generator Step
        gen_metrics = generator_step(generator=generator, 
                                     discriminator=discriminator, 
                                     optimizer_gen=optimizer_gen, 
                                     criterion=criterion, 
                                     obs_seq_batch=obs_seq_batch, 
                                     target_seq_batch=target_seq_batch, 
                                     target_hat_seq_batch=target_hat_seq_batch, 
                                     k=k, 
                                     batch_size=batch_size, 
                                     device=device, 
                                     mode=mode
                                    )
            
        metrics_over_steps['Generator Loss'].append(gen_metrics[0])
        metrics_over_steps['Discriminator Loss'].append(disc_metrics[0])
        metrics_over_steps['Variety Loss'].append(gen_metrics[1])
        if mode == "gan":
            metrics_over_steps['D(x)'].append(disc_metrics[1])
            metrics_over_steps['D(G(z)) Before Discriminator Is Updated'].append(disc_metrics[2])
            metrics_over_steps['D(G(z)) After Discriminator Is Updated'].append(gen_metrics[2])
        
    scheduler_disc.step()
    scheduler_gen.step()

    return metrics_over_steps