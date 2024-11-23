

import torch
import json
import random
import matplotlib.pyplot as plt
from scripts.dataset import BoidDatasetLoader, dataLoader
from scripts.utils import temporal_signal_split
from scripts.models import GraphSeqGenerator, GraphSeqDiscriminator
from scripts.Train import train_gan
from scripts.Test import test_gan


loader = BoidDatasetLoader()
dataset = loader.get_dataset()

train_dataset, test_dataset = temporal_signal_split(dataset)

OBS_LENGTH = 8
PRED_LENGTH = 8 
BATCH_SIZE = 128
NUM_BOIDS = 100
VISUAL_RANGE = 75
K = 5

train_batches = dataLoader(train_dataset,
                           window=OBS_LENGTH,
                           horizon=PRED_LENGTH,
                           batch_size=BATCH_SIZE 
                           )

device = torch.device("cuda")
generator = GraphSeqGenerator(node_feat_dim=4,
                              enc_hidden_dim=32,
                              enc_latent_dim=16,
                              dec_hidden_dim=32,
                              obs_len=OBS_LENGTH,
                              target_len=PRED_LENGTH,
                              num_boids=NUM_BOIDS,
                              batch_size=BATCH_SIZE,
                              min_max_x=(loader.min_features[0], loader.max_features[0]),
                              min_max_y=(loader.min_features[1], loader.max_features[1]),
                              min_max_edge_weight=(loader.min_edge_weight, loader.max_edge_weight),
                              visual_range=VISUAL_RANGE,
                              device=device,
                            )
generator.to(device)

discriminator = GraphSeqDiscriminator(node_feat_dim=4,
                                      enc_hidden_dim=32,
                                      enc_latent_dim=16,
                                      obs_len=OBS_LENGTH,
                                      target_len=PRED_LENGTH,
                                      num_boids=NUM_BOIDS,
                                    )
discriminator.to(device)

criterion = torch.nn.BCELoss()
optimizerG = torch.optim.Adam(generator.parameters(), lr=1e-3)
optimizerD = torch.optim.Adam(discriminator.parameters(), lr=5e-5)

train_losses = train_gan(train_batches=train_batches,
              num_epochs=50,
              generator=generator,
              discriminator=discriminator,
              criterion=criterion,
              optimizer_gen=optimizerG,
              optimizer_disc=optimizerD,
              batch_size=BATCH_SIZE,
              device=device,
              k=K,
              )

with open("train_losses.json", "w") as outfile: 
    json.dump(train_losses, outfile)

torch.save(generator, 'generator.pt')
torch.save(discriminator, 'discriminator.pt')
# generator = torch.load('generator.pt' , weights_only=False)
# discriminator = torch.load('discriminator.pt', weights_only=False)

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(train_losses['Generator'],label="Generator")
plt.plot(train_losses['Discriminator'],label="Discriminator")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
plt.savefig('gen_and_disc_train_loss.png')

plt.figure(figsize=(10,5))
plt.title("D(x) and D(G(z)) During Training")
plt.plot(train_losses['Dx'],label="D(x)")
plt.plot(train_losses['D_G_z1'],label="D(G(z)) Before Discriminator Is Updated")
plt.plot(train_losses['D_G_z2'],label="D(G(z)) After Discriminator Is Updated")
plt.axhline(y=0.5, color='r', linestyle='--')
plt.xlabel("Iterations")
plt.ylabel("D(·)") 
# Want the discriminator to get to a point where it guesses btw real and fake (0.5)
plt.legend()
plt.show()
plt.savefig('disc_output_train.png')

plt.figure(figsize=(10,5))
plt.title("Variety Loss During Training")
plt.plot(train_losses['Variety'])
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()
plt.savefig('variety_train_loss.png')

test_batches = dataLoader(test_dataset, 
                          window=OBS_LENGTH, 
                          horizon=PRED_LENGTH, 
                          batch_size=BATCH_SIZE
                          )

test_losses = test_gan(test_batches=test_batches,
                  generator=generator,
                  discriminator=discriminator,
                  criterion=criterion,
                  device=device,
                  batch_size=BATCH_SIZE,
                  k=K,
                 )
with open("test_losses.json", "w") as outfile: 
    json.dump(test_losses, outfile)

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
    
    y_hat_seq_xy_batch = [torch.reshape(y_hat_batch.x[:, 0:2], (-1, NUM_BOIDS, 2)) for y_hat_batch in y_hat_seq_batch]
    target_seq_xy_batch = [torch.reshape(target_batch.x[:, 0:2], (-1, NUM_BOIDS, 2)) for target_batch in target_seq_batch]
    obs_seq_xy_batch = [torch.reshape(obs_batch.x[:, 0:2], (-1, NUM_BOIDS, 2)) for obs_batch in obs_seq_batch]

    
    # Pick a random sequence from the batch
    random_i = random.randint(0, y_hat_seq_xy_batch[0].shape[0]-1)
    random_y_hat_seq_xy = [y_hat_xy_batch[random_i] for y_hat_xy_batch in y_hat_seq_xy_batch]
    random_target_seq_xy = [target_xy_batch[random_i] for target_xy_batch in target_seq_xy_batch]
    random_obs_seq_xy = [obs_xy_batch[random_i] for obs_xy_batch in obs_seq_xy_batch]
    random_boids = [random.randint(0, NUM_BOIDS-1) for _ in range(5)]
    plt.title('Predicted Vs. Actual Paths')
    for boid_id in random_boids:
        plt.plot([random_obs_seq_xy[i][boid_id][0] for i in range(len(random_obs_seq_xy))], [random_obs_seq_xy[i][boid_id][1] for i in range(len(random_obs_seq_xy))], marker='^', label=f'Observed Past Path For Boid #{boid_id}')
        plt.plot([random_target_seq_xy[i][boid_id][0] for i in range(len(random_target_seq_xy))], [random_target_seq_xy[i][boid_id][1] for i in range(len(random_target_seq_xy))], marker='.', label=f'Actual Future Path For Boid #{boid_id}')
        plt.plot([random_y_hat_seq_xy[i][boid_id][0] for i in range(len(random_y_hat_seq_xy))], [random_y_hat_seq_xy[i][boid_id][1] for i in range(len(random_y_hat_seq_xy))], marker='x', label=f'Predicted Future Path For Boid #{boid_id}')

    plt.legend(loc=(1.04, 0))
    plt.show()