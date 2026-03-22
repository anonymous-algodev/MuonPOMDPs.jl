import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class StateEncoder(nn.Module):
    def __init__(self, state_channels, latent_dim, dropout_rate):
        super(StateEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(state_channels, 32, kernel_size=4, stride=2, padding=1),  # 80x80 -> 40x40
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 40x40 -> 20x20
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 20x20 -> 10x10
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_encoder = nn.Sequential(    
            nn.Linear(128 * 10 * 10, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, state):
        params = self.encoder(state)
        h = self.fc_encoder(params)
        return h


class Decoder(nn.Module):
    def __init__(self, latent_dim, state_channels, dropout_rate):
        super(Decoder, self).__init__()
        self.decoder_fc = nn.Sequential(
            nn.Linear(256 + 2 * latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128 * 10 * 10),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 10x10 -> 20x20
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 20x20 -> 40x40
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.ConvTranspose2d(32, state_channels, kernel_size=4, stride=2, padding=1),  # 40x40 -> 80x80
            nn.Sigmoid()  # Assuming binary images
        )

    def forward(self, z):
        x = self.decoder_fc(z)
        x = x.view(-1, 128, 10, 10)
        x = self.decoder_conv(x)
        return x


class ObsEncoder(nn.Module):
    def __init__(self, obs_channels, latent_dim, dropout_rate):
        super(ObsEncoder, self).__init__()
        self.obs_encoder = nn.Sequential(
            nn.Conv2d(in_channels=obs_channels, out_channels=16, kernel_size=5, stride=1, padding=2),  # (200, 200) -> (200, 200)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.AvgPool2d(kernel_size=2, stride=2),  # (200, 200) -> (100, 100)

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),  # (100, 100) -> (100, 100)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.AvgPool2d(kernel_size=2, stride=2),  # (100, 100) -> (50, 50)

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0),  # (50, 50) -> (46, 46)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.AvgPool2d(kernel_size=2, stride=2),  # (46, 46) -> (23, 23)

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),  # (23, 23) -> (21, 21)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.AvgPool2d(kernel_size=2, stride=2),  # (21, 21) -> (10, 10)

            nn.Flatten(),
        )
        self.fc_obs_encoder = nn.Linear(128 * 10 * 10, 256)

    def forward(self, obs):
        params = self.obs_encoder(obs)
        h = self.fc_obs_encoder(params)
        return h


class CVAE(nn.Module):
    def __init__(self, state_channels, obs_channels, latent_dim, dropout_rate):
        super(CVAE, self).__init__()
        self.state_encoder = StateEncoder(state_channels, latent_dim, dropout_rate)
        self.decoder = Decoder(latent_dim, state_channels, dropout_rate)
        self.obs_encoder = ObsEncoder(obs_channels, latent_dim, dropout_rate)
        self.latent_dim = latent_dim

        self.fc_mu = nn.Linear(2 * 256, 2 * latent_dim)
        self.fc_logvar = nn.Linear(2 * 256, 2 * latent_dim)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mean + std * epsilon

    def encode(self, state, obs):
        state_h = self.state_encoder(state)
        obs_h = self.obs_encoder(obs)
        h = torch.cat([state_h, obs_h], dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def decode(self, z, obs):
        obs_enc = self.obs_encoder(obs)
        zso = torch.cat([z, obs_enc], dim=1)
        recon_state = self.decoder(zso)
        return recon_state

    def forward(self, state, obs):
        mu, logvar = self.encode(state, obs)
        z = self.reparameterize(mu, logvar)
        recon_state = self.decode(z, obs)
        return recon_state, mu, logvar

    def sample(self, obs_or_belief, m=1, thresholded=True):
        # Example of sampling from the model after training
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            z = torch.randn(m, 2 * self.latent_dim).to(device)
            obs_enc = self.obs_encoder(obs_or_belief)
            if m > 1:
                obs_enc = obs_enc.repeat(m, 1)
            zso = torch.cat([z, obs_enc], dim=1)
            sampled_states = self.decoder(zso)
        if thresholded:
            return (sampled_states > 0.5).float()
        else:
            return sampled_states

    def update(self, obs):
        return obs


def cvae_loss(recon_state, state, observation, mu, logvar, beta=1.0):
    # Reconstruction loss (binary cross-entropy for binary images)
    recon_loss = F.binary_cross_entropy(recon_state, state, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta*kl_div


def train(model, dataloader, val_dataloader, optimizer, device, epoch, epochs, best_loss=None, freq=1):
    model.train()
    total_loss = 0
    progress_trigger = epoch % freq == 0
    if progress_trigger:
        pbar = tqdm(dataloader)
    else:
        pbar = dataloader
    for (state, obs) in pbar:
        state = state.to(device)
        obs = obs.to(device)
        optimizer.zero_grad()
        recon_state, mu, logvar = model(state, obs)
        loss = cvae_loss(recon_state, state, obs, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if progress_trigger:
            pbar.set_postfix({f'[{epoch}/{epochs}] Loss': total_loss / len(dataloader.dataset)})
    average_loss = total_loss / len(dataloader.dataset)

    ######################
    # Validate the model #
    ######################
    model.eval()  # Set model to evaluation mode
    val_loss = 0
    with torch.no_grad():
        if progress_trigger:
            pbar = tqdm(val_dataloader)
        else:
            pbar = val_dataloader
        for (state, obs) in pbar:
            state = state.to(device)
            obs = obs.to(device)
            recon_state, mu, logvar = model(state, obs)
            loss = cvae_loss(recon_state, state, obs, mu, logvar)
            val_loss += loss.item()
            avg_val_loss = val_loss / len(val_dataloader.dataset)
            if progress_trigger:
                if best_loss is not None:
                    pbar.set_postfix({f'[{epoch}/{epochs}] Loss': average_loss, 'Val. Loss': avg_val_loss, 'Best Val. Loss': best_loss})
                else:
                    pbar.set_postfix({f'[{epoch}/{epochs}] Loss': average_loss, 'Val. Loss': avg_val_loss})

    return avg_val_loss
