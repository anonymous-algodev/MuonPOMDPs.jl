import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
from ivae import load_model

class StateEncoder(nn.Module):
    def __init__(self, state_channels, latent_dim, dropout_rate):
        super(StateEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
        )
        self.fc = nn.Linear(1000, 2 * latent_dim)

    def forward(self, state):
        h = self.encoder(state)
        params = self.fc(h)
        mean, log_var = torch.chunk(params, 2, dim=1)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, state_channels, dropout_rate):
        super(Decoder, self).__init__()
        self.state_channels = state_channels
        self.decoder = nn.Sequential(
            nn.Linear(1000 + latent_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 28*28),
            nn.Sigmoid()
        )

    def forward(self, z, obs_h):
        z = torch.cat([z, obs_h], dim=1)
        x = self.decoder(z)
        x = x.view(-1, self.state_channels, 28, 28)
        return x


class ObsEncoder(nn.Module):
    def __init__(self, obs_channels, latent_dim, dropout_rate):
        super(ObsEncoder, self).__init__()
        self.obs_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
        )
        self.obs_fc = nn.Linear(1000, 2 * latent_dim)

    def forward(self, obs):
        h = self.obs_encoder(obs)
        params = self.obs_fc(h)
        mean, log_var = torch.chunk(params, 2, dim=1)
        return h, mean, log_var

class IVAE_MNIST(nn.Module):
    def __init__(self, state_channels, obs_channels, latent_dim, dropout_rate):
        super(IVAE_MNIST, self).__init__()
        self.state_encoder = StateEncoder(state_channels, latent_dim, dropout_rate)
        self.decoder = Decoder(latent_dim, state_channels, dropout_rate)
        self.obs_encoder = ObsEncoder(obs_channels, latent_dim, dropout_rate)
        self.latent_dim = latent_dim

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mean + std * epsilon

    def forward(self, state, obs):
        qs_mean, qs_log_var = self.state_encoder(state)
        z = self.reparameterize(qs_mean, qs_log_var)
        obs_h, qo_mean, qo_log_var = self.obs_encoder(obs)
        recon_state = self.decoder(z, obs_h)
        return recon_state, qs_mean, qs_log_var, qo_mean, qo_log_var

    def plot(self):
        device = next(self.parameters()).device
        obs = torch.full((1,1,28,28), -1).float().to(device)

        mnist_states = self.sample(obs, thresholded=False, m=16).detach().cpu()

        fig = plt.figure(figsize=(5,5))
        for i in range(16):
            plt.subplot(4,4,i+1)
            plt.imshow(mnist_states[i].reshape(28,28).detach().cpu(), cmap='gray')
            fig.patch.set_alpha(0)
            plt.gca().get_xaxis().set_ticks([])
            plt.gca().get_yaxis().set_ticks([])
            plt.gca().spines['bottom'].set_color('white')
            plt.gca().spines['left'].set_color('white')
            plt.gca().spines['top'].set_color('white')
            plt.gca().spines['right'].set_color('white')
        plt.tight_layout()
        plt.savefig('mnist-training-samples.png')
        plt.clf()

    
    def sample(self, obs_or_belief, m=1, thresholded=True):
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            if isinstance(obs_or_belief, tuple):
                qo_mean, qo_log_var = obs_or_belief
            else:
                obs = obs_or_belief
                obs = obs.to(device)
                h, qo_mean, qo_log_var = self.obs_encoder(obs)
            if m > 1:
                h = h.repeat(m, 1)
                qo_mean = qo_mean.repeat(m, 1)
                qo_log_var = qo_log_var.repeat(m, 1)
        with torch.no_grad():
            z = self.reparameterize(qo_mean, qo_log_var)
            sampled_states = self.decoder(z, h)
        if thresholded:
            return (sampled_states > 0.5).float()
        else:
            return sampled_states

    def update(self, obs):
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            obs = obs.to(device)
            h, qo_mean, qo_log_var = self.obs_encoder(obs)
        return h, qo_mean, qo_log_var

def ivae_loss(recon_state, state, qs_mean, qs_log_var, qo_mean, qo_log_var, beta=1.0):
    recon_loss = F.binary_cross_entropy(recon_state, state, reduction='sum')
    kl_div = 0.5 * torch.sum(
        qo_log_var - qs_log_var +
        (torch.exp(qs_log_var) + (qs_mean - qo_mean) ** 2) / torch.exp(qo_log_var) - 1
    )
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
        recon_state, qs_mean, qs_log_var, qo_mean, qo_log_var = model(state, obs)
        loss = ivae_loss(recon_state, state, qs_mean, qs_log_var, qo_mean, qo_log_var)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if progress_trigger:
            pbar.set_postfix({f'[{epoch}/{epochs}] Loss': total_loss / len(dataloader.dataset)})
    average_loss = total_loss / len(dataloader.dataset)
    model.plot()

    avg_val_loss = np.inf
    return avg_val_loss


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from early_stopping import EarlyStopping
from state_dataset import StateDataset
from torchvision import datasets, transforms
from state_pixel_dataset import StatePixelObsDataset

class MNISTImagesOnly(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.mnist = datasets.MNIST(root=root, train=train, transform=transform, download=download)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, _ = self.mnist[idx]  # Ignore the label
        return image

class MNISTData(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.mnist = datasets.MNIST(root=root, train=train, transform=transform, download=download)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]
        return image, label

class Binarize:
    def __call__(self, tensor):
        return (tensor > 0.5).float()

def train_muon_ivae_mnist(
        epochs=10,
        lr=1e-3,
        weight_decay=None,
        dropout_rate=0.0,
        state_channels=1,
        obs_channels=1,
        num_samples=None, # None indicates len(dataset)
        num_observations=196,
        latent_dim=32,
        batch_size=256,
        val_batch_size=256,
        freq=1,
        model_path='ivae-mnist.pth',
        early_stopping=False,
        parallel=False,
        reload=False,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):

    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    mnist_train_image_dataset = MNISTImagesOnly(root='data', train=True, download=False, transform=mnist_transform)
    if num_samples is None:
        num_samples = len(mnist_train_image_dataset)
    mnist_train_dataset = StatePixelObsDataset(data=mnist_train_image_dataset, num_samples=num_samples, num_observations=num_observations, grid=True, force_num_obs=False)
    dataloader = DataLoader(mnist_train_dataset, batch_size=batch_size, shuffle=True)

    mnist_test_image_dataset = MNISTImagesOnly(root='data', train=False, download=False, transform=mnist_transform)
    mnist_test_dataset = StatePixelObsDataset(data=mnist_test_image_dataset, num_samples=len(mnist_test_image_dataset), num_observations=num_observations, grid=True, force_num_obs=False)
    val_dataloader = DataLoader(mnist_test_dataset, batch_size=val_batch_size, shuffle=False)

    model = IVAE_MNIST(state_channels, obs_channels, latent_dim, dropout_rate)
    if parallel:
        model = nn.DataParallel(model)
    model = model.to(device)

    if reload:
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        if weight_decay is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if early_stopping:
            early_stopper = EarlyStopping(patience=np.inf, delta=0.0, verbose=False, path=model_path)
            best_loss = early_stopper.best_loss
        else:
            best_loss = None

        for epoch in range(1, epochs + 1):
            avg_val_loss = train(model, dataloader, val_dataloader, optimizer, device, epoch, epochs, best_loss=best_loss, freq=freq)
            if early_stopping:
                early_stopper(avg_val_loss, model)
                best_loss = early_stopper.best_loss
                if early_stopper.early_stop:
                    print("Early stopping triggered. Loading the best model.")
                    model.load_state_dict(torch.load(early_stopper.path))
                    break
            else:
                torch.save(model.state_dict(), model_path)

        if early_stopping:
            model.load_state_dict(torch.load(model_path))

        torch.save(model.state_dict(), model_path)

    return model


def load_mnist_ivae(model_path, latent_dim=100, device='cuda'):
    device = 'cuda'
    model = IVAE_MNIST(state_channels=1, obs_channels=1, latent_dim=latent_dim, dropout_rate=0.0)
    model = load_model(model, model_path, device)
    return model
