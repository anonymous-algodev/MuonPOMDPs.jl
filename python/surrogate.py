import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from early_stopping import EarlyStopping
from state_dataset import StateDataset
from tqdm import tqdm
import random

class MuonObservationModel(nn.Module):
    def __init__(self, output_channels=1, output_type='binary', dropout_rate=0.0):
        super(MuonObservationModel, self).__init__()
        self.output_channels = output_channels
        self.output_type = output_type
        self.dropout_rate = dropout_rate

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate)
        )

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate)
        )
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate)
        )
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate)
        )

        # Upsample from 80x80 to 100x100
        self.upconv0 = nn.ConvTranspose2d(64, 32, kernel_size=21, stride=1, padding=0)
        self.dec0 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate)
        )

        # Upsample from 100x100 to 200x200
        self.upconv_final = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.dec_final = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate)
        )

        # Final Convolution
        self.final_conv = nn.Conv2d(16, output_channels, kernel_size=1)

        # Output activation
        if output_type == 'continuous':
            self.output_activation = nn.Identity()  # Linear activation
        elif output_type == 'categorical':
            self.output_activation = nn.Softmax(dim=1)
        elif output_type == 'binary':
            self.output_activation = nn.Sigmoid()
        else:
            raise ValueError("Unsupported output type. Choose from 'continuous', 'categorical', or 'binary'.")

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)          # [batch, 64, 80, 80]
        p1 = self.pool(e1)         # [batch, 64, 40, 40]

        e2 = self.enc2(p1)         # [batch, 128, 40, 40]
        p2 = self.pool(e2)         # [batch, 128, 20, 20]

        e3 = self.enc3(p2)         # [batch, 256, 20, 20]
        p3 = self.pool(e3)         # [batch, 256, 10, 10]

        e4 = self.enc4(p3)         # [batch, 512, 10, 10]

        # Decoder path
        up3 = self.upconv3(e4)     # [batch, 256, 20, 20]
        cat3 = torch.cat([up3, e3], dim=1)  # [batch, 512, 20, 20]
        d3 = self.dec3(cat3)       # [batch, 256, 20, 20]

        up2 = self.upconv2(d3)     # [batch, 128, 40, 40]
        cat2 = torch.cat([up2, e2], dim=1)  # [batch, 256, 40, 40]
        d2 = self.dec2(cat2)       # [batch, 128, 40, 40]

        up1 = self.upconv1(d2)     # [batch, 64, 80, 80]
        cat1 = torch.cat([up1, e1], dim=1)  # [batch, 128, 80, 80]
        d1 = self.dec1(cat1)       # [batch, 64, 80, 80]

        # Upsample from 80x80 to 100x100
        up0 = self.upconv0(d1)     # [batch, 32, 100, 100]
        d0 = self.dec0(up0)        # [batch, 32, 100, 100]

        # Upsample from 100x100 to 200x200
        up_final = self.upconv_final(d0)  # [batch, 16, 200, 200]
        d_final = self.dec_final(up_final)  # [batch, 16, 200, 200]

        # Output layer
        out = self.final_conv(d_final)  # [batch, output_channels, 200, 200]

        # Apply output activation
        out = self.output_activation(out)

        return out


def muon_obs_loss(decoded_obs, obs, output_type):
    if output_type == 'continuous':
        # Negative Log Likelihood for Gaussian distribution with uncertainty between [0, 1]
        recon_loss = F.mse_loss(decoded_obs, obs, reduction='mean')
    elif output_type == 'categorical':
        # Cross Entropy Loss for categorical values
        recon_loss = F.cross_entropy(decoded_obs, obs, reduction='mean')
    elif output_type == 'binary':
        recon_loss = F.binary_cross_entropy(decoded_obs, obs, reduction='mean')
    else:
        raise ValueError("Unsupported output type. Choose from 'continuous', 'categorical', or 'binary'.")

    return recon_loss


def train_obs(model, dataloader, val_dataloader, epochs=10, lr=1e-3, freq=1, output_freq=None, scheduler=None, weight_decay=1e-1, patience=5, early_stopping=False, show_progress=False, device='cpu'):
    """
    Train the CVAE model with optional learning rate scheduler.

    Args:
        model (nn.Module): The CVAE model to train.
        dataloader (DataLoader): DataLoader providing the training data.
        epochs (int): Number of training epochs.
        lr (float): Initial learning rate for the optimizer.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler.
        weight_decay (float): L2-regularization strength
        device (str): Device to train on ('cpu' or 'cuda').
    """

    if weight_decay is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler is not None:
        optimizer = scheduler(optimizer)  # Initialize scheduler with optimizer

    model.to(device)

    if early_stopping:
        early_stopper = EarlyStopping(patience=patience, verbose=True, path='best_model.pth')

    if show_progress:
        pbar = tqdm(dataloader)
    else:
        pbar = dataloader

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for (x, obs) in pbar:
            x = x.to(device)
            obs = obs.to(device)
            optimizer.zero_grad()
            obs_est = model(x)

            loss = muon_obs_loss(obs_est, obs, model.output_type)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader.dataset)
        
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_loss)
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = optimizer.param_groups[0]['lr']
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
                scheduler.step()
        
        # ######################
        # # Validate the model #
        # ######################
        if show_progress:
            val_pbar = tqdm(val_dataloader)
        else:
            val_pbar = val_dataloader

        model.eval()  # Set model to evaluation mode
        val_loss = 0
        with torch.no_grad():  # No need to track gradients
            for (x, obs) in val_pbar:
                x = x.to(device)
                obs = obs.to(device)
                optimizer.zero_grad()
                obs_est = model(x)
                loss = muon_obs_loss(obs_est, obs, model.output_type)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_dataloader.dataset)

        if epoch % freq == 0:
            if show_progress:
                pbar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.10f}, Val. Loss: {avg_val_loss:.10f}, Learning Rate: {current_lr:.6f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.10f}, Val. Loss: {avg_val_loss:.10f}, Learning Rate: {current_lr:.6f}")

        # if output_freq is not None and epoch % output_freq == 0:
        #     plot_obs_surrogate(model, output_suffix="training")

        if early_stopping:
            # Early Stopping
            early_stopper(avg_val_loss, model)
            
            if early_stopper.early_stop:
                print("Early stopping triggered. Loading the best model.")
                model.load_state_dict(torch.load(early_stopper.path))
                break

    return model


def train_obs_model(
        x_train, x_train3d,
        x_test, x_test3d,
        epochs=5000,
        num_samples=900,
        batch_size=100,
        lr=1e-4,
        weight_decay=None,
        dropout_rate=0.0,
        patience=5,
        early_stopping=False,
        output_type='binary',
        output_freq=10,
        freq=10,
        show_progress=False,
        seed=0,
        model_path='obs_model.pth'):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset and dataloader
    train_dataset = StateDataset(data=x_train, data3d=x_train3d, num_samples=num_samples, image_size=80, num_observations=100, force_num_obs=True, is_muon=True)
    test_dataset = StateDataset(data=x_test, data3d=x_test3d, num_samples=100, image_size=80, num_observations=100, force_num_obs=True, is_muon=True, is_test=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)

    # Muon observation surrogate model
    obs_model = MuonObservationModel(dropout_rate=dropout_rate, output_type=output_type).to(device)

    # Train the model
    trained_obs_model = train_obs(obs_model, train_dataloader, test_dataloader, output_freq=output_freq, freq=freq, lr=lr, epochs=epochs, weight_decay=weight_decay, patience=patience, early_stopping=early_stopping, show_progress=show_progress, device=device)

    # Save the trained model
    torch.save(trained_obs_model.state_dict(), model_path)
    print(f"Training complete. Model saved as '{model_path}'.")

    return trained_obs_model


def load_obs_model(model_path='obs_model.pth', output_type='binary', device='cuda:0'):
    obs_model = MuonObservationModel(output_type=output_type).to(device)
    obs_model.load_state_dict(torch.load(model_path, map_location=device))
    return obs_model
