import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from early_stopping import EarlyStopping
from state_dataset import StateDataset
from ivae import IVAE, train, load_model

def train_muon_ivae(
        x_train, x_train3d,
        x_test, x_test3d,
        muon_data, muon_test_data,
        epochs=100,
        lr=5e-4,
        weight_decay=None,
        dropout_rate=0.01,
        state_channels=1,
        obs_channels=1,
        num_observations=100,
        latent_dim=32,
        batch_size=16,
        val_batch_size=32,
        freq=1,
        model_path='ivae.pth',
        early_stopping=False,
        parallel=False,
        reload=False,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):

    # Initialize dataset and dataloader
    dataset = StateDataset(data=x_train, data3d=x_train3d, muon_data=muon_data, muon_test_data=muon_test_data, num_samples=len(x_train), num_observations=num_observations, is_muon=True, rand_obs=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    val_dataset = StateDataset(data=x_test, data3d=x_test3d, muon_data=muon_data, muon_test_data=muon_test_data, num_samples=len(x_test), num_observations=num_observations, is_muon=True, rand_obs=True, is_test=True, all_obs=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    # Initialize the model, optimizer
    model = IVAE(state_channels, obs_channels, latent_dim, dropout_rate)
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

        # Training loop
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

        # Saving the model
        torch.save(model.state_dict(), model_path)

    return model


def load_muon_ivae(model_path, device='cuda'):
    device = 'cuda'
    model = IVAE(state_channels=1, obs_channels=1, latent_dim=32, dropout_rate=0.01)
    model = load_model(model, model_path, device)
    return model
