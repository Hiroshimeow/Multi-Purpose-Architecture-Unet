import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import glob
import random
from pathlib import Path
from src.datasets import TiledHyperspectralDataset
from tqdm import tqdm
import yaml

class HyperspectralAutoencoder(nn.Module):
    def __init__(self, in_channels=25, latent_dim=3):
        super(HyperspectralAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, latent_dim, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 8, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, in_channels, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_autoencoder(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HyperspectralAutoencoder(
        in_channels=config['model']['params']['in_channels'],
        latent_dim=3
    ).to(device)

    data_cfg = config['data']
    data_dir = Path(data_cfg['dir'])
    cube_dir = data_dir / "cubes_fl32"
    
    all_files = sorted(glob.glob(os.path.join(cube_dir, "*.mat")))
    if not all_files:
        all_files = sorted(glob.glob(os.path.join(cube_dir, "*.h5")))
    if not all_files:
        raise FileNotFoundError(f"No .mat or .h5 files found in {cube_dir}.")
        
    random.seed(42)
    random.shuffle(all_files)
    
    split_idx = int(0.8 * len(all_files))
    train_files, val_files = all_files[:split_idx], all_files[split_idx:]

    # For autoencoder, we don't need z-score stats, default normalization is fine
    config['data']['normalization'] = 'min-max' # Override normalization for autoencoder
    train_dataset = TiledHyperspectralDataset(file_paths=train_files, config=config, augment=True, stats=None)
    val_dataset = TiledHyperspectralDataset(file_paths=val_files, config=config, augment=False, stats=None)

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    print("Starting Autoencoder Training...")
    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0
        for image, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
            image = image.to(device)
            optimizer.zero_grad()
            outputs = model(image)
            loss = criterion(outputs, image)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config['num_epochs']}], Train Loss: {avg_train_loss:.6f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for image, _ in val_loader:
                image = image.to(device)
                outputs = model(image)
                loss = criterion(outputs, image)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Val Loss: {avg_val_loss:.6f}")

    torch.save(model.encoder.state_dict(), 'encoder.pth')
    print("Encoder model saved to encoder.pth")

if __name__ == '__main__':
    with open('configs/run9_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Adjust config for autoencoder training
    config['training']['num_epochs'] = 20
    config['learning_rate'] = 1e-3 # This is a new attribute for the autoencoder
    config['num_epochs'] = config['training']['num_epochs'] # for the train_autoencoder function

    train_autoencoder(config)