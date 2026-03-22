import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import pickle
import os
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRIOR_DATA_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "data", "muon_priors.pkl"))

class PickleDataset(Dataset):
    def __init__(self, data, transform=None, is_3d=False):
        self.data = data
        self.transform = transform
        self.is_3d = is_3d

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx] # (80,80,60)

        if not self.is_3d:
            sample = np.rot90(sample.max(axis=2), k=1, axes=(0,1))

        sample = sample.copy() # avoid negative strides

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToFloat:
    def __call__(self, tensor):
        return tensor.float()


class TransposeTransform:
    def __call__(self, tensor):
        return tensor.permute(1,2,0)


def load_prior_datasets(prior_data_path=PRIOR_DATA_PATH):
    with open(prior_data_path, 'rb') as f:
        m_ensemble_combined = pickle.load(f)

    transform = transforms.Compose([
        transforms.ToTensor(),
        ToFloat(),
    ])
    prior_dataset = PickleDataset(m_ensemble_combined, transform=transform, is_3d=False)

    transform3d = transforms.Compose([
        transforms.ToTensor(),
        ToFloat(),
        TransposeTransform(),
    ])
    prior_dataset3d = PickleDataset(m_ensemble_combined, transform=transform3d, is_3d=True)

    return prior_dataset, prior_dataset3d


def dataset_to_tensors(dataset):
    tensors_list = [dataset[i] for i in range(len(dataset))]
    combined_tensor = torch.stack(tensors_list)
    return combined_tensor


def augment_images(X, n=1, combined=True):
    # Assume 'X' is your (num_priors, 1, 80, 80) tensor
    augmentation = transforms.RandomAffine(
        degrees=(0, 0),         # Rotation
        translate=(0.25, 0.1),  # Translate up to (x%, y%) of image dimensions
    )

    augmented_images = []
    is_3d = X.shape[1] != 1

    for _ in range(n):
        # Apply the augmentation to each image in 'X'
        if is_3d:
            augmented_batch = torch.stack([augmentation(img.permute(2, 0, 1)).permute(1, 2, 0) for img in X])
        else:
            augmented_batch = torch.stack([augmentation(img) for img in X])
        augmented_images.append(augmented_batch)

    # Concatenate all augmented images along the batch dimension
    augmented_X = torch.cat(augmented_images, dim=0)
    if combined:
        combined_X = torch.cat([X, augmented_X], dim=0)
        return combined_X.float()
    else:
        return augmented_X.float()
    

def load_datasets(seed=0, use_augmented=False, train_frac=0.9):
    prior_dataset, prior_dataset3d = load_prior_datasets()
    
    random.seed(seed) # Important for PRECOMPUTED muon values

    if use_augmented:
        X3d = dataset_to_tensors(prior_dataset3d)
        torch.manual_seed(seed)
        aX3d = augment_images(X3d, n=1) # 22,000 = (1000 + n*1000)
        aX = aX3d.max(dim=3).values.rot90(dims=(1,2)).unsqueeze(1)
        num_train = int(len(aX3d) * train_frac)
        data_perm = random.sample(range(len(aX3d)), len(aX3d))
        X = aX
        X3d = aX3d
    else:
        num_train = int(len(prior_dataset) * train_frac)
        data_perm = random.sample(range(len(prior_dataset)), len(prior_dataset))
        X = dataset_to_tensors(prior_dataset)
        X3d = dataset_to_tensors(prior_dataset3d)
    x_train = X[data_perm[:num_train], :].numpy()
    x_test = X[data_perm[num_train:], :].numpy()
    x_train3d = X3d[data_perm[:num_train], :].numpy()
    x_test3d = X3d[data_perm[num_train:], :].numpy()
    # TODO: Add 3d versions of rotation...
    return x_train, x_train3d, x_test, x_test3d
