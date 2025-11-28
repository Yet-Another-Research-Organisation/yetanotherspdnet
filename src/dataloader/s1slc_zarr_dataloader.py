"""
Dataloader optimisé pour S1SLC avec Zarr
The S1SLC is a complex dataset with real and imaginary parts. In .utils we provide a conversion script to convert the original .npy files into a Zarr file with real and imaginary parts stored as separate channels for each polarization (HH and HV).
"""

import numpy as np
import zarr
from typing import List, Tuple, Optional, Dict
from torch.utils.data import Dataset, DataLoader
import torch
import os


def get_dataset_info_zarr(zarr_path: str) -> Dict:
    """
    Récupère les informations sur le dataset Zarr
    
    Args:
        zarr_path: Chemin vers le fichier Zarr
        
    Returns:
        Dict avec 'cities', 'num_samples_per_city', 'data_shape', 'num_classes'
    """
    zarr_root = zarr.open(zarr_path, mode='r')
    
    cities = list(zarr_root.keys())
    num_samples_per_city = {}
    all_labels = []
    data_shape = None
    
    for city in cities:
        n_samples = zarr_root[city]['data'].shape[0]
        num_samples_per_city[city] = n_samples
        
        if data_shape is None:
            data_shape = zarr_root[city]['data'].shape[1:]
        
        # Charger les labels pour compter les classes
        labels = zarr_root[city]['labels'][:]
        all_labels.extend(labels)
    
    unique_labels = np.unique(all_labels)
    
    info = {
        'cities': cities,
        'num_samples_per_city': num_samples_per_city,
        'data_shape': data_shape,
        'num_classes': len(unique_labels),
        'label_range': (int(unique_labels.min()), int(unique_labels.max()))
    }
    
    return info


def get_filtered_samples_efficient_zarr(
    zarr_path: str,
    cities: List[str],
    max_samples_per_class: Optional[int] = None,
    seed: int = 42
) -> List[Tuple[str, int]]:
    """
    Filtre les échantillons par classe de manière efficace avec Zarr
    
    Args:
        zarr_path: Chemin vers le fichier Zarr
        cities: Liste des villes à inclure
        max_samples_per_class: Nombre maximum d'échantillons par classe (None = tous)
        seed: Seed pour la reproductibilité
        
    Returns:
        Liste de tuples (city, local_idx)
    """
    zarr_root = zarr.open(zarr_path, mode='r')
    rng = np.random.RandomState(seed)
    all_samples = []
    
    for city in cities:
        # Zarr charge les labels très rapidement
        labels = np.array(zarr_root[city]['labels'][:])
        n_samples = len(labels)
        
        if max_samples_per_class is None:
            # Prendre tous les échantillons
            indices = np.arange(n_samples)
            rng.shuffle(indices)
            samples = [(city, int(idx)) for idx in indices]
        else:
            # Filtrer par classe
            class_indices = {}
            for idx, label in enumerate(labels):
                label_scalar = int(label) if np.ndim(label) == 0 else int(label.item())
                if label_scalar not in class_indices:
                    class_indices[label_scalar] = []
                class_indices[label_scalar].append(idx)
            
            # Sélectionner max_samples_per_class / nombre de villes
            samples_per_class_per_city = max(1, max_samples_per_class // len(cities))
            samples = []
            
            for label, indices in class_indices.items():
                indices_array = np.array(indices)
                rng.shuffle(indices_array)
                selected = indices_array[:samples_per_class_per_city]
                samples.extend([(city, int(idx)) for idx in selected])
        
        all_samples.extend(samples)
    
    # Mélanger tous les échantillons
    rng.shuffle(all_samples)
    
    return all_samples


class S1SLCDatasetZarr(Dataset):
    """
    Dataset PyTorch pour S1SLC utilisant Zarr
    
    Avantages par rapport à HDF5:
    - Support natif du multiprocessing (pas de problèmes de thread safety)
    - Accès aléatoires plus rapides
    - Meilleure compression
    - Compatible avec le cloud storage
    
    Retourne des numpy arrays (pas de tenseurs PyTorch).
    Chaque worker peut lire en parallèle sans conflit.
    
    Args:
        zarr_path: Chemin vers le fichier Zarr
        samples: Liste de tuples (city, local_idx)
        transform: Transformations à appliquer (sur numpy arrays)
    """
    CLASS_NAME = {
        1: "Agricultural fields (AG)",
        2: "Forest and Woodlands (FR)",
        3: "High Density Urban Areas (HD)",
        4: "High Rise Buildings (HR)",
        5: "Low Density Urban Areas (LD)",
        6: "Industrial Regions (IR)",
        7: "Water Regions (WR)"
    }
    
    def __init__(
        self,
        zarr_path: str,
        samples: List[Tuple[str, int]],
        transform=None
    ):
        super().__init__()
        self.zarr_path = zarr_path
        self.samples = samples
        self.transform = transform
        # Zarr peut être ouvert une seule fois et partagé entre workers !
        self.zarr_root = zarr.open(zarr_path, mode='r')
        self.class_to_idx = {v: k-1 for k, v in self.CLASS_NAME.items()}
        self.target_size = (100,100)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, str]:
        city, local_idx = self.samples[idx]
        
        # Zarr supporte le vrai parallélisme sans problème
        data = np.array(self.zarr_root[city]['data'][local_idx], dtype=np.float32)
        original_label = int(self.zarr_root[city]['labels'][local_idx])
        label = original_label - 1  # Convert 1-7 to 0-6 for PyTorch
        label_name = self.CLASS_NAME.get(original_label, "Unknown")
        
        # Transposer de (H, W, C) à (C, H, W) si nécessaire
        if data.ndim == 3 and data.shape[-1] == 4:
            data = data.transpose(2, 0, 1)
        
        if self.transform:
            data = self.transform(data)
        metadata = {'city': city,
                    'label_name': label_name
                    }
        return data, label, metadata


def get_city_split_indices(
    n_total: int,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sépare les indices en train/val/test
    
    Args:
        n_total: Nombre total d'échantillons
        train_ratio: Ratio pour train+val (le reste sera test)
        val_ratio: Ratio de validation dans le train (par rapport au train+val)
        seed: Seed pour reproductibilité
        
    Returns:
        train_indices, val_indices, test_indices
    """
    rng = np.random.RandomState(seed)
    indices = np.arange(n_total)
    rng.shuffle(indices)
    
    # Split train+val / test
    n_test = int(n_total * (1 - train_ratio))
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return train_indices, test_indices


def create_s1slc_dataloader(
    zarr_path: str,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    seed: int = 42,
    num_workers: int = 8,  
    shuffle_train: bool = True,
    transform=None,
    max_samples_per_class: Optional[int] = None,
    verbose: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crée les dataloaders en mode standard avec Zarr (beaucoup plus rapide que HDF5)
    
    Split: 80% train+val, 20% test
           Dans train+val: 80% train, 20% val
    
    Args:
        zarr_path: Chemin vers le fichier Zarr
        batch_size: Taille des batchs
        train_ratio: Ratio train+val (le reste = test)
        val_ratio: Ratio validation dans train+val
        seed: Seed pour reproductibilité
        num_workers: Nombre de workers (peut être élevé avec Zarr !)
        shuffle_train: Mélanger les données d'entraînement
        transform: Transformations à appliquer
        max_samples_per_class: Limitation du nombre d'échantillons par classe
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Obtenir les villes disponibles
    zarr_root = zarr.open(zarr_path, mode='r')
    all_cities = list(zarr_root.keys())
    
    
    # Filtrer les échantillons si nécessaire
    all_samples = get_filtered_samples_efficient_zarr(
        zarr_path, 
        all_cities, 
        max_samples_per_class, 
        seed
    )

    if verbose:
        print(f"Found cities: {all_cities}")
        print(f"Total samples after filtering: {len(all_samples)}")
        #print all labels:
        all_labels = [zarr_root[city]['labels'][idx] for city, idx in all_samples]
        unique_labels = np.unique(all_labels)
        

    # Split train / test
    train_indices, test_indices = get_city_split_indices(
        len(all_samples), train_ratio, seed
    )
    
    train_samples = [all_samples[i] for i in train_indices]
    test_samples = [all_samples[i] for i in test_indices]
    
    
    
    # Créer les datasets Zarr
    train_dataset = S1SLCDatasetZarr(zarr_path, train_samples, transform)
    test_dataset = S1SLCDatasetZarr(zarr_path, test_samples, transform)
    if verbose:
        print(f"Unique labels: {[train_dataset.CLASS_NAME.get(int(lbl), 'Unknown') for lbl in unique_labels]}")
    
    # Créer les dataloaders -
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None
    )
    
   
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    return train_loader, test_loader


def create_s1slc_DA_dataloader(
    zarr_path: str,
    train_cities: Optional[List[str]] = None,
    test_city: Optional[str] = None,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 8,
    shuffle_train: bool = True,
    transform=None,
    max_samples_per_class: Optional[int] = None,
    verbose: bool = False   
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crée les dataloaders en mode domain adaptation avec Zarr
    
    Train sur plusieurs villes, test sur une ville différente
    
    Args:
        zarr_path: Chemin vers le fichier Zarr
        train_cities: Villes pour l'entraînement (None = toutes sauf test_city)
        test_city: Ville pour le test (None = dernière ville)
        batch_size: Taille des batchs
        val_ratio: Ratio de validation (extrait du train)
        seed: Seed pour reproductibilité
        num_workers: Nombre de workers
        shuffle_train: Mélanger les données d'entraînement
        transform: Transformations à appliquer
        max_samples_per_class: Limitation du nombre d'échantillons par classe
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Obtenir les villes disponibles
    zarr_root = zarr.open(zarr_path, mode='r')
    all_cities = list(zarr_root.keys())
    
    # Déterminer les villes de train et test
    if test_city is None:
        test_city = all_cities[-1]
    
    if train_cities is None:
        train_cities = [c for c in all_cities if c != test_city]
    
    # Obtenir les échantillons filtrés pour train et test
    train_samples = get_filtered_samples_efficient_zarr(
        zarr_path, train_cities, max_samples_per_class, seed
    )
    test_samples = get_filtered_samples_efficient_zarr(
        zarr_path, [test_city], max_samples_per_class, seed
    )
    if verbose:
        print(f"Train cities: {train_cities}")
        print(f"Test city: {test_city}")
        print(f"Total train samples: {len(train_samples)}")
        print(f"Total test samples: {len(test_samples)}")
        
    
    # Créer les datasets
    train_dataset = S1SLCDatasetZarr(zarr_path, train_samples, transform)
    test_dataset = S1SLCDatasetZarr(zarr_path, test_samples, transform)
    
    # Créer les dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Exemple d'utilisation
    data_root = "/home/mgallet/Téléchargements/S1SLC_CVDL/complex_HHHV.zarr"
    print("\n" + "=" * 60)
    print("Mode CLassic ")
    print("=" * 60)
    train_loader, test_loader = create_s1slc_dataloader(
        data_root,
        batch_size=32,
        shuffle_train=True,
        max_samples_per_class=100,
        num_workers=8,
        verbose=True
        
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")


    for batch_idx, (images, targets, metadata) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Images dtype: {images.dtype}")
        print(f"  Targets shape: {targets.shape}")
        print(f"  Targets dtype: {targets.dtype}")
        print(f"  Cities (first 5): {metadata['city'][:5]}")
        print(f"  Unique cities in batch: {set(metadata['city'])}")
        print(f"  Labels (first 5): {metadata['label_name'][:5]}")
        print(f"  Unique labels in batch: {set(metadata['label_name'])}")   
        print(f"  Min/Max pixel values: {images.min():.3f} / {images.max():.3f}")
        
        if batch_idx >= 2:
            break

    # Mode domain adaptation
    print("\n" + "=" * 60)
    print("Mode Domain Adaptation (avec Zarr)")
    print("=" * 60)
    
    info = get_dataset_info_zarr(data_root)
    if len(info['cities']) >= 2:
        train_cities = info['cities'][:-1]
        test_city = info['cities'][-1]
        
        train_loader_da, test_loader_da = create_s1slc_DA_dataloader(
            data_root,
            train_cities=train_cities,
            test_city=test_city,
            batch_size=32,
            shuffle_train=True,
            max_samples_per_class=50,
            num_workers=8
        )

        for batch_idx, (images, targets, metadata) in enumerate(train_loader_da):
            print(f"DA Batch {batch_idx + 1}:")
            print(f"  Images shape: {images.shape}")
            print(f"  Targets shape: {targets.shape}")
            print(f"  Cities (first 5): {metadata['city'][:5]}")
            print(f"  Unique cities in batch: {set(metadata['city'])}")
            print(f"  Labels (first 5): {metadata['label_name'][:5]}")
            print(f"  Unique labels in batch: {set(metadata['label_name'])}")   
            print(f"  Min/Max pixel values: {images.min():.3f} / {images.max():.3f}")
            
            if batch_idx >= 2:
                break
    else:
        print("Not enough cities for domain adaptation example (need at least 2)")
