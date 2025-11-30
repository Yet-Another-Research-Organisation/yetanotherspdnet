#!/usr/bin/env python3
"""
Dataloader pour le dataset Rices_90 avec matrices de covariance précalculées.
Supporte la classification avec extraction automatique des noms de classes depuis les noms de fichiers.

Les noms de classes correspondent au nom du fichier split("-")[0].
Exemple: "91RH-02_000.pt" -> classe "91RH"

90 classes de variétés de riz vietnamiennes avec matrices de covariance 256x256.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import Optional, List, Tuple, Union
import numpy as np
import warnings


class Rices90Dataset(Dataset):
    """
    Dataset class pour Rices_90 avec matrices de covariance précalculées.

    Args:
        data_dir: Dossier contenant les fichiers .pt des matrices de covariance
        classes: Liste optionnelle des classes à utiliser (toutes si None)
        max_samples_per_class: Nombre maximum d'échantillons par classe
        max_classes: Nombre maximum de classes à utiliser (None = toutes)
        transform: Transformations optionnelles à appliquer
        label_noise_ratio: Ratio de labels à corrompre aléatoirement (0.0 à 1.0)
        seed: Graine aléatoire pour la reproductibilité
    """
    
    # Les 90 classes de variétés de riz vietnamiennes
    CLASS_NAMES = [
        '91RH', '9d', 'A128', 'AH1000', 'BacThomSo7', 'BC15', 'BQ10', 'BT6', 'CH12',
        'CL61', 'CNC12', 'CS6', 'CT286', 'CTX30', 'DA1', 'DaiThom8', 'DMV58', 'DT52',
        'DT66', 'DTH155', 'DTL2', 'DV108', 'GiaLoc301', 'GS55R', 'H229', 'HaNa39',
        'HaPhat28', 'HoangLong', 'HongQuang15', 'HS1', 'HT18', 'HungDan1', 'KB16',
        'KB27', 'KB6', 'KhangDan18', 'KimCuong111', 'KL25', 'KN5', 'LDA8', 'LocTroi183',
        'LTH35', 'MT15', 'MyHuong88', 'N54', 'N97', 'N98', 'NBK', 'NBP', 'NBT1',
        'NC2', 'NC7', 'ND9', 'NDC1', 'NepCoTien', 'NepDacSanLienHoa', 'NepHongNgoc',
        'NepKB19', 'NepPhatQuy', 'NepThomBacHai', 'NepThomHungYen', 'NH92', 'NM14',
        'NN4B', 'NPT1', 'NPT3', 'NT16', 'NTP', 'NV1', 'PC10', 'PD211', 'R068',
        'R998KBL', 'SHPT1', 'SVN1', 'TB13', 'TB14', 'TC10', 'TC11', 'ThuanViet2',
        'TQ14', 'TQ36', 'TruongXuan1', 'TruongXuanHQ', 'VietHuong8', 'VietThom8',
        'VinhPhuc1', 'VS1', 'VS5', 'VS6'
    ]

    def __init__(self,
                 data_dir: str,
                 classes: Optional[List[str]] = None,
                 max_samples_per_class: Optional[int] = None,
                 max_classes: Optional[int] = None,
                 transform: Optional[callable] = None,
                 label_noise_ratio: float = 0.0,
                 seed: int = 42):

        self.data_dir = Path(data_dir)
        self.transform = transform
        self.max_samples_per_class = max_samples_per_class
        self.max_classes = max_classes
        self.label_noise_ratio = label_noise_ratio
        self.seed = seed
        self.target_size = (256, 256) 
        
        # Définir le générateur aléatoire pour la reproductibilité
        self.rng = np.random.RandomState(seed)
        self.rng_max_classes = np.random.RandomState(42)

        # Vérifier que le dossier existe
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dossier de données introuvable: {self.data_dir}")

        # Lister tous les fichiers .pt (exclure chessboard qui n'est pas une classe)
        pt_files = [f for f in self.data_dir.glob("*.pt") if not f.stem.startswith('chessboard')]
        if len(pt_files) == 0:
            raise ValueError(f"Aucun fichier .pt trouvé dans {self.data_dir}")

        print(f"Trouvé {len(pt_files)} fichiers de covariance")

        # Extraire les classes depuis les noms de fichiers
        # Format: "CLASS-VARIANT_INDEX.pt" (ex: "91RH-02_000.pt" -> classe "91RH")
        file_info = []
        for pt_file in pt_files:
            filename = pt_file.stem  # Nom sans extension
            try:
                class_name = filename.split("-")[0]  # Première partie avant le tiret
                file_info.append({
                    'path': pt_file,
                    'class_name': class_name,
                    'filename': filename
                })
            except IndexError:
                warnings.warn(f"Nom de fichier invalide (pas de tiret): {filename}")
                continue

        # Obtenir la liste unique des classes
        all_classes = sorted(list(set(info['class_name'] for info in file_info)))
        print(f"Classes trouvées: {len(all_classes)} classes")
        
        # Limiter le nombre de classes si demandé
        if max_classes is not None and max_classes < len(all_classes):
            # Utiliser le RNG avec seed pour sélectionner les classes
            selected_classes = self.rng_max_classes.choice(all_classes, size=max_classes, replace=False).tolist()
            all_classes = sorted(selected_classes)
            file_info = [info for info in file_info if info['class_name'] in all_classes]
            print(f"Utilisation de {max_classes} classes sur {len(all_classes)}: {all_classes}")

        # Filtrer les classes si spécifié
        if classes is not None:
            classes = [c for c in classes if c in all_classes]
            if len(classes) == 0:
                raise ValueError(f"Aucune des classes spécifiées trouvée dans {all_classes}")
            file_info = [info for info in file_info if info['class_name'] in classes]
            print(f"Utilisation de {len(classes)} classes personnalisées: {classes}")
        else:
            classes = all_classes

        # Créer le mapping classe -> index
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
        self.classes = classes

        # Limiter le nombre d'échantillons par classe si demandé
        if max_samples_per_class is not None:
            limited_file_info = []
            for class_name in classes:
                class_files = [info for info in file_info if info['class_name'] == class_name]
                if len(class_files) > max_samples_per_class:
                    # Sélectionner aléatoirement max_samples_per_class fichiers avec seed
                    selected_indices = self.rng.choice(
                        len(class_files),
                        size=max_samples_per_class,
                        replace=False
                    )
                    selected_files = [class_files[i] for i in selected_indices]
                    limited_file_info.extend(selected_files)
                    print(f"Classe {class_name}: {len(class_files)} -> {max_samples_per_class} échantillons")
                else:
                    limited_file_info.extend(class_files)
                    print(f"Classe {class_name}: {len(class_files)} échantillons (non limité)")
            file_info = limited_file_info

        self.file_info = file_info
        self.samples = file_info  # Pour compatibilité avec certaines fonctions
        
        # Appliquer le bruit sur les labels si demandé
        if label_noise_ratio > 0.0:
            self._corrupt_labels()

        print(f"\nDataset final:")
        print(f"  - {len(self.file_info)} échantillons")
        print(f"  - {len(self.classes)} classes")
        print(f"  - Dimensions des matrices: 256x256")

        # Afficher la distribution des classes
        self._print_class_distribution()
    
    def _corrupt_labels(self):
        """
        Corrompre aléatoirement un pourcentage de labels en les assignant à des classes différentes.
        Similaire à HyperLeaf.
        """
        if self.label_noise_ratio <= 0.0 or self.label_noise_ratio >= 1.0:
            if self.label_noise_ratio < 0.0 or self.label_noise_ratio > 1.0:
                warnings.warn(f"label_noise_ratio doit être entre 0.0 et 1.0, reçu {self.label_noise_ratio}")
            return
        
        num_classes = len(self.classes)
        num_samples = len(self.file_info)
        num_to_corrupt = int(num_samples * self.label_noise_ratio)
        
        # Sélectionner aléatoirement les indices à corrompre
        corrupt_indices = self.rng.choice(num_samples, size=num_to_corrupt, replace=False)
        
        # Pour chaque index sélectionné, assigner une classe différente aléatoire
        corrupted_count = 0
        for idx in corrupt_indices:
            original_class = self.file_info[idx]['class_name']
            # Choisir une classe différente aléatoirement
            other_classes = [c for c in self.classes if c != original_class]
            if other_classes:
                new_class = self.rng.choice(other_classes)
                self.file_info[idx]['class_name'] = new_class
                corrupted_count += 1
        
        print(f"\nBruit sur les labels appliqué:")
        print(f"  - {corrupted_count} labels corrompus ({self.label_noise_ratio*100:.1f}% de {num_samples} échantillons)")

    def _print_class_distribution(self):
        """Affiche la distribution des classes."""
        class_counts = {}
        for info in self.file_info:
            class_name = info['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        print("\nDistribution des classes:")
        for class_name in sorted(class_counts.keys()):
            count = class_counts[class_name]
            percentage = count / len(self.file_info) * 100
            print(f"  {class_name}: {count} échantillons ({percentage:.1f}%)")

    def __len__(self) -> int:
        return len(self.file_info)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retourne un échantillon et son label.

        Args:
            idx: Index de l'échantillon

        Returns:
            Tuple (covariance_matrix, class_index)
        """
        info = self.file_info[idx]

        # Charger la matrice de covariance
        try:
            cov_matrix = torch.load(info['path'])
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement de {info['path']}: {e}")

        # Vérifier les dimensions (doivent être 256x256)
        if cov_matrix.shape != (256, 256):
            raise ValueError(f"Matrice de covariance invalide: {cov_matrix.shape} != (256, 256)")

        # Appliquer les transformations si spécifiées
        if self.transform is not None:
            cov_matrix = self.transform(cov_matrix)

        # Obtenir l'index de la classe
        class_idx = self.class_to_idx[info['class_name']]

        return cov_matrix, class_idx

    def get_class_weights(self) -> torch.Tensor:
        """
        Calcule les poids des classes pour équilibrer l'entraînement.

        Returns:
            Tensor des poids des classes
        """
        class_counts = np.zeros(len(self.classes))
        for info in self.file_info:
            class_idx = self.class_to_idx[info['class_name']]
            class_counts[class_idx] += 1

        # Calculer les poids (inversement proportionnel au nombre d'échantillons)
        total_samples = len(self.file_info)
        weights = total_samples / (len(self.classes) * class_counts)

        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_info(self, idx: int) -> dict:
        """
        Retourne les informations détaillées sur un échantillon.

        Args:
            idx: Index de l'échantillon

        Returns:
            Dictionnaire avec les informations
        """
        info = self.file_info[idx]
        return {
            'filename': info['filename'],
            'class_name': info['class_name'],
            'class_idx': self.class_to_idx[info['class_name']],
            'path': str(info['path'])
        }


def create_rices90_dataloaders(data_dir: str,
                              classes: Optional[List[str]] = None,
                              max_samples_per_class: Optional[int] = None,
                              max_classes: Optional[int] = None,
                              batch_size: int = 32,
                              num_workers: int = 4,
                              val_ratio: float = 0.2,
                              test_ratio: float = 0.1,
                              seed: int = 42,
                              label_noise_ratio: float = 0.0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crée les dataloaders train/val/test pour Rices_90.

    Args:
        data_dir: Dossier contenant les fichiers .pt
        classes: Liste optionnelle des classes à utiliser
        max_samples_per_class: Nombre maximum d'échantillons par classe
        max_classes: Nombre maximum de classes à utiliser (None = toutes)
        batch_size: Taille des batches
        num_workers: Nombre de workers pour le chargement
        val_ratio: Proportion des données pour la validation
        test_ratio: Proportion des données pour le test
        seed: Graine aléatoire pour la reproductibilité
        label_noise_ratio: Ratio de labels à corrompre (0.0 à 1.0)

    Returns:
        Tuple (train_loader, val_loader, test_loader)
    """

    # Créer le dataset complet
    full_dataset = Rices90Dataset(
        data_dir=data_dir,
        classes=classes,
        max_samples_per_class=max_samples_per_class,
        max_classes=max_classes,
        label_noise_ratio=label_noise_ratio,
        seed=seed
    )

    # Diviser en train/val/test avec random_split de torch
    dataset_size = len(full_dataset)
    train_size = int(dataset_size * (1 - val_ratio - test_ratio))
    val_size = int(dataset_size * val_ratio)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    print(f"\nDivision des données:")
    print(f"  Train: {len(train_dataset)} échantillons ({train_size/dataset_size*100:.1f}%)")
    print(f"  Validation: {len(val_dataset)} échantillons ({val_size/dataset_size*100:.1f}%)")
    print(f"  Test: {len(test_dataset)} échantillons ({test_size/dataset_size*100:.1f}%)")

    # Créer les dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    return train_loader, val_loader, test_loader


def get_num_classes(data_dir: str, classes: Optional[List[str]] = None, max_classes: Optional[int] = None) -> int:
    """
    Retourne le nombre de classes dans le dataset.

    Args:
        data_dir: Dossier contenant les fichiers .pt
        classes: Liste optionnelle des classes à utiliser
        max_classes: Nombre maximum de classes (None = toutes)

    Returns:
        Nombre de classes
    """
    dataset = Rices90Dataset(
        data_dir, 
        classes=classes, 
        max_samples_per_class=1,
        max_classes=max_classes
    )
    return len(dataset.classes)


def main():
    """
    Fonction principale pour tester le dataloader.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Test du dataloader Rices_90")
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Dossier contenant les fichiers .pt')
    parser.add_argument('--classes', type=str, nargs='*',
                       help='Classes à utiliser (optionnel)')
    parser.add_argument('--max_samples_per_class', type=int, default=None,
                       help='Nombre maximum d\'échantillons par classe')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Taille des batches')
    parser.add_argument('--max_classes', type=int, default=None,
                       help='Nombre maximum de classes à utiliser')
    parser.add_argument('--label_noise_ratio', type=float, default=0.0,
                       help='Ratio de bruit sur les labels (0.0 à 1.0)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Graine aléatoire')

    args = parser.parse_args()

    print("=== Test du Dataloader Rices_90 ===")

    # Créer les dataloaders
    train_loader, val_loader, test_loader = create_rices90_dataloaders(
        data_dir=args.data_dir,
        classes=args.classes,
        max_samples_per_class=args.max_samples_per_class,
        max_classes=args.max_classes,
        batch_size=args.batch_size,
        val_ratio=0.2,
        test_ratio=0.1,
        label_noise_ratio=args.label_noise_ratio,
        seed=args.seed
    )

    print(f"\nNombre de classes: {get_num_classes(args.data_dir, args.classes, args.max_classes)}")

    # Tester quelques batches
    print("\n=== Test des batches ===")

    for split_name, loader in [("Train", train_loader), ("Val", val_loader), ("Test", test_loader)]:
        print(f"\n{split_name} loader:")
        for batch_idx, (cov_matrices, labels) in enumerate(loader):
            print(f"  Batch {batch_idx}: matrices shape {cov_matrices.shape}, labels shape {labels.shape}")
            print(f"    Labels: {labels.tolist()}")
            if batch_idx >= 2:  # Montrer seulement les 3 premiers batches
                break

    print("\n=== Test terminé avec succès ===")


if __name__ == "__main__":
    main()