#!/usr/bin/env python3
"""
Script complet pour extraire les imagettes et précalculer les matrices de covariance normalisées.
Combine l'extraction d'imagettes depuis les fichiers ENVI et le calcul des covariances avec normalisation.

Usage:
    python extract_and_precompute_covariances.py --data_dir ~/Documents/Dataset/BATCHNORM_dataset/Rices_90
    python extract_and_precompute_covariances.py --data_dir ~/Documents/Dataset/BATCHNORM_dataset/Rices_90 --num_workers 8
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import tifffile
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from spectral.io import envi
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, footprint_rectangle


def extract_imagettes_from_envi(hdr_path, output_dir, min_area=50, margin_h=0.35, margin_w=0.15):
    """
    Extrait les imagettes depuis un fichier ENVI (.hdr).

    Args:
        hdr_path: Chemin vers le fichier .hdr
        output_dir: Dossier de sortie pour les imagettes
        min_area: Aire minimale des régions (en pixels)
        margin_h: Marge verticale pour l'élargissement des bbox (35%)
        margin_w: Marge horizontale pour l'élargissement des bbox (15%)

    Returns:
        Liste des chemins vers les imagettes extraites
    """
    try:
        # Charger le fichier ENVI
        data = envi.open(str(hdr_path))

        # Lire toutes les bandes
        img_full = data.asarray()  # Shape: (height, width, bands)

        # Prendre la dernière bande pour la détection
        last_band = img_full[:, :, -1]

        # Normaliser la dernière bande pour la segmentation
        O = last_band.astype(np.float32) / 65536.0
        O = (O - O.min()) / (O.max() - O.min())

        # Appliquer le seuillage
        thresh = threshold_otsu(O)
        bw = closing(O > thresh, footprint_rectangle((3, 3)))

        # Supprimer les artefacts connectés aux bords
        cleared = clear_border(bw)

        # Étiqueter les régions
        label_image = label(cleared)

        # Récupérer les propriétés des régions
        regions = regionprops(label_image)

        # Filtrer et élargir les bounding boxes
        valid_bboxes = []
        for region in regions:
            if region.area >= min_area:
                minr, minc, maxr, maxc = region.bbox

                # Calculer les dimensions
                height = maxr - minr
                width = maxc - minc

                # Élargir la bbox (35% vertical, 15% horizontal)
                new_minr = max(0, minr - int(margin_h * height))
                new_maxr = min(img_full.shape[0], maxr + int(margin_h * height))
                new_minc = max(0, minc - int(margin_w * width))
                new_maxc = min(img_full.shape[1], maxc + int(margin_w * width))

                valid_bboxes.append((new_minr, new_minc, new_maxr, new_maxc))

        # Sauvegarder les imagettes
        base_name = hdr_path.stem  # Nom du fichier sans extension
        extracted_paths = []

        for i, (minr, minc, maxr, maxc) in enumerate(valid_bboxes):
            # Extraire l'imagette sur tous les canaux (non normalisés)
            imagette = img_full[minr:maxr, minc:maxc, :]

            # Nom du fichier de sortie
            output_filename = f"{base_name}_{i:03d}.tif"
            output_path = output_dir / output_filename

            # Sauvegarder en TIFF
            tifffile.imwrite(output_path, imagette.astype(np.uint16))

            extracted_paths.append(output_path)

        return extracted_paths

    except Exception as e:
        print(f"Erreur lors du traitement de {hdr_path}: {e}")
        return []


def load_imagette(image_path: str) -> np.ndarray:
    """
    Charge une imagette TIFF.

    Args:
        image_path: Chemin vers le fichier TIFF

    Returns:
        Array numpy de shape (H, W, C)
    """
    return tifffile.imread(image_path).astype(np.float32)


def compute_covariance_matrix(image: np.ndarray, normalization_factor: float = 91.0) -> torch.Tensor:
    """
    Calcule la matrice de covariance d'une image hyperspectrale avec normalisation.

    Args:
        image: Array de shape (H, W, C) où C est le nombre de bandes spectrales
        normalization_factor: Facteur de normalisation (diviser par cette valeur)

    Returns:
        Matrice de covariance normalisée de shape (C, C)
    """
    # Normaliser l'image par le facteur
    image_normalized = image / normalization_factor

    # Reshape en (C, H*W) pour calculer la covariance
    H, W, C = image_normalized.shape
    X = image_normalized.reshape(H * W, C).T  # Shape: (C, H*W)

    # Centrer les données (recommandé pour la covariance)
    X_centered = X - np.mean(X, axis=1, keepdims=True)

    # Calculer la covariance
    cov_matrix = 1/(H * W - 1) * (X_centered @ X_centered.T)  # Shape: (C, C)

    # Convertir en tensor PyTorch
    return torch.tensor(cov_matrix, dtype=torch.float32)


def process_single_imagette(args_tuple):
    """
    Traite une seule imagette: charge, calcule la covariance, et sauvegarde.
    Cette fonction est utilisée pour le traitement parallèle.

    Args:
        args_tuple: Tuple de (imagette_path, output_path, normalization_factor)

    Returns:
        Tuple de (imagette_id, success, error_message)
    """
    imagette_path, output_path, normalization_factor = args_tuple
    imagette_id = Path(imagette_path).stem

    try:
        # Charger l'imagette
        image = load_imagette(imagette_path)

        # Calculer la matrice de covariance avec normalisation
        cov_matrix = compute_covariance_matrix(image, normalization_factor)

        # Sauvegarder comme tensor PyTorch
        torch.save(cov_matrix, output_path)

        return (imagette_id, True, None)

    except Exception as e:
        return (imagette_id, False, str(e))


def extract_all_imagettes(data_dir: str, imagettes_dir: str, num_workers: int = None) -> list:
    """
    Extrait toutes les imagettes depuis les fichiers ENVI du dataset.

    Args:
        data_dir: Dossier racine contenant les fichiers ENVI
        imagettes_dir: Dossier de sortie pour les imagettes
        num_workers: Nombre de workers pour le parallélisme

    Returns:
        Liste de tous les chemins d'imagettes extraites
    """
    data_dir = Path(data_dir)
    imagettes_dir = Path(imagettes_dir)
    imagettes_dir.mkdir(exist_ok=True)

    print(f"Recherche de fichiers .hdr dans: {data_dir}")

    # Lister tous les fichiers .hdr
    hdr_files = list(data_dir.rglob("*.hdr"))
    print(f"Nombre total de fichiers .hdr trouvés: {len(hdr_files)}")

    if len(hdr_files) == 0:
        raise ValueError(f"Aucun fichier .hdr trouvé dans {data_dir}")

    # Déterminer le nombre de workers
    if num_workers is None:
        num_workers = min(cpu_count(), len(hdr_files))

    print(f"Extraction des imagettes avec {num_workers} workers...")

    # Traiter les fichiers ENVI en parallèle
    all_imagette_paths = []

    if num_workers == 1:
        # Traitement séquentiel
        for hdr_file in tqdm(hdr_files, desc="Extraction des imagettes"):
            imagette_paths = extract_imagettes_from_envi(hdr_file, imagettes_dir)
            all_imagette_paths.extend(imagette_paths)
    else:
        # Traitement parallèle
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(extract_imagettes_from_envi, hdr_file, imagettes_dir): hdr_file
                      for hdr_file in hdr_files}

            with tqdm(total=len(hdr_files), desc="Extraction des imagettes") as pbar:
                for future in as_completed(futures):
                    imagette_paths = future.result()
                    all_imagette_paths.extend(imagette_paths)
                    pbar.update(1)

    print(f"Total d'imagettes extraites: {len(all_imagette_paths)}")
    return all_imagette_paths


def precompute_covariances(imagette_paths: list, cov_dir: str,
                          normalization_factor: float = 91.0,
                          num_workers: int = None,
                          force_recompute: bool = False) -> dict:
    """
    Précalcule toutes les matrices de covariance pour les imagettes.

    Args:
        imagette_paths: Liste des chemins vers les imagettes
        cov_dir: Dossier de sortie pour les covariances
        normalization_factor: Facteur de normalisation
        num_workers: Nombre de workers pour le parallélisme
        force_recompute: Si True, recalcule même si la covariance existe déjà

    Returns:
        Dictionnaire avec les statistiques
    """
    cov_dir = Path(cov_dir)
    cov_dir.mkdir(exist_ok=True)

    print(f"\nPrécalcul des covariances avec facteur de normalisation: {normalization_factor}")

    # Déterminer le nombre de workers
    if num_workers is None:
        num_workers = min(cpu_count(), len(imagette_paths))

    print(f"Utilisation de {num_workers} workers pour le calcul des covariances")

    # Préparer les tâches
    tasks = []
    skipped = 0

    for imagette_path in imagette_paths:
        imagette_path = Path(imagette_path)
        imagette_id = imagette_path.stem
        output_path = cov_dir / f"{imagette_id}.pt"

        # Sauter si existe déjà et pas de force_recompute
        if output_path.exists() and not force_recompute:
            skipped += 1
            continue

        tasks.append((str(imagette_path), str(output_path), normalization_factor))

    if skipped > 0:
        print(f"Saut de {skipped} covariances déjà calculées (utilisez --force pour recalculer)")

    if len(tasks) == 0:
        print("Toutes les covariances sont déjà calculées!")
        return {"total": len(imagette_paths), "skipped": skipped, "processed": 0, "failed": 0}

    print(f"Traitement de {len(tasks)} imagettes...")

    # Traiter en parallèle
    successful = 0
    failed = []

    if num_workers == 1:
        # Traitement séquentiel
        for task in tqdm(tasks, desc="Calcul des covariances"):
            imagette_id, success, error = process_single_imagette(task)
            if success:
                successful += 1
            else:
                failed.append((imagette_id, error))
    else:
        # Traitement parallèle
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_imagette, task): task for task in tasks}

            with tqdm(total=len(tasks), desc="Calcul des covariances") as pbar:
                for future in as_completed(futures):
                    imagette_id, success, error = future.result()
                    if success:
                        successful += 1
                    else:
                        failed.append((imagette_id, error))
                    pbar.update(1)

    # Statistiques
    stats = {
        "total": len(imagette_paths),
        "skipped": skipped,
        "processed": successful,
        "failed": len(failed)
    }

    # Afficher le résumé
    print("\n" + "="*60)
    print("PRÉCALCUL DES COVARIANCES TERMINÉ")
    print("="*60)
    print(f"Total d'imagettes: {stats['total']}")
    print(f"Déjà calculées: {stats['skipped']}")
    print(f"Traitées avec succès: {stats['processed']}")
    print(f"Échouées: {stats['failed']}")
    print(f"Covariances sauvegardées dans: {cov_dir}")

    if failed:
        print("\nImagettes échouées:")
        for imagette_id, error in failed[:10]:  # Montrer les 10 premières erreurs
            print(f"  - {imagette_id}: {error}")
        if len(failed) > 10:
            print(f"  ... et {len(failed) - 10} autres")

    # Montrer un exemple
    if successful > 0:
        example_file = cov_dir / f"{Path(tasks[0][0]).stem}.pt"
        if example_file.exists():
            example_cov = torch.load(example_file)
            print(f"\nExemple de matrice de covariance:")
            print(f"  Forme: {example_cov.shape}")
            print(f"  Type: {example_cov.dtype}")
            print(f"  Taille du fichier: {os.path.getsize(example_file) / 1024:.2f} KB")
            print(f"  Trace: {torch.trace(example_cov).item():.6f}")

    print("="*60)

    return stats


def verify_covariances(cov_dir: str, num_samples: int = 5):
    """
    Vérifie que les covariances précalculées sont valides.

    Args:
        cov_dir: Dossier contenant les covariances
        num_samples: Nombre d'échantillons à vérifier
    """
    cov_dir = Path(cov_dir)

    if not cov_dir.exists():
        print(f"Dossier de covariances introuvable: {cov_dir}")
        return

    cov_files = sorted([f for f in cov_dir.iterdir() if f.suffix == '.pt'])

    if len(cov_files) == 0:
        print(f"Aucun fichier de covariance trouvé dans {cov_dir}")
        return

    print(f"\nVérification de {min(num_samples, len(cov_files))} matrices de covariance...")
    print("-" * 60)

    for i, cov_file in enumerate(cov_files[:num_samples]):
        try:
            cov = torch.load(cov_file)

            # Vérifier les propriétés
            is_square = cov.shape[0] == cov.shape[1]
            is_symmetric = torch.allclose(cov, cov.T, rtol=1e-5, atol=1e-7)

            # Vérifier semi-définie positive (valeurs propres non-négatives)
            eigenvalues = torch.linalg.eigvalsh(cov)
            min_eig = eigenvalues.min().item()
            is_psd = min_eig >= -1e-6  # Tolérer petites erreurs numériques

            print(f"{i+1}. {cov_file.name}")
            print(f"   Forme: {cov.shape}")
            print(f"   Type: {cov.dtype}")
            print(f"   Carrée: {is_square}")
            print(f"   Symétrique: {is_symmetric}")
            print(f"   Semi-définie positive: {is_psd} (valeur propre min: {min_eig:.2e})")
            print(f"   Trace: {torch.trace(cov).item():.6f}")
            print(f"   Nombre de condition: {eigenvalues.max().item() / (eigenvalues.min().item() + 1e-10):.2e}")

            if not (is_square and is_symmetric and is_psd):
                print("   ⚠️  ATTENTION: Propriétés de matrice invalides!")
            else:
                print("   ✓ Matrice de covariance valide")
            print()

        except Exception as e:
            print(f"{i+1}. {cov_file.name}")
            print(f"   ✗ ERREUR: {e}")
            print()

    print("-" * 60)
    print(f"Total de fichiers de covariance: {len(cov_files)}")


def main():
    parser = argparse.ArgumentParser(
        description="Extraction d'imagettes et précalcul des matrices de covariance normalisées",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Dossier racine contenant les fichiers ENVI (.hdr)'
    )

    parser.add_argument(
        '--imagettes_dir',
        type=str,
        default=None,
        help='Dossier de sortie pour les imagettes (défaut: data_dir + "_imagettes")'
    )

    parser.add_argument(
        '--cov_dir',
        type=str,
        default=None,
        help='Dossier de sortie pour les covariances (défaut: data_dir + "_cov")'
    )

    parser.add_argument(
        '--normalization_factor',
        type=float,
        default=91.0,
        help='Facteur de normalisation pour diviser les images'
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Nombre de workers parallèles (défaut: auto-détection CPU)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Forcer le recalcul même si les fichiers existent déjà'
    )

    parser.add_argument(
        '--skip_extraction',
        action='store_true',
        help='Sauter l\'extraction d\'imagettes et utiliser celles existantes'
    )

    parser.add_argument(
        '--skip_covariances',
        action='store_true',
        help='Sauter le calcul des covariances'
    )

    parser.add_argument(
        '--verify',
        action='store_true',
        help='Vérifier les covariances après calcul'
    )

    parser.add_argument(
        '--verify_only',
        action='store_true',
        help='Seulement vérifier les covariances existantes'
    )

    parser.add_argument(
        '--verify_samples',
        type=int,
        default=5,
        help='Nombre d\'échantillons à vérifier'
    )

    args = parser.parse_args()

    print("="*60)
    print("EXTRACTION D'IMAGETTES ET PRÉCALCUL DES COVARIANCES")
    print("="*60)
    print(f"Dataset: {args.data_dir}")
    print(f"Facteur de normalisation: {args.normalization_factor}")

    # Définir les dossiers de sortie
    data_path = Path(args.data_dir)
    if args.imagettes_dir is None:
        imagettes_dir = data_path.parent / f"{data_path.name}_imagettes"
    else:
        imagettes_dir = Path(args.imagettes_dir)

    if args.cov_dir is None:
        cov_dir = data_path.parent / f"{data_path.name}_cov"
    else:
        cov_dir = Path(args.cov_dir)

    print(f"Imagettes: {imagettes_dir}")
    print(f"Covariances: {cov_dir}")

    if args.verify_only:
        # Seulement vérifier
        verify_covariances(cov_dir, args.verify_samples)
    else:
        # Pipeline complet
        imagette_paths = []

        # Étape 1: Extraction des imagettes
        if not args.skip_extraction:
            print("\n" + "-"*60)
            print("ÉTAPE 1: EXTRACTION DES IMAGETTES")
            print("-"*60)
            imagette_paths = extract_all_imagettes(args.data_dir, imagettes_dir, args.num_workers)
        else:
            print("\nUtilisation des imagettes existantes...")
            imagette_paths = list(imagettes_dir.glob("*.tif"))
            print(f"Imagettes trouvées: {len(imagette_paths)}")

        # Étape 2: Calcul des covariances
        if not args.skip_covariances and imagette_paths:
            print("\n" + "-"*60)
            print("ÉTAPE 2: CALCUL DES COVARIANCES")
            print("-"*60)
            stats = precompute_covariances(
                imagette_paths=imagette_paths,
                cov_dir=cov_dir,
                normalization_factor=args.normalization_factor,
                num_workers=args.num_workers,
                force_recompute=args.force
            )

            # Vérifier si demandé
            if args.verify:
                print("\n")
                verify_covariances(cov_dir, args.verify_samples)

        print("\n" + "="*60)
        print("TRAITEMENT TERMINÉ")
        print("="*60)


if __name__ == "__main__":
    main()