import os
import numpy as np
import zarr
import tqdm
import shutil

def process_and_save_to_zarr(input_folder, output_file):
    """
    Lit les données des fichiers .npy, extrait les parties réelle et imaginaire,
    et les écrit dans un fichier Zarr avec les labels associés.

    Args:
        input_folder (str): Chemin du dossier contenant les sous-dossiers.
        output_file (str): Chemin du fichier Zarr de sortie.
    """
    # Supprimer le fichier Zarr s'il existe déjà
    if os.path.exists(output_file):
        print(f"Removing existing Zarr file: {output_file}")
        shutil.rmtree(output_file)
    
    # Créer un fichier Zarr pour stocker les données
    zarr_root = zarr.open_group(output_file, mode='w')

    # Obtenir le nom du fichier de sortie (sans le chemin)
    output_basename = os.path.basename(output_file)
    
    # Parcourir chaque sous-dossier
    for city in os.listdir(input_folder):
        city_path = os.path.join(input_folder, city)
        
        # Ignorer si ce n'est pas un dossier OU si c'est le fichier de sortie
        if not os.path.isdir(city_path) or city == output_basename:
            continue
        
        # Vérifier que le dossier contient bien les fichiers attendus
        hh_file = os.path.join(city_path, "HH_Complex_Patches.npy")
        hv_file = os.path.join(city_path, "HV_Complex_Patches.npy")
        labels_file = os.path.join(city_path, "Labels.npy")
        
        # Vérifier que tous les fichiers existent
        if not all(os.path.exists(f) for f in [hh_file, hv_file, labels_file]):
            print(f"⚠️  Skipping {city}: missing required files")
            continue

        print(f"Processing city: {city}")

        # Charger les labels
        labels = np.load(labels_file, mmap_mode='r')
        # S'assurer que labels est 1D
        labels = np.atleast_1d(labels).flatten()

        # Charger les données HH et HV en mode mémoire mappée
        hh_data = np.load(hh_file, mmap_mode='r')
        hv_data = np.load(hv_file, mmap_mode='r')

        # Vérifier que les dimensions sont compatibles
        assert hh_data.shape == hv_data.shape, "HH et HV doivent avoir les mêmes dimensions"
        num_samples = hh_data.shape[0]
        
        print(f"  Num samples: {num_samples}, Data shape: {hh_data.shape}, Labels shape: {labels.shape}")

        # Déterminer la taille des chunks
        chunk_size_samples = 100
        
        # Créer un groupe pour la ville dans le fichier Zarr
        city_group = zarr_root.create_group(city)

        # Pour Zarr v3, utiliser create_array
        zarr_data = city_group.create_array(
            name='data',
            shape=(num_samples, hh_data.shape[1], hh_data.shape[2], 4),
            dtype=np.float32,
            chunks=(chunk_size_samples, hh_data.shape[1], hh_data.shape[2], 4),
            fill_value=0
        )

        # Stocker les labels - chunks doit être un tuple
        zarr_labels = city_group.create_array(
            name='labels',
            shape=labels.shape,
            dtype=np.int32,
            chunks=(chunk_size_samples,),  # Tuple pour 1D array
            fill_value=0
        )
        # Écrire les labels d'un coup
        zarr_labels[:] = np.array(labels, dtype=np.int32)

        # Traiter les données par chunks
        for i in tqdm.tqdm(range(0, num_samples, chunk_size_samples), desc=f"Processing {city}"):
            # Charger un chunk des données
            hh_chunk = hh_data[i:i + chunk_size_samples]
            hv_chunk = hv_data[i:i + chunk_size_samples]

            # Extraire les parties réelle et imaginaire
            hh_real = np.real(hh_chunk).astype("float32")
            hh_imag = np.imag(hh_chunk).astype("float32")
            hv_real = np.real(hv_chunk).astype("float32")
            hv_imag = np.imag(hv_chunk).astype("float32")

            # Empiler les résultats
            stacked_chunk = np.stack([hh_real, hh_imag, hv_real, hv_imag], axis=-1)

            # Écrire dans le fichier Zarr
            zarr_data[i:i + chunk_size_samples] = stacked_chunk

        print(f"✅ Finished processing city: {city}")

    # Zarr n'a pas besoin d'être fermé explicitement (synchronisation automatique)
    print(f"\n Data successfully written to {output_file}")


# Exemple d'utilisation
input_folder = "/home/mgallet/Téléchargements/S1SLC_CVDL"
output_file = "/home/mgallet/Téléchargements/S1SLC_CVDL/complex_HHHV.zarr"

# Afficher les dossiers qui seront traités
print("Dossiers trouvés:")
output_basename = os.path.basename(output_file)
for city in os.listdir(input_folder):
    city_path = os.path.join(input_folder, city)
    if os.path.isdir(city_path) and city != output_basename:
        has_files = all(os.path.exists(os.path.join(city_path, f)) 
                       for f in ["HH_Complex_Patches.npy", "HV_Complex_Patches.npy", "Labels.npy"])
        status = "✅" if has_files else "(missing files)"
        print(f"  {status} {city}")

print("\n Starting conversion...")
process_and_save_to_zarr(input_folder, output_file)