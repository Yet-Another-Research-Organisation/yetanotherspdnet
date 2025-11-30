# ========================================
# FileName: data.py
# Date: 29 juin 2023 - 16:38
# Author: Ammar Mian
# Email: ammar.mian@univ-smb.fr
# GitHub: https://github.com/ammarmian
# Brief: Data classes and utilities
# =========================================

import os
import glob
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset
from scipy.stats import wishart
from itertools import product
from tqdm import tqdm

# Type hints
from typing import Optional, Iterable, Tuple, List, Union, Callable, Dict
from numpy.typing import ArrayLike

from torchvision.datasets import ImageFolder
import pandas as pd


def replace_(C, D, unknown_value="UNKNOWN"):
    result = []
    for c, d in zip(C, D):
        if d == unknown_value:
            result.append(unknown_value + "_" + c[0])
        else:
            result.append(d)
    return result


# ----------------------------
# Dataset classes
# ----------------------------
class RegroupedImageFolder(ImageFolder):
    """
    Extended ImageFolder dataset that allows regrouping of classes.

    Arguments:
        root (string): Root directory path containing class folders
        transform (callable, optional): Transform to be applied to the images
        target_transform (callable, optional): Transform to be applied to the targets
        regrouping (Dict[str, List[str]]): Dictionary mapping new class names to lists of original class names
    """

    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        regrouping=None,
    ):
        # Initialize the parent ImageFolder class
        super().__init__(
            root=root, transform=transform, target_transform=target_transform
        )

        # If no regrouping specified, keep original classes
        if regrouping is None:
            return

        # Store original information
        self.regrouping = regrouping
        self.original_class_to_idx = self.class_to_idx.copy()
        self.original_classes = self.classes.copy()
        self.original_targets = self.targets.copy()

        # Create new classes list based on regrouping dictionary keys
        self.classes = list(regrouping.keys())

        # Create new class_to_idx mapping
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Create mapping from original class names to new indices
        orig_name_to_new_idx = {}
        for new_cls_name, orig_cls_list in regrouping.items():
            new_idx = self.class_to_idx[new_cls_name]
            for orig_cls_name in orig_cls_list:
                orig_name_to_new_idx[orig_cls_name] = new_idx

        # Create mapping from original idx to new idx
        orig_idx_to_new_idx = {}
        for orig_cls_name, orig_idx in self.original_class_to_idx.items():
            if orig_cls_name in orig_name_to_new_idx:
                orig_idx_to_new_idx[orig_idx] = orig_name_to_new_idx[orig_cls_name]

        # Update all targets using the mapping and filter samples
        new_targets = []
        new_imgs = []
        new_samples = []

        for i, (img_path, target) in enumerate(self.samples):
            orig_class_idx = self.original_targets[i]
            if (
                orig_class_idx in orig_idx_to_new_idx
            ):  # Only keep samples from mapped classes
                new_idx = orig_idx_to_new_idx[orig_class_idx]
                new_targets.append(new_idx)
                new_imgs.append(self.imgs[i])
                new_samples.append(
                    (img_path, new_idx)
                )  # Update target in samples tuple

        # Replace class data
        self.targets = new_targets
        self.imgs = new_imgs
        self.samples = new_samples

        # Store number of classes after regrouping
        self.num_classes = len(self.classes)

    def print_class_statistics(self):
        """
        Print statistics about class distribution
        """
        print(f"Original classes: {self.original_classes}")
        print(f"Regrouped classes: {self.classes}")

        # Count instances per regrouped class
        class_counts = {}
        for target in self.targets:
            class_name = self.classes[target]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        print("\nClass distribution after regrouping:")
        for cls_name, count in class_counts.items():
            print(f"  {cls_name}: {count} images")


class GPRFilteredImageFolder(RegroupedImageFolder):
    """
    Extended RegroupedImageFolder dataset that allows filtering based on radar metadata.

    Arguments:
        root (string): Root directory path containing class folders
        metadata_csv (string): Path to the CSV file with metadata information
        transform (callable, optional): Transform to be applied to the images
        target_transform (callable, optional): Transform to be applied to the targets
        regrouping (Dict[str, List[str]]): Dictionary mapping new class names to lists of original class names
    """

    def __init__(
        self,
        root: str,
        metadata_csv: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        regrouping: Optional[Dict[str, List[str]]] = None,
    ):
        # Initialize the parent RegroupedImageFolder class
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            regrouping=regrouping,
        )

        # Load and process the metadata CSV
        self.metadata_df = pd.read_csv(metadata_csv)

        # Create a dictionary for fast lookup: image_path -> metadata_row
        self.metadata_map = {}
        for idx, row in self.metadata_df.iterrows():
            # Use the relative path for matching with self.imgs paths
            rel_path = os.path.join(
                os.path.basename(os.path.dirname(row["image_path"])),
                os.path.basename(row["image_path"]),
            )
            self.metadata_map[rel_path] = row

        # Store original samples and targets
        self.original_samples = self.samples.copy()
        self.original_targets = self.targets.copy()
        self.original_imgs = self.imgs.copy()

        # Store original classes for print_class_statistics() method compatibility
        if hasattr(self, "classes"):
            self.original_classes = self.classes.copy()

        # Add metadata to each sample
        self._add_metadata_to_samples()

    def _add_metadata_to_samples(self):
        """
        Adds metadata to each sample in the dataset.
        """
        self.samples_with_metadata = []
        valid_samples_idx = []

        for i, (img_path, target) in enumerate(self.samples):
            # Get relative path for matching with metadata_map keys
            rel_path = os.path.join(
                os.path.basename(os.path.dirname(img_path)), os.path.basename(img_path)
            )

            if rel_path in self.metadata_map:
                metadata = self.metadata_map[rel_path]
                self.samples_with_metadata.append((img_path, target, metadata))
                valid_samples_idx.append(i)

        # If some samples don't have metadata, we keep only those with metadata
        if len(valid_samples_idx) < len(self.samples):
            print(
                f"Warning: {len(self.samples) - len(valid_samples_idx)} samples don't have metadata and will be excluded."
            )
            self.samples = [self.samples[i] for i in valid_samples_idx]
            self.targets = [self.targets[i] for i in valid_samples_idx]
            self.imgs = [self.imgs[i] for i in valid_samples_idx]

    def apply_filter(self, filter_dict: Dict[str, List[str]]):
        """
        Filter samples based on metadata attributes.

        Args:
            filter_dict: Dictionary with keys being metadata attributes and values being lists
                        of accepted values for those attributes.
                        Example: {'elevation': ['0'], 'ground_type': ['sable']}
        """
        if not filter_dict:
            return

        # Validate filter keys
        valid_keys = [
            "class",
            "elevation",
            "frequency",
            "ground_type",
            "mask_category",
            "mask_subcategory",
            "mask_category_confidence",
            "mask_subcategory_confidence",
        ]
        for key in filter_dict.keys():
            if key not in valid_keys and key not in self.metadata_df.columns:
                raise ValueError(
                    f"Filter key '{key}' is not a valid metadata attribute. "
                    f"Valid keys are: {valid_keys + list(self.metadata_df.columns)}"
                )

        # Reset to ensure we're filtering from all samples
        self.samples = self.original_samples.copy()
        self.targets = self.original_targets.copy()
        self.imgs = self.original_imgs.copy()
        self._add_metadata_to_samples()

        # Apply filters
        filtered_samples = []
        filtered_targets = []
        filtered_imgs = []

        for img_path, target, metadata in self.samples_with_metadata:
            keep_sample = True

            for filter_key, accepted_values in filter_dict.items():
                if (
                    filter_key in metadata
                    and str(metadata[filter_key]) not in accepted_values
                ):
                    keep_sample = False
                    break

            if keep_sample:
                filtered_samples.append((img_path, target))
                filtered_targets.append(target)
                filtered_imgs.append(img_path)

        # Update dataset with filtered data
        self.samples = filtered_samples
        self.targets = filtered_targets
        self.imgs = filtered_imgs

        print(f"Applied filters: {filter_dict}")
        print(f"Remaining samples: {len(self.samples)}")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def print_class_statistics(self):
        """
        Override the parent method to be compatible with GPRFilteredImageFolder.
        Print statistics about class distribution.
        """
        # Print original classes before regrouping, if available
        if hasattr(self, "original_class_to_idx") and hasattr(self, "regrouping"):
            print("Original classes before regrouping:")
            for cls in sorted(self.original_class_to_idx.keys()):
                print(f"  {cls}")

        # Print current classes after regrouping
        print("\nCurrent classes:")
        for cls in self.classes:
            print(f"  {cls}")

        # Count instances per class
        class_counts = {}
        for target in self.targets:
            class_name = self.classes[target]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        print("\nClass distribution:")
        for cls_name, count in sorted(class_counts.items()):
            print(f"  {cls_name}: {count} images")

        # If regrouping is used, show the mapping
        if hasattr(self, "regrouping") and self.regrouping is not None:
            print("\nClass regrouping mapping:")
            for new_cls, orig_classes in self.regrouping.items():
                print(f"  {new_cls}: {orig_classes}")

    def print_metadata_statistics(self):
        """
        Print statistics about the metadata distribution in the current filtered dataset
        """
        if len(self.samples) == 0:
            print("No samples in the dataset after filtering.")
            return

        metadata_cols = [
            "class",
            "elevation",
            "frequency",
            "ground_type",
            "mask_category",
            "mask_subcategory",
        ]

        print("\nMetadata statistics for the current filtered dataset:")

        for col in metadata_cols:
            if col in self.metadata_df.columns:
                # Get values only for current samples
                values = []
                for img_path, _ in self.samples:
                    rel_path = os.path.join(
                        os.path.basename(os.path.dirname(img_path)),
                        os.path.basename(img_path),
                    )
                    if rel_path in self.metadata_map:
                        values.append(str(self.metadata_map[rel_path][col]))

                # Count occurrences
                value_counts = {}
                for value in values:
                    value_counts[value] = value_counts.get(value, 0) + 1

                print(f"\n{col} distribution:")
                for value, count in sorted(value_counts.items()):
                    print(
                        f"  {value}: {count} images ({count / len(self.samples) * 100:.1f}%)"
                    )


# ========================================
# Class: FixedWishartDataset
# ========================================
class FixedWishartDataset(Dataset):
    """Dataset of Wishart distributed matrices with a fixed degree of freedom
    and scale matrix."""

    def __init__(
        self,
        df: int,
        scale: ArrayLike,
        size: Optional[int] = 1000,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Constructor of the dataset with a given degree of freedom
        and scale matrix.

        Parameters
        ----------
        df : int
            Degree of freedom of the Wishart distribution.

        scale : ArrayLike
            Scale matrix of the Wishart distribution.

        size : int, optional
            Number of samples in the dataset, by default 1000.

        seed : int, optional
            Seed for the random number generator, by default None.

        device : torch.device, optional
            Device on which the data is stored, by default None.
        """
        self.df = df
        self.scale = scale
        self.size = size
        self.seed = seed
        self.device = device

        # Create the random number generator
        self.rng = np.random.default_rng(seed)

        # Create the Wishart distribution
        self.wishart = wishart(df, scale)

    def __len__(self) -> int:
        """Returns the size of the dataset"""
        return self.size

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Returns the idx-th sample of the dataset"""

        if idx >= self.size:
            raise IndexError(
                f"Index {idx} out of range (size of the dataset is {{self.size}})"
            )

        # Generate the random matrix
        X = self.wishart.rvs(size=1, random_state=self.rng).astype(np.float32)

        # Convert to tensor
        X = torch.from_numpy(X).to(self.device)

        return X

    def get_all(self) -> torch.Tensor:
        """Returns all the samples of the dataset"""
        return torch.from_numpy(
            self.wishart.rvs(size=self.size, random_state=self.rng)
        ).to(self.device)

    def __repr__(self) -> str:
        """Returns the representation of the dataset"""
        return (
            f"WishartDataset(df={self.df}, scale={self.scale}, "
            f"size={self.size}, seed={self.seed})"
        )


# ========================================
# Class: WishartDataset
# ========================================
class WishartDataset(Dataset):
    """Dataset of Wishart distributed matrices with a variable degree of
    freedom and scale matrix."""

    def __init__(
        self,
        df: Iterable[int],
        scale: Iterable[ArrayLike],
        size: Optional[int] = 1000,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Constructor of the dataset with a given degree of freedom
        and scale matrix.

        Parameters
        ----------
        df : Iterable[int]
            Iterable of degree of freedom of the Wishart distribution.

        scale : Iterable[ArrayLike]
            Iterable of scale matrix of the Wishart distribution.

        size : int, optional
            Number of samples in the dataset per combination of parameters,
            by default 1000.

        seed : int, optional
            Seed for the random number generator, by default None.

        device : torch.device, optional
            Device on which the data is stored, by default None.
        """
        self.df = df
        self.scale = scale
        self.size = size
        self.seed = seed
        self.device = device

        # Create the random number generator
        self.rng = np.random.default_rng(seed)

        # Create a mapping between the index of the dataset and the
        # index of the parameters + the number of samples per combination
        self.idx_to_param = list(
            product(range(len(self.df)), range(len(self.scale)), range(self.size))
        )
        self.total_size = len(self.idx_to_param)

    def __len__(self) -> int:
        """Returns the size of the dataset"""
        return self.total_size

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Returns the idx-th sample of the dataset"""

        # Get the index of the parameters and the index per combination
        idx_df, idx_scale, idx_sample = self.idx_to_param[idx]

        # Check if the index is valid
        if (
            idx_df >= len(self.df)
            or idx_scale >= len(self.scale)
            or idx_sample >= self.size
        ):
            raise IndexError(
                f"Index {idx} out of range (size of the dataset is {{self.total_size}})"
            )

        # Generate the random matrix
        X = (
            wishart(self.df[idx_df], self.scale[idx_scale])
            .rvs(size=1, random_state=self.rng)
            .astype(np.float32)
        )

        # Convert to tensor
        X = torch.from_numpy(X).to(self.device)

        return X

    def __repr__(self) -> str:
        """Returns the representation of the dataset"""
        return (
            f"WishartDataset(df={self.df}, scale={self.scale}, "
            f"size={self.size}, seed={self.seed})"
        )


# ========================================
# AFEW SPnet dataset
# ========================================
def loadmat_spdnet(file_path: str):
    try:
        data = loadmat(file_path)
        if "Y1" not in data:
            raise KeyError(f"'Y1' key not found in {file_path}")
        return data["Y1"]
    except Exception as e:
        raise ValueError(f"Failed to load {file_path}: {e}")


class AFEWSPDnetDataset(Dataset):
    """Dataset from paper: A Riemannian Network for SPD Matrix Learning, CVPR 2017.

    The dataset is provided by the authors from their GitHub repository:
        https://github.com/zhiwu-huang/SPDNet/tree/master

    It consists in pre-computed SPD matrices from AFEW dataset, which doesn't
    help reproducibility. We suppose that the data is already downloaded and
    unzipped in directory_path."""

    def __init__(
        self,
        directory_path: str,
        preload: bool = False,
        shuffle: bool = False,
        subset: str = "train",
        rng: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
        verbose: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Constructor of the dataset.

        Parameters
        ----------
        directory_path : str
            Path to the directory containing the dataset.

        subset : str, optional
            Subset of the dataset to load, by default "train".

        preload : bool, optional
           Whether to preload the dataset in memory, by default False.

        shuffle : bool, optional
           Whether to shuffle the dataset, by default False.

        rng : np.random.Generator, optional
           Random number generator, by default None.

        device : torch.device, optional
           Device on which the data is stored, by default None.

        verbose : int, optional
            Verbosity level, by default 0.

        dtype: torch.dtype, optional
            dtype of data, by default float32.
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory {directory_path} does not exist")

        super().__init__()
        self.directory_path = directory_path
        self.preload = preload
        self.device = device
        self.verbose = verbose
        self.shuffle = shuffle
        self.rng = rng
        self.subset = subset
        self.labels_str = [
            "Angry",
            "Disgust",
            "Fear",
            "Happy",
            "Neutral",
            "Sad",
            "Surprise",
        ]

        self.d = 400
        self.dtype = dtype

        self._discover_files()
        self._preload_data()
        self._shuffle()

    def _shuffle(self) -> None:
        """Shuffle the dataset"""
        if self.shuffle:
            indexes = torch.randperm(len(self.list_files), generator=self.rng)
            # Convert tensor indices to list for proper indexing
            indexes = indexes.tolist()
            self.list_files = [self.list_files[i] for i in indexes]
            self.labels = [self.labels[i] for i in indexes]

    def _discover_files(self) -> None:
        """Discovers the files in the dataset directory"""
        self.list_files = glob.glob(
            os.path.join(self.directory_path, self.subset, "**/*.mat")
        )
        if len(self.list_files) == 0:
            self.list_files = glob.glob(
                os.path.join(
                    self.directory_path,
                    "spdface_400_inter_histeq",
                    self.subset,
                    "**/*.mat",
                )
            )
        # Find the labels from the file names. -1 because labels start at 0
        # in torch
        self.labels = [
            int(os.path.basename(os.path.dirname(x))) - 1 for x in self.list_files
        ]
        if len(self.labels) == 0:
            print(f"No data found for subset {self.subset}")

    def __len__(self) -> int:
        return len(self.list_files)

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        if self.preload:
            return self.data[index], self.labels[index]
        else:
            X = torch.from_numpy(loadmat_spdnet(self.list_files[index])).to(
                self.device, dtype=self.dtype
            )
            return X, self.labels[index]

    def __repr__(self) -> str:
        string = f"AFEWSPnetDataset(directory_path={self.directory_path}, "
        string += f"preload={self.preload}, subset={self.subset})"
        return string

    def _preload_data(self) -> None:
        """Preload the data"""
        if self.preload:
            if self.verbose:
                pbar = tqdm(total=len(self.list_files))
            self.data = []
            for file in self.list_files:
                if self.verbose:
                    filename = os.path.basename(file)
                    pbar.set_description(f"Loading ({filename})")
                    pbar.refresh()
                self.data.append(
                    torch.from_numpy(loadmat_spdnet(file)).to(
                        self.device, dtype=self.dtype
                    )
                )

                if self.verbose:
                    pbar.update(1)

            if self.verbose:
                pbar.close()
        self.n_classes = len(np.unique(self.labels))


class SPMartiDataset(Dataset):
    """Pre-computed covariance matrices data described in:
       https://marti.ai/qfin/2020/02/03/sp500-sharpe-vs-corrmats.html
       Accessed: 11/09/23 3:40pm.

    The dataset corresponds to financial data from S&P evaluations
    with classes:
        * correlation matrices associated to a stressed market,
        * correlation matrices associated to a rally market,
        * correlation matrices associated to a normal market.

    It consists in pre-computed SPD matrices. We suppose that the data is
    already downloaded and unzipped in directory_path."""

    def __init__(
        self,
        directory_path: str,
        preload: bool = False,
        shuffle: bool = False,
        rng: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
        verbose: int = 0,
    ) -> None:
        """Constructor of the dataset.

        Parameters
        ----------
        directory_path : str
            Path to the directory containing the dataset.

        preload : bool, optional
           Whether to preload the dataset in memory, by default False.

        shuffle : bool, optional
           Whether to shuffle the dataset, by default False.

        rng : np.random.Generator, optional
           Random number generator, by default None.

        device : torch.device, optional
           Device on which the data is stored, by default None.

        verbose : int, optional
            Verbosity level, by default 0.
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory {directory_path} does not exist")

        super().__init__()
        self.directory_path = directory_path
        self.preload = preload
        self.device = device
        self.verbose = verbose
        self.shuffle = shuffle
        self.rng = rng
        self.labels_str = ["Stressed", "Normal", "Rally"]

        self.d = 80

        self._discover_files()
        self._preload_data()
        self._shuffle()

    def _shuffle(self) -> None:
        """Shuffle the dataset"""
        # TODO: fix this
        if self.shuffle:
            raise NotImplementedError(
                "Sorry this code is not working properly, to debug!!!"
            )
            # indexes = torch.randperm(len(self.list_files), generator=self.rng)
            # self.list_files = [self.list_files[i] for i in indexes]
            # self.labels = [self.labels[i] for i in indexes]

    def _discover_files(self) -> None:
        """Discovers the files in the dataset directory"""
        self.list_files = glob.glob(os.path.join(self.directory_path, "**/*.npy"))
        if len(self.list_files) == 0:
            self.list_files = glob.glob(os.path.join(self.directory_path, "*.npy"))
        # Find the labels from the file names
        self.labels = [
            int(os.path.basename(x).split("class_")[-1].split(".npy")[0])
            for x in self.list_files
        ]

    def __len__(self) -> int:
        return len(self.list_files)

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        if self.preload:
            return self.data[index], self.labels[index]
        else:
            X = torch.from_numpy(np.load(self.list_files[index])).real.to(self.device)
            return X, self.labels[index]

    def __repr__(self) -> str:
        string = f"SPMartiDataset(directory_path={self.directory_path}, "
        string += f"preload={self.preload})"
        return string

    def _preload_data(self) -> None:
        """Preload the data"""
        if self.preload:
            if self.verbose:
                pbar = tqdm(total=len(self.list_files))
            self.data = []
            for file in self.list_files:
                if self.verbose:
                    pbar.set_description(f"Loading {file}")
                    pbar.refresh()
                self.data.append(torch.from_numpy(np.load(file)).real.to(self.device))

                if self.verbose:
                    pbar.update(1)

            if self.verbose:
                pbar.close()
        self.n_classes = len(np.unique(self.labels))


class HDM05Dataset(Dataset):
    """Dataset of action recongition from:
    https://resources.mpi-inf.mpg.de/HDM05/, that has been preprocessed
    to generate SPD matrices from the skeleton data. The preprocessing has
    been done and the data has been downloaded from:
    https://github.com/zhiwu-huang/SPDNet"""

    def __init__(
        self,
        directory_path: str,
        metadata_sets: Union[List[int], str] = "both",  # [1], [2], [1,2], or "both"
        use_metadata: bool = True,  # Whether to use metadata at all
        preload: bool = False,
        shuffle: bool = False,
        rng: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
        verbose: int = 0,
    ) -> None:
        """Constructor of the dataset.

        Parameters
        ----------
        directory_path : str
            Path to the directory containing the dataset.

        metadata_sets : Union[List[int], str], optional
            Which metadata sets to include (only used if use_metadata=True):
            - [1]: Use only set 1 files (augmented, ~18k files)
            - [2]: Use only set 2 files (original, ~1.2k files)
            - [1,2] or "both": Use both sets (~19k files)

        use_metadata : bool, optional
            Whether to use metadata filtering at all:
            - True: Only include files that have metadata entries (may lose ~50% of files)
            - False: Use all existing files, ignore metadata (use for basic experiments)

        preload : bool, optional
           Whether to preload the dataset in memory, by default False.

        shuffle : bool, optional
           Whether to shuffle the dataset, by default False.

        rng : np.random.Generator, optional
           Random number generator, by default None.

        device : torch.device, optional
           Device on which the data is stored, by default None.

        verbose : int, optional
            Verbosity level, by default 0.
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory {directory_path} does not exist")

        super().__init__()
        self.directory_path = directory_path
        self.metadata_sets = metadata_sets if metadata_sets != "both" else [1, 2]
        self.use_metadata = use_metadata
        self.preload = preload
        self.device = device
        self.verbose = verbose
        self.shuffle = shuffle
        self.rng = rng
        self.d = 93

        if self.use_metadata:
            self._load_metadata()
            self._discover_files_with_metadata()
        else:
            self._discover_files_without_metadata()

        self._preload_data()
        self._shuffle()

    def _load_metadata(self) -> None:
        """Load the metadata.mat file to get predefined splits"""
        if not self.use_metadata:
            return

        metadata_path = os.path.join(self.directory_path, "metadata.mat")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"metadata.mat not found at {metadata_path}")

        metadata = loadmat(metadata_path)
        spd_data = metadata["spd"][0, 0]

        # Extract the relevant information
        self.file_ids = spd_data["id"][0]  # IDs of files
        self.file_names = spd_data["name"][0]  # Relative paths
        self.file_sets = spd_data["set"][
            0
        ]  # Train/test split (1=augmented, 2=original)
        self.file_labels = spd_data["label"][0]  # Action class labels

        if self.verbose:
            print(f"Loaded metadata: {len(self.file_ids)} files")
            print(f"Set 1 (augmented): {np.sum(self.file_sets == 1)} samples")
            print(f"Set 2 (original): {np.sum(self.file_sets == 2)} samples")
            print(f"Requested metadata_sets: {self.metadata_sets}")

    def _discover_files_with_metadata(self) -> None:
        """Discovers the files based on metadata set filtering"""

        # First, discover all existing files in the feature directory
        self._discover_existing_files()

        # Filter by metadata sets
        if isinstance(self.metadata_sets, str) and self.metadata_sets == "both":
            set_mask = np.ones_like(self.file_sets, dtype=bool)
        else:
            set_mask = np.isin(self.file_sets, self.metadata_sets)

        # Apply mask to get filtered metadata
        self.filtered_file_names = self.file_names[set_mask]
        self.filtered_labels = (
            self.file_labels[set_mask] - 1
        )  # Convert to 0-based indexing
        self.filtered_sets = self.file_sets[
            set_mask
        ]  # Keep track of which set each file belongs to

        # Create reverse mapping: extract action/filename from metadata and map to metadata info
        self.metadata_map = {}  # Map (action_folder, filename) -> (label, set, metadata_index)
        self.label_to_action = {}  # Map original label -> action name

        if self.verbose > 1:
            print(
                f"\nCreating reverse mapping from {len(self.filtered_file_names)} metadata entries..."
            )

        for i, file_name in enumerate(self.filtered_file_names):
            metadata_path = file_name[0].replace(
                "\\", "/"
            )  # Normalize to forward slashes

            if self.verbose > 1 and i < 5:
                print(f"  Processing metadata: {metadata_path}")

            # Extract action and filename from metadata path
            action_folder, filename = self._extract_action_filename(metadata_path)

            if action_folder and filename:
                key = (action_folder, filename)
                original_label = self.filtered_labels[i]
                self.metadata_map[key] = (original_label, self.filtered_sets[i], i)

                # Store action name for this label
                self.label_to_action[original_label] = action_folder

                if self.verbose > 1 and i < 5:
                    print(
                        f"    -> Mapped ({action_folder}, {filename}) to label {original_label}, set {self.filtered_sets[i]}"
                    )

        if self.verbose:
            print(f"Created metadata mapping with {len(self.metadata_map)} entries")

            # Debug: Show some metadata keys for comparison
            if self.verbose > 1:
                print("Sample metadata keys:")
                for i, key in enumerate(list(self.metadata_map.keys())[:5]):
                    print(f"  {key}")

        # Now match existing files to metadata
        self.list_files = []
        valid_labels = []
        valid_sets = []

        if self.verbose > 1:
            print(
                f"\nMatching {len(self.existing_files)} existing files to metadata..."
            )

        matched_count = 0
        unmatched_existing = []

        for rel_path, full_path in self.existing_files.items():
            # Extract action and filename from existing file path
            action_folder, filename = self._extract_action_filename_from_path(rel_path)

            if action_folder and filename:
                key = (action_folder, filename)

                if key in self.metadata_map:
                    label, file_set, meta_idx = self.metadata_map[key]
                    self.list_files.append(full_path)
                    valid_labels.append(label)
                    valid_sets.append(file_set)
                    matched_count += 1

                    if self.verbose > 1 and matched_count <= 5:
                        print(
                            f"  ✓ Matched: {rel_path} -> label {label}, set {file_set}"
                        )
                else:
                    unmatched_existing.append((rel_path, key))
                    if self.verbose > 1 and len(unmatched_existing) <= 5:
                        print(f"  ✗ No metadata for: {rel_path} (key: {key})")

        # Check which metadata entries weren't matched
        matched_metadata_keys = set()
        for rel_path, full_path in self.existing_files.items():
            action_folder, filename = self._extract_action_filename_from_path(rel_path)
            if action_folder and filename:
                key = (action_folder, filename)
                if key in self.metadata_map:
                    matched_metadata_keys.add(key)

        unmatched_metadata = []
        for key in self.metadata_map.keys():
            if key not in matched_metadata_keys:
                unmatched_metadata.append(key)

        if self.verbose:
            print(f"Matched {len(self.list_files)} existing files to metadata")
            print(f"Unmatched existing files: {len(unmatched_existing)}")
            print(f"Unmatched metadata entries: {len(unmatched_metadata)}")

            if self.verbose > 1:
                print(f"\nSample unmatched existing files:")
                for rel_path, key in unmatched_existing[:10]:
                    print(f"  {rel_path} -> {key}")

                print(f"\nSample unmatched metadata entries:")
                for key in unmatched_metadata[:10]:
                    print(f"  {key}")

                # Try to find pattern in unmatched files
                print(f"\nAnalyzing filename patterns...")
                existing_patterns = set()
                metadata_patterns = set()

                for rel_path, key in unmatched_existing[:50]:
                    action, filename = key
                    if "_" in filename:
                        existing_patterns.add("contains_underscore")
                    if filename.isdigit() or filename.replace(".mat", "").isdigit():
                        existing_patterns.add("simple_number")
                    if any(char.isalpha() for char in filename):
                        existing_patterns.add("contains_letters")

                for key in unmatched_metadata[:50]:
                    action, filename = key
                    if "_" in filename:
                        metadata_patterns.add("contains_underscore")
                    if filename.isdigit() or filename.replace(".mat", "").isdigit():
                        metadata_patterns.add("simple_number")
                    if any(char.isalpha() for char in filename):
                        metadata_patterns.add("contains_letters")

                print(f"  Unmatched existing file patterns: {existing_patterns}")
                print(f"  Unmatched metadata patterns: {metadata_patterns}")

        # Convert to arrays and tensors
        self.filtered_labels = np.array(valid_labels)
        self.filtered_sets = np.array(valid_sets)

        # Recompute classes and create consecutive label mapping
        unique_labels = np.unique(self.filtered_labels)
        self.n_classes = len(unique_labels)

        # Create mapping from original labels to consecutive labels (0, 1, 2, ...)
        self.original_to_consecutive = {
            orig: consec for consec, orig in enumerate(unique_labels)
        }
        self.consecutive_to_original = {
            consec: orig for orig, consec in self.original_to_consecutive.items()
        }

        # Remap labels to be consecutive
        consecutive_labels = [
            self.original_to_consecutive[label] for label in self.filtered_labels
        ]
        self.labels = torch.tensor(consecutive_labels, dtype=torch.long)

        # Create consecutive label to action name mapping
        self.consecutive_label_to_action = {}
        for original_label, action_name in self.label_to_action.items():
            if original_label in self.original_to_consecutive:
                consecutive_label = self.original_to_consecutive[original_label]
                self.consecutive_label_to_action[consecutive_label] = action_name

        if self.verbose:
            print(
                f"Metadata sets {self.metadata_sets}: {len(self.list_files)} files matched"
            )
            print(f"  Set 1 files: {np.sum(self.filtered_sets == 1)}")
            print(f"  Set 2 files: {np.sum(self.filtered_sets == 2)}")
            print(
                f"Original labels present: {len(unique_labels)} out of {np.max(self.file_labels) if hasattr(self, 'file_labels') else 'unknown'}"
            )
            print(f"Remapped to consecutive labels: 0 to {self.n_classes - 1}")
            print(
                f"Label range: {min(consecutive_labels) if consecutive_labels else 0} to {max(consecutive_labels) if consecutive_labels else 0}"
            )

            if self.verbose > 1 and hasattr(self, "file_labels"):
                all_original_labels = set(self.file_labels - 1)  # Convert to 0-based
                missing_labels = all_original_labels - set(unique_labels)
                if missing_labels:
                    print(
                        f"Missing original labels: {sorted(list(missing_labels))[:10]}{'...' if len(missing_labels) > 10 else ''}"
                    )
                print(
                    f"Label mapping (first 10): {dict(list(self.original_to_consecutive.items())[:10])}"
                )
                if len(self.original_to_consecutive) > 10:
                    print(f"  ... and {len(self.original_to_consecutive) - 10} more")

                print(f"Sample label names:")
                for consec_label, action_name in list(
                    self.consecutive_label_to_action.items()
                )[:5]:
                    print(f"  {consec_label}: {action_name}")
                if len(self.consecutive_label_to_action) > 5:
                    print(f"  ... and {len(self.consecutive_label_to_action) - 5} more")

    def _discover_files_without_metadata(self) -> None:
        """Discover all files without using metadata (for basic experiments)"""

        # Discover all existing files in the feature directory
        self._discover_existing_files()

        # Use all existing files
        self.list_files = list(self.existing_files.values())

        # Create labels from action directory names
        action_to_label = {}
        labels = []

        for rel_path in self.existing_files.keys():
            action_folder, filename = self._extract_action_filename_from_path(rel_path)

            if action_folder:
                if action_folder not in action_to_label:
                    action_to_label[action_folder] = len(action_to_label)
                labels.append(action_to_label[action_folder])
            else:
                labels.append(0)  # Default label

        self.labels = torch.tensor(labels, dtype=torch.long)
        self.n_classes = len(action_to_label)

        # Create label to action mapping (reverse of action_to_label)
        self.label_to_action = {
            label: action for action, label in action_to_label.items()
        }

        # Create identity mapping for consistency with metadata version
        self.original_to_consecutive = {i: i for i in range(self.n_classes)}
        self.consecutive_to_original = {i: i for i in range(self.n_classes)}

        # For non-metadata case, consecutive labels are same as original labels
        self.consecutive_label_to_action = self.label_to_action.copy()

        # No metadata sets info
        self.filtered_sets = None

        if self.verbose:
            print(f"Using all {len(self.list_files)} existing files without metadata")
            print(f"Number of classes (action folders): {self.n_classes}")
            print(
                f"Label range: {min(labels) if labels else 0} to {max(labels) if labels else 0}"
            )
            if self.verbose > 1:
                print("Action to label mapping:")
                for action, label in sorted(action_to_label.items())[:10]:
                    print(f"  {action} -> {label}")
                if len(action_to_label) > 10:
                    print(f"  ... and {len(action_to_label) - 10} more")

                print("Label names:")
                for label, action in sorted(self.consecutive_label_to_action.items())[
                    :10
                ]:
                    print(f"  {label}: {action}")
                if len(self.consecutive_label_to_action) > 10:
                    print(
                        f"  ... and {len(self.consecutive_label_to_action) - 10} more"
                    )

    def get_original_label(self, consecutive_label: int) -> int:
        """Convert consecutive label back to original label"""
        return self.consecutive_to_original.get(consecutive_label, consecutive_label)

    def get_consecutive_label(self, original_label: int) -> Optional[int]:
        """Convert original label to consecutive label (returns None if not present)"""
        return self.original_to_consecutive.get(original_label, None)

    def get_label_names(self) -> dict:
        """Get mapping from consecutive integer labels to action names

        Returns
        -------
        dict
            Mapping from consecutive label (int) to action name (str)
        """
        return self.consecutive_label_to_action.copy()

    def get_label_name(self, consecutive_label: int) -> Optional[str]:
        """Get action name for a given consecutive label

        Parameters
        ----------
        consecutive_label : int
            The consecutive label to get the name for

        Returns
        -------
        Optional[str]
            Action name or None if label not found
        """
        return self.consecutive_label_to_action.get(consecutive_label, None)

    def _extract_action_filename(
        self, metadata_path: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract action folder and filename from metadata path"""
        # Handle paths like 'spd93_half_inter9/cartwheelLHandStart1Reps/1.mat'
        parts = metadata_path.strip().split("/")

        if len(parts) >= 2:
            # Remove any prefix directories and get last two parts
            action_folder = parts[-2]  # Second to last part
            filename = parts[-1]  # Last part
            return action_folder, filename
        elif len(parts) == 1:
            # Just a filename, no action folder
            return None, parts[0]
        else:
            return None, None

    def _extract_action_filename_from_path(
        self, file_path: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract action folder and filename from existing file path"""
        # Handle paths like 'cartwheelLHandStart1Reps/1.mat'
        parts = file_path.replace("\\", "/").split("/")

        if len(parts) >= 2:
            action_folder = parts[-2]  # Second to last part (action directory)
            filename = parts[-1]  # Last part (filename)
            return action_folder, filename
        elif len(parts) == 1:
            # Just a filename, no action folder
            return None, parts[0]
        else:
            return None, None

    def _discover_existing_files(self) -> None:
        """Discover all existing .mat files in the feature directory"""
        self.existing_files = {}  # Map relative path to absolute path

        feature_path = os.path.join(self.directory_path, "feature")
        if os.path.exists(feature_path):
            for root, dirs, files in os.walk(feature_path):
                for file in files:
                    if file.endswith(".mat"):
                        full_path = os.path.join(root, file)
                        # Create relative path from feature directory
                        rel_path = os.path.relpath(full_path, feature_path)
                        self.existing_files[rel_path] = full_path

        if self.verbose:
            print(f"Found {len(self.existing_files)} existing .mat files")
            if self.verbose > 1:
                print("Sample existing files:")
                for i, (rel_path, full_path) in enumerate(
                    list(self.existing_files.items())[:5]
                ):
                    print(f"  {rel_path} -> {full_path}")
                if len(self.existing_files) > 5:
                    print(f"  ... and {len(self.existing_files) - 5} more")

    def _shuffle(self) -> None:
        """Shuffle the dataset"""
        if self.shuffle:
            # Create random permutation
            if self.rng is not None:
                perm = torch.randperm(len(self.list_files), generator=self.rng)
            else:
                perm = torch.randperm(len(self.list_files))

            # Apply permutation
            self.list_files = [self.list_files[i] for i in perm]
            self.labels = self.labels[perm]

            if self.preload:
                self.data = [self.data[i] for i in perm]

    def _preload_data(self) -> None:
        """Preload the data"""
        if self.preload:
            if self.verbose:
                from tqdm import tqdm

                pbar = tqdm(total=len(self.list_files))

            self.data = []
            for file in self.list_files:
                if self.verbose:
                    pbar.set_description(f"Loading {os.path.basename(file)}")
                    pbar.refresh()

                try:
                    data = loadmat(file)["Y1"]
                    self.data.append(torch.from_numpy(data).to(self.device))
                except Exception as e:
                    print(f"Error loading {file}: {e}")
                    # Add a dummy matrix to maintain indexing
                    self.data.append(torch.zeros(93, 93, device=self.device))

                if self.verbose:
                    pbar.update(1)

            if self.verbose:
                pbar.close()

    def __len__(self) -> int:
        return len(self.list_files)

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        if self.preload:
            return self.data[index], self.labels[index]
        else:
            try:
                X = torch.from_numpy(loadmat(self.list_files[index])["Y1"]).to(
                    self.device
                )
                return X, self.labels[index]
            except Exception as e:
                print(f"Error loading {self.list_files[index]}: {e}")
                return torch.zeros(93, 93, device=self.device), self.labels[index]

    def __repr__(self) -> str:
        if self.use_metadata:
            return (
                f"HDM05Dataset(directory_path={self.directory_path}, "
                f"metadata_sets={self.metadata_sets}, use_metadata={self.use_metadata}, "
                f"preload={self.preload}, "
                f"n_samples={len(self.list_files)}, n_classes={self.n_classes})"
            )
        else:
            return (
                f"HDM05Dataset(directory_path={self.directory_path}, "
                f"use_metadata={self.use_metadata}, "
                f"preload={self.preload}, "
                f"n_samples={len(self.list_files)}, n_classes={self.n_classes})"
            )


class HDM05New(Dataset):
    """Simple HDM05 dataset for new format with .npy files.

    This dataset loads SPD matrices from .npy files with filename pattern:
    {id}_{unknown}_{label}.npy where label is the class (0-116).

    Based on the simple pattern provided by the user, keeping the best
    of both worlds: simplicity and compatibility with existing training code.
    """

    def __init__(self, path, names, dtype: torch.dtype = torch.float32):
        """Initialize dataset with path and list of filenames.

        Args:
            path (str): Path to the directory containing .npy files
            names (list): List of filenames to load
        """
        self._path = path
        self._names = names
        self.d = 93  # Matrix dimension for HDM05
        self.n_classes = 117  # Total number of classes in HDM05
        self.dtype = dtype

    def __len__(self):
        return len(self._names)

    def __getitem__(self, item):
        """Load and return a single sample.

        Args:
            item (int): Index of the sample to load

        Returns:
            tuple: (x, y) where x is the SPD matrix and y is the label
        """
        # Load the matrix from .npy file (no extra dimension like original)
        x = np.load(os.path.join(self._path, self._names[item]))
        x = torch.from_numpy(x).to(dtype=self.dtype)

        # Extract label from filename pattern: {id}_{unknown}_{label}.npy
        y = int(self._names[item].split(".")[0].split("_")[-1])
        y = torch.from_numpy(np.array(y)).long()

        return x, y


class DatasetRadar(Dataset):
    """Dataset for radar data with complex numpy arrays.

    This dataset loads complex radar data from .npy files with filename pattern:
    {id}_{value}_{label}.npy where label is the class (0, 1, 2).

    The data is processed by separating real and imaginary parts and concatenating
    them along the channel dimension.
    """

    def __init__(self, path, names, dtype=torch.float32, window_size=20):
        """Initialize dataset with path and list of filenames.

        Args:
            path (str): Path to the directory containing .npy files
            names (list): List of filenames to load
            dtype (torch.dtype): Data type for the tensors
            window_size (int): Size of windows for covariance computation
        """
        self._path = path
        self._names = names
        self.dtype = dtype
        self.window_size = window_size

        # SPDNet requires SPD matrices, so we always use covariance transformation
        self.d = 2 * window_size  # 40x40 real matrix from complex form mapping

        # Extract all unique labels to determine number of classes
        labels = [int(name.split(".")[0].split("_")[-1]) for name in names]
        self.n_classes = len(set(labels))

    def __len__(self):
        return len(self._names)

    def __getitem__(self, item):
        """Load and return a single sample.

        Args:
            item (int): Index of the sample to load

        Returns:
            tuple: (x, y) where x is the SPD covariance matrix and y is the label
        """
        # Load the complex array from .npy file
        x = np.load(os.path.join(self._path, self._names[item]))

        # Transform to 40x40 real SPD matrix using complex form mapping
        x = self._compute_complex_covariance_matrix(x)

        x = torch.from_numpy(x).to(dtype=self.dtype)

        # Extract label from filename pattern: {id}_{value}_{label}.npy
        y = int(self._names[item].split(".")[0].split("_")[-1])
        y = torch.from_numpy(np.array(y)).long()

        return x, y

    def _compute_complex_covariance_matrix(self, complex_signal):
        """
        Transform complex signal to real covariance matrix using complex form mapping.

        Args:
            complex_signal: Complex numpy array of shape (1000,)

        Returns:
            Real covariance matrix of shape (40, 40) using complex form mapping
        """
        # Ensure we have a complex signal
        if not np.iscomplexobj(complex_signal):
            raise ValueError("Expected complex signal")

        # Step 1: Window the signal
        signal_length = len(complex_signal)
        num_windows = signal_length // self.window_size  # 50 windows for 1000/20

        # Reshape into windows: (num_windows, window_size)
        windowed_signal = complex_signal[: num_windows * self.window_size].reshape(
            num_windows, self.window_size
        )

        # Step 2: Center each window (subtract mean)
        windowed_centered = windowed_signal - np.mean(
            windowed_signal, axis=1, keepdims=True
        )

        # Step 3: Compute complex covariance matrix
        # C = (1/K) * Σ W_k^H * W_k
        windowed_conj_T = np.conj(windowed_centered).T
        complex_cov = (windowed_conj_T @ windowed_centered) / num_windows

        # Step 4: Extract real and imaginary parts
        A = np.real(complex_cov)  # 20x20 real part
        B = np.imag(complex_cov)  # 20x20 imaginary part

        # Step 5: Apply complex form mapping {C}ℝ = [A -B; B A]
        # Use dtype compatible with the dataset
        target_dtype = np.float32 if self.dtype == torch.float32 else np.float64
        real_cov = np.zeros(
            (2 * self.window_size, 2 * self.window_size), dtype=target_dtype
        )

        # Top-left block: A
        real_cov[: self.window_size, : self.window_size] = A.astype(target_dtype)
        # Top-right block: -B
        real_cov[: self.window_size, self.window_size :] = (-B).astype(target_dtype)
        # Bottom-left block: B
        real_cov[self.window_size :, : self.window_size] = B.astype(target_dtype)
        # Bottom-right block: A
        real_cov[self.window_size :, self.window_size :] = A.astype(target_dtype)

        # Step 6: Ensure positive definiteness
        eigenvals = np.linalg.eigvals(real_cov)
        min_eigenval = np.min(np.real(eigenvals))  # Extract real part
        if min_eigenval <= 0:
            epsilon = 1e-6 - min_eigenval
            real_cov += np.eye(2 * self.window_size, dtype=target_dtype) * epsilon

        return real_cov

    def __str__(self):
        return f"DatasetRadar(path={self._path}, samples={len(self._names)}, classes={self.n_classes}, dtype={self.dtype}, window_size={self.window_size})"

    def __repr__(self):
        return self.__str__()
