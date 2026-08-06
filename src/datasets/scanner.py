"""Dataset scanner module for SEM image restoration.

This module provides `DatasetScanner` to index and pair Ground Truth (GT) and Noisy
Low-Resolution (NoisyLR) `.npy` image files while filtering out hidden metadata
files (such as macOS `__MACOSX` and `._*` resource forks).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union


@dataclass(frozen=True)
class DatasetPair:
    """Dataclass holding file path pairings for a dataset sample.

    Attributes:
        sample_id: Unique string identifier extracted from filename.
        input_path: Path to NoisyLR array file.
        target_path: Path to Ground Truth (GT) array file, if available.
    """

    sample_id: str
    input_path: Path
    target_path: Optional[Path] = None


class DatasetScanner:
    """Directory scanner for indexing paired SEM restoration dataset files."""

    def __init__(self, root_dir: Union[str, Path]) -> None:
        """Initialize DatasetScanner with dataset root directory.

        Args:
            root_dir: Path to root dataset directory.

        Raises:
            FileNotFoundError: If root_dir does not exist on disk.
        """
        self.root_dir = Path(root_dir).resolve()
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset root directory not found: {self.root_dir}"
            )

    @staticmethod
    def _is_valid_npy_file(path: Path) -> bool:
        """Check if file is a valid .npy array file and not hidden metadata.

        Args:
            path: File path to evaluate.

        Returns:
            bool: True if path is a valid non-hidden .npy file.
        """
        if not path.is_file():
            return False
        if not path.name.lower().endswith(".npy"):
            return False
        if path.name.startswith("._") or path.name.startswith("."):
            return False
        if "__MACOSX" in path.parts:
            return False
        return True

    def _get_valid_files(self, directory: Path) -> List[Path]:
        """Collect all valid .npy files within a directory sorted by name.

        Args:
            directory: Directory to search.

        Returns:
            List[Path]: Sorted list of valid file paths.

        Raises:
            FileNotFoundError: If target directory does not exist.
        """
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        files = [f for f in directory.iterdir() if self._is_valid_npy_file(f)]
        files.sort(key=lambda p: p.name)
        return files

    def scan_split(self, split: str = "train") -> List[DatasetPair]:
        """Scan a dataset split ('train' or 'test') and build paired file index.

        Args:
            split: Name of dataset split ('train' or 'test').

        Returns:
            List[DatasetPair]: List of paired dataset samples.

        Raises:
            ValueError: If split is unsupported or paired files are mismatched.
            FileNotFoundError: If split directory does not exist.
        """
        split_dir = self.root_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

        if split == "train":
            gt_dir = split_dir / "GT"
            noisy_dir = split_dir / "NoisyLR"

            gt_files = self._get_valid_files(gt_dir)
            noisy_files = self._get_valid_files(noisy_dir)

            if len(gt_files) != len(noisy_files):
                raise ValueError(
                    f"Mismatched dataset counts in '{split}': "
                    f"{len(gt_files)} GT files vs {len(noisy_files)} NoisyLR files."
                )

            gt_map: Dict[str, Path] = {f.name: f for f in gt_files}
            pairs: List[DatasetPair] = []

            for noisy_file in noisy_files:
                if noisy_file.name not in gt_map:
                    raise ValueError(
                        f"Unpaired NoisyLR file '{noisy_file.name}' has no matching GT file."
                    )
                sample_id = noisy_file.stem
                pairs.append(
                    DatasetPair(
                        sample_id=sample_id,
                        input_path=noisy_file,
                        target_path=gt_map[noisy_file.name],
                    )
                )

            return pairs

        elif split == "test":
            noisy_dir = split_dir / "NoisyLR"
            if not noisy_dir.exists():
                noisy_dir = split_dir

            noisy_files = self._get_valid_files(noisy_dir)
            return [
                DatasetPair(
                    sample_id=f.stem,
                    input_path=f,
                    target_path=None,
                )
                for f in noisy_files
            ]

        else:
            raise ValueError(
                f"Unsupported dataset split '{split}'. Expected 'train' or 'test'."
            )
