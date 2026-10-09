"""
Build the training dataset from the outputs of the (MATLAB) solver.

Planned workflow:
1. In MATLAB, sweep the chosen parameter space (e.g. source amplitude, bubble
   number density, medium properties) and export each simulation to data/raw/
   (.mat or .csv format, one file per simulation or a single file with all of
   them).
2. This script reads those files, extracts the inputs (parameters) and the
   outputs (the v1 target quantity), and builds a tabular dataset ready for
   PyTorch.
3. It saves the result in data/processed/ (e.g. as .npz or .csv), already
   split into train/val/test.

This is currently a skeleton: the exact export format from MATLAB and the
target quantity have to be decided before the real logic can be written.
"""

from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def load_raw_simulations(raw_dir: Path = RAW_DIR):
    """Read all the simulations exported from MATLAB in raw_dir.

    TODO: implement once the export format is decided
    (.mat via scipy.io.loadmat, or .csv via pandas).
    """
    raise NotImplementedError("Pending: define the MATLAB export format")


def build_dataset(simulations):
    """Convert the list of simulations into input/output arrays ready for
    training (numpy arrays or torch tensors).
    """
    raise NotImplementedError


def save_dataset(inputs, outputs, out_dir: Path = PROCESSED_DIR):
    """Save the processed dataset (and make the train/val/test split)."""
    raise NotImplementedError


if __name__ == "__main__":
    sims = load_raw_simulations()
    X, y = build_dataset(sims)
    save_dataset(X, y)
