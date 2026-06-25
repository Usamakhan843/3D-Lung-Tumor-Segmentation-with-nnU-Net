"""nnU-Net v2 / MSD configuration. Edit the paths to your WSL environment."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Where the extracted MSD task lives (the folder with imagesTr/, labelsTr/, dataset.json)
MSD_EXTRACTED = Path("/home/usama/msd/Task06_Lung")

# nnU-Net's three working directories (created automatically if missing)
NNUNET_BASE         = Path("/home/usama/nnunet")
NNUNET_RAW          = NNUNET_BASE / "nnUNet_raw"
NNUNET_PREPROCESSED = NNUNET_BASE / "nnUNet_preprocessed"
NNUNET_RESULTS      = NNUNET_BASE / "nnUNet_results"

# Dataset / training settings
DATASET_ID   = 6                 # MSD lung -> Dataset006_Lung
DATASET_NAME = "Dataset006_Lung"
CONFIG       = "3d_lowres"       # lighter fallbacks if VRAM is tight: "3d_lowres", "2d"
FOLD         = 0                 # single fold for a first result; run 0..4 for full CV

# Optional faster trainer. None = nnU-Net's full 1000-epoch default.
# On an 8 GB RTX 5050 the full run takes a very long time, so for a demonstration
# result set this to a reduced trainer, e.g.:
#   TRAINER = "nnUNetTrainer_250epochs"   (or "nnUNetTrainer_100epochs" for quicker)
TRAINER = "nnUNetTrainer_250epochs"

# Preprocessing worker processes. Lower uses less RAM, which helps avoid the WSL
# VM running out of memory and dropping the VS Code connection. Raise it later if
# your machine has RAM to spare.
NUM_PROC = 2
