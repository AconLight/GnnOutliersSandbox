# Import of libraries
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.dataloading.dataloader_gen import PerDirectoryImgClassificationDataset
from src.utils.showimg.show_images import show_image

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == "__main__":
    # Setting reproducibility
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Definitions
    STORE_PATH_MALE = f"ddpm_model_male.pt"
    STORE_PATH_FEMALE = f"ddpm_model_female.pt"

    no_train = False
    is_male = True
    batch_size = 128
    n_epochs = 20
    lr = 0.001
    store_path = "ddpm_male.pt" if is_male else "ddpm_female.pt"

    female_path = Path(__file__).parents[1] / "datasets" / "malefemale" / "female"
    male_path = Path(__file__).parents[1] / "datasets" / "malefemale" / "male"
    dataset = PerDirectoryImgClassificationDataset([female_path, male_path], (-1, 0), (64, 64))
    loader = DataLoader(dataset, batch_size, shuffle=False)

    show_image(loader.dataset.__getitem__(11)[0])