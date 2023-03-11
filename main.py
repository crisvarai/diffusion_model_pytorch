"""To train a diffusion model."""

import torch
from torch.utils.data import DataLoader

from train.fit import fit
from model.unet import Unet
from utils.load_args import get_args
from utils.data import transform_dataset

if __name__ == "__main__":
    args = get_args()
    DATASET_PATH = args.data_path
    IMG_SIZE = args.imgsize
    BATCH_SIZE = args.batchsize
    EPOCHS = args.epochs
    LR = args.lr
    T = args.timesteps
    WEIGHTS_PATH = args.wgts_path
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = transform_dataset(img_size=IMG_SIZE, path=DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    model = Unet()

    fit(model=model, 
        dataloader=dataloader, 
        img_size=IMG_SIZE, 
        batch_size=BATCH_SIZE, 
        epochs=EPOCHS, 
        lr=LR, 
        T=T,
        weights_path=WEIGHTS_PATH, 
        device=DEVICE)