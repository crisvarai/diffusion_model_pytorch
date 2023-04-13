"""To train a diffusion model."""

import torch
import logging
from torch.utils.data import DataLoader

from train.fit import fit
from model.unet import Unet
from utils.load_args import get_args
from utils.data import transform_dataset, load_model

logging.basicConfig(
    filename="runing.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
)

if __name__ == "__main__":
    args = get_args()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = transform_dataset(img_size=args.imgsize, path=args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True, drop_last=True)
    model = Unet()

    logging.info("Load last model weights")
    model = load_model(model, args.wgts_path).to("cpu")

    logging.info("Start training...")
    fit(model=model, 
        dataloader=dataloader, 
        img_size=args.imgsize, 
        batch_size=args.batchsize, 
        epochs=args.epochs, 
        lr=args.lr, 
        T=args.timesteps,
        weights_path=args.wgts_path, 
        device=DEVICE)
    logging.info("Finished!")