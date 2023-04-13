import torch
import torchvision
from torchvision import transforms
from torch.utils.data import ConcatDataset

def transform_dataset(img_size, path):
    data_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scale between [0, 1] 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ])

    train = torchvision.datasets.StanfordCars(root=path, download=True, transform=data_transforms)
    test = torchvision.datasets.StanfordCars(root=path, download=True, transform=data_transforms, split='test')
                                         
    return ConcatDataset([train, test])

def load_model(model, weights_path):
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint)
    return model