import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from torchvision.utils import save_image
from diffusion.sampling import sample_timestep

def show_tensor_img(image, i):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    # plt.imshow(reverse_transforms(image))
    save_image(image, f"samples/sample_ex{i}.png")

@torch.no_grad()
def sample_plot_image(img_size, T, model, device, betas, sqrt_recip_alphas, sqrt_one_minus_alphas_cumprod, posterior_variance):
    # Sample noise
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_imgs = 10
    stepsize = int(T/num_imgs)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t, model, betas, sqrt_recip_alphas, sqrt_one_minus_alphas_cumprod, posterior_variance)
        if i % stepsize == 0:
            # plt.subplot(1, num_imgs, int(i/stepsize+1))
            show_tensor_img(img.detach().cpu(), i)
    # plt.show()     