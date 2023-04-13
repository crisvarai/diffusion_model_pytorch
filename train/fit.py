import torch
import logging
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from utils.visualize import sample_plot_image
from diffusion.forward import pre_calc, forward_diffusion_sample

def get_loss(model, x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device):
    x_noisy, noise = forward_diffusion_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device=device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)

def fit(model, dataloader, img_size, batch_size, epochs, lr, T, weights_path, device):
    
    # Pre-calculations
    betas, sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance = pre_calc(T)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []

    for epoch in range(epochs):
        train_loss = []
        model.train()
        bar = tqdm(dataloader)
        for step, batch in enumerate(bar):
            optimizer.zero_grad()
            t = torch.randint(0, T, (batch_size,), device=device).long()
            loss = get_loss(model, batch[0], t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device=device)
            train_loss.append(loss.item())
            mean_tl = np.mean(train_loss)
            loss.backward()
            optimizer.step()
            bar.set_description(f"Loss {mean_tl}")

        if epoch % 5 == 0 and epoch != 0:
            logging.info(f"Epoch {epoch} | step {step:03d} Loss: {mean_tl}")
            torch.save(model.state_dict(), weights_path)
            logging.info("WEIGHTS-ARE-SAVED")
            logging.info("Generate images...")
            sample_plot_image(img_size, T, model, device, \
                              betas, sqrt_recip_alphas, sqrt_one_minus_alphas_cumprod, posterior_variance)
            logging.info("Saved!")
        train_losses.append(mean_tl)
    return train_losses