import random
import math
import torch
import numpy as np
from functools import partial
from PIL import Image
from model.stable_diffusion.util import make_beta_schedule, exists, extract_into_tensor

class Identity(object):
    def __init__(self):
        """Identity transform as an augmentation placeholder."""
        pass
    
    def __call__(self, img):
        return img

class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

class MultiGrainedSlice(object):
    def __init__(self, img_size, num_regions):
        self.img_size = img_size
        self.num_regions = num_regions
        assert img_size[0] % self.num_regions == 0, f"Image height={self.img_size[0]} cannot be divided by num_regions={self.num_regions}!"
        
    def __call__(self, img):
        """
        img: PIL.Image
        """
        img = np.asarray(img)
        imgs = np.stack(np.split(img, self.num_regions, axis=0), axis=0) # [num_regions, h, W, C]
        return imgs

class DiffusionNoise(object):
    def __init__(self, cfg):
        """
        Add diffusion-like noise on input images with given noise scheduler.
        It should be applied in preprocessing transform, after `T.Normalize()`.
        """
        self.scheduler = self.register_schedule(cfg)
        self.max_timestep_sampled = cfg.INPUT.MAX_TIMESTEP_SAMPLED
        print('Enable DiffusionNoise augmentation.')
        
    def register_schedule(self, cfg):
        
        timesteps = cfg.INPUT.TIMESTEPS
        beta_schedule = cfg.INPUT.BETA_SCHEDULE
        linear_start = cfg.INPUT.LINEAR_START
        linear_end = cfg.INPUT.LINEAR_END
        cosine_s = cfg.INPUT.COSINE_S
        given_betas = cfg.INPUT.GIVEN_BETAS
        v_posterior = cfg.INPUT.V_POSTERIOR
        
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                    cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        setattr(self, 'betas', to_torch(betas))
        setattr(self, 'alphas_cumprod', to_torch(alphas_cumprod))
        setattr(self, 'alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        setattr(self, 'sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        setattr(self, 'sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        setattr(self, 'log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        setattr(self, 'sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        setattr(self, 'sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        setattr(self, 'posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        setattr(self, 'posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        setattr(self, 'posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        setattr(self, 'posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        
    def __call__(self, img):
        """
        img: [B, C, H, W] torch.Tensor produced by `T.Normalize()`. Value range [-1, 1]
        """
        assert self.max_timestep_sampled <= self.num_timesteps
        ts = min(self.max_timestep_sampled, self.num_timesteps)
        t = torch.randint(0, ts, (img.shape[0],)).long()
        noise = torch.randn_like(img)
        img_noisy = (extract_into_tensor(self.sqrt_alphas_cumprod, t, img.shape) * img +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, img.shape) * noise)
        img_noisy = torch.clip(img_noisy, min=-1.0, max=1.0) # cutoff to keep value in [-1, 1]
        return img_noisy