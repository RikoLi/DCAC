# Unleashing the Potential of Pre-Trained Diffusion Models for Generalizable Person Re-Identification
Official implementation code for the paper [Unleashing the Potential of Pre-Trained Diffusion Models for Generalizable Person Re-Identification](todo).

In this work, we propose a novel method called diffusion model-assisted representation learning with a correlation-aware conditioning scheme (DCAC) to enhance DG Re-ID. Our method integrates a discriminative and contrastive Re-ID model with a pre-trained diffusion model through a correlation-aware conditioning scheme. By incorporating ID classification probabilities generated from the Re-ID model with a set of learnable ID-wise prompts, the conditioning scheme injects dark knowledge that captures ID correlations to guide the diffusion process. Simultaneously, feedback from the diffusion model is back-propagated through the conditioning scheme to the Re-ID model, effectively improving the generalization capability of Re-ID features.

## Upload History

- 2025/2/11: Initiate README

## Pipeline

![pipeline](assets/pipeline.png)

## Installation

Create a python environment and install dependencies in `requirements.txt`.

```bash
conda create -n dcac python=3.10
conda activate dcac
pip install -r requirements.txt
```

## Prerequisites

Clone this repository to your own device.

```bash
git clone https://github.com/RikoLi/DCAC.git
cd DCAC
```

### Pre-trained Diffusion Weight

We use `stable-diffusion-v1-5`, which you can find it on [Huggingface](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5). We use the full weight [`v1-5-pruned.ckpt`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt) for fine-tuning. Download the pre-trained weight and put it in a new folder `pretrained`.

```bash
mkdir pretrained && cd pretrained
wget https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt
```

### Configuration

We use [AlchemyCat](https://github.com/HAL-42/AlchemyCat) configuration system to support efficient experiment management. You can directly install it through CLI:

```bash
pip install alchemy-cat
```

Please refer to its official guide for basic configuration settings.

## Acknowledgement

Our implementation mainly refers to [CLIP-ReID](https://github.com/Syliz517/CLIP-ReID) and [stable-diffusion](https://github.com/CompVis/stable-diffusion). Thanks to their amazing works!

## Citation

```bibtex
@article{li2025unleashing,
  title={Unleashing the Potential of Pre-Trained Diffusion Models for Generalizable Person Re-Identification},
  author={Li, Jiachen and Gong, Xiaojin},
  journal={arXiv preprint xxx},
  year={2025},
  doi={10.3390/s25020552}
}
```
