# Minitron

<p align="center">
<img src="https://www.sauravm.com/assets/img/minitron.png"  width="256">
</p>
<p align="center">
        🤗 <a href="">Hugging Face Models</a>&nbsp&nbsp | &nbsp&nbsp 📄 <a href="">Paper</a> &nbsp&nbsp | &nbsp&nbsp 📜 <a href="">Blog</a> &nbsp | &nbsp 💬 <a href="">Demo</a>
</p>


## Introduction

<div style="text-align: center;">
  <img src="images/minitron.png" alt="Sample Image" width="300"/>
  <p>Figure 1: Results for Minitron. Compression results in significant reduction of training costs for additional models(40x) while producing better results.</p>
</div>

Minitron is a family of small language models (SLMs) obtained by pruning NVIDIA's [Nemotron-4 15B]() model. We prune model embedding size, attention heads, and MLP intermediate dimension, following which, we perform continued training with distillation to arrive at the final models.

Deriving the Minitron 8B and 4B models from the base 15B model using our approach requires up to **40x fewer training tokens** per model compared to training from scratch; this results in **compute cost savings of 1.8x** for training the full model family (15B, 8B, and 4B). Minitron models exhibit up to a 16% improvement in MMLU scores compared to training from scratch, perform comparably to other community models such as Mistral 7B, Gemma 7B and Llama-3 8B, and outperform state-of-the-art compression techniques from the literature. Please refer to our [arXiv paper]() for more details.

Minitron models are for research and development only.

## Demonstration of Various Pruning Strategies

| DEP | MLP | ATT | EMB | Distillation Loss | LM Val Loss |
|-----|-----|-----|-----|-------------------|-------------|
| ✔️  | ✔️  | ✔️  | ✔️  | 5.35 → 0.38       | 2.062       |
| ❌  | ✔️  | ✔️  | ✔️  | **6.33 → 0.37**   | **2.049**   |
| ❌  | ✔️  | ✔️  | ❌  | 5.07 → 0.42       | 2.101       |
| ✔️  | ❌  | ❌  | ❌  | 8.35 → 0.49       | 2.155       |
| Train from scratch (random init) |  |  |  | 12.27 → 2.34 | 3.953       |

*Table 1: Demonstration of how various pruning strategies perform before and after lightweight retraining using ~1.8B tokens. We prune the Nemotron-4 15B model down to the size of Nemotron-3 8B and report the change in distillation loss (KL divergence on logits) and the final LM validation loss with retraining. We see that width (attention, MLP, embedding) pruning outperforms depth, but only after retraining. The last row shows change in loss for the Nemotron-3 8B model.*

## Quickstart

## Model Card
Please see [MODEL_CARD.md](MODEL_CARD.md).

## License

Minitron is released under the [NVIDIA Open Model License Agreement](https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf).

## Citation

If you find our work helpful, please consider citing our paper:
```
@article{minitron2024,
      title={Compact Language Models via Pruning and Knowledge Distillation}, 
      author={Saurav Muralidharan and Sharath Turuvekere Sreenivas and Raviraj Joshi and Marcin Chochowski and Mostofa Patwary and Mohammad Shoeybi and Bryan Catanzaro and Jan Kautz and Pavlo Molchanov},
      journal={arXiv preprint arXiv:XXX},
      year={2024}
}
```
