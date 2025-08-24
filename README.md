# SLAFCoM: A Study on Loss Functions for Adversarial Finetuning of Contrastive Models
[Aayan Yadav](https://github.com/ydvaayan), [Shree Sighi](https://github.com/ShreeSinghi)

Advisor: [Prof. Sanjeev Kumar](https://scholar.google.co.in/citations?user=FWh8EFkAAAAJ&hl=en)

## Docs
1. Read report at [docs/report.pdf](docs/report.pdf)
2. Find our presentation at [docs/presentation.pptx](docs/presentation.pptx)

## Setup
```bash
pip install -r requirements.txt
```

## Training

```bash
 python apgd_train.py --lr=1e-4 --dataset=imagenet --clean_weight 0.1
```

## References

Schlarmann, C., Singh, N. D., Croce, F., & Hein, M. (2024). Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models. http://arxiv.org/abs/2402.12336