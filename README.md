# Eigenvalue Ratios and Generalization in LLMs

This project investigates the relationship between eigenvalue ratios and model generalization in Large Language Models, based on the paper:

**"Enhancing Parameter-efficient Fine-tuning with Simple Calibration based on Stable Rank"** (LREC-COLING 2024)
by Peiyu Liu, Ze-Feng Gao, Xiao Zhang, Wayne Xin Zhao, Ji-Rong Wen.

## Paper Abstract

Lightweight fine-tuning is widely used as an important technique for efficiently adapting pre-trained language models (PLM) to downstream tasks. Despite the reduction in trainable parameters, existing lightweight fine-tuning methods are found to be effective in low-resource settings but often fail in high-resource settings, leading to unreliable outcomes. This limitation can be attributed to inflexible strategies: they identify the parameters of the model to be trained before fine-tuning and remain unchanged without taking into account the inherent variance of generalization ability in model components (i.e., feed-forward, attention layers) and potential changes during the fine-tuning process.

This repository contains code to investigate their claims about the relationship between eigenvalue ratios and generalization performance.

## Project Structure

```
./
├── data/           # Data directory (gitignored)
├── src/            # Source code
│   ├── analysis/   # Analysis utilities
│   ├── models/     # Model definitions
│   └── utils/      # Utility functions
├── notebooks/      # Jupyter notebooks
├── tests/          # Unit tests
├── configs/        # Configuration files
└── results/        # Experimental results (gitignored)
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. Run experiments:
```bash
python src/experiments/run_experiment.py
```

2. Analyze results:
```bash
python src/analyze_results.py
```

## Experiments

This repository contains code for two main experimental tracks:

1. Correlation Analysis: Track eigenvalue ratios during natural training
2. Controlled Intervention: Artificially modify eigenvalue ratios

The experiments aim to validate or challenge the paper's claims about the relationship between eigenvalue ratios and model generalization.

## Citation

```bibtex
@inproceedings{liu2024enhancing,
  title={Enhancing Parameter-efficient Fine-tuning with Simple Calibration based on Stable Rank},
  author={Liu, Peiyu and Gao, Ze-Feng and Zhang, Xiao and Zhao, Wayne Xin and Wen, Ji-Rong},
  booktitle={Proceedings of LREC-COLING 2024},
  pages={6024--6035},
  year={2024}
}
```

## License

MIT