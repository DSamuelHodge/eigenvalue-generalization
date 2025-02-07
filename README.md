# Eigenvalue Ratios and Generalization in LLMs

This project investigates the relationship between eigenvalue ratios and model generalization in Large Language Models, based on the paper:

**"Enhancing Parameter-efficient Fine-tuning with Simple Calibration based on Stable Rank"** (LREC-COLING 2024)
by Peiyu Liu, Ze-Feng Gao, Xiao Zhang, Wayne Xin Zhao, Ji-Rong Wen.

## Paper Abstract

Lightweight fine-tuning is widely used as an important technique for efficiently adapting pre-trained language models (PLM) to downstream tasks. Despite the reduction in trainable parameters, existing lightweight fine-tuning methods are found to be effective in low-resource settings but often fail in high-resource settings, leading to unreliable outcomes. This limitation can be attributed to inflexible strategies: they identify the parameters of the model to be trained before fine-tuning and remain unchanged without taking into account the inherent variance of generalization ability in model components (i.e., feed-forward, attention layers) and potential changes during the fine-tuning process.

## Methodology

### 1. Stable Rank Analysis
We investigate the paper's key theoretical claim about stable rank:
```
srank(W) = ||W||²_F / ||W||²_2 = (Σⱼ₌₁ᵏ σⱼ²(W)) / σ₁²(W)
```
where:
- W is a weight matrix
- ||W||_F is the Frobenius norm
- ||W||_2 is the spectral norm
- σⱼ(W) are singular values

### 2. Experimental Setup
Our investigation consists of three main components:

a) Natural Training Analysis:
   - Track eigenvalue ratios during normal model training
   - Monitor correlation with generalization metrics
   - Compare FFT vs LFT trajectories

b) Controlled Experiments:
   - Artificially modify eigenvalue ratios
   - Keep other factors constant
   - Measure impact on generalization

c) Layer-wise Analysis:
   - Examine differences between attention and FFN layers
   - Track layer-specific eigenvalue evolution
   - Study impact of layer position

### 3. Metrics
We track multiple metrics to evaluate the claims:

- Eigenvalue Metrics:
  * Stable rank
  * Top eigenvalue ratio
  * Eigenvalue distribution shape

- Performance Metrics:
  * Validation accuracy
  * Generalization gap
  * Loss landscape smoothness

- Efficiency Metrics:
  * Memory usage
  * Computational cost
  * Training time

## Expected Findings

### 1. Paper Claims to Validate
The paper makes several key claims we aim to verify:

a) Stable Rank and Generalization:
   - Lower stable rank → Better generalization
   - LFT achieves lower stable rank than FFT
   - Optimal ratio exists for each layer type

b) Layer-wise Behavior:
   - Different components (attention vs FFN) have different optimal ratios
   - Layer position affects optimal ratio
   - Some layers are more important for generalization

c) Training Dynamics:
   - FFT randomly modifies all eigenvalues
   - LFT selectively modifies important ones
   - Early training phase crucial for ratio development

### 2. Additional Investigations

Beyond the paper's claims, we investigate:

a) Causality Analysis:
   - Does changing eigenvalue ratios cause better generalization?
   - Are there confounding factors?
   - What's the role of implicit regularization?

b) Alternative Explanations:
   - Impact of reduced parameter space
   - Role of optimization dynamics
   - Influence of task complexity

c) Practical Implications:
   - Optimal switching point from FFT to LFT
   - Task-specific considerations
   - Scaling behavior with model size

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

## Setup and Requirements

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Required compute:
- GPU with at least 16GB VRAM for base models
- 40GB+ VRAM for large model experiments
- Sufficient CPU RAM for data processing (16GB+)

## Usage

1. Run experiments:
```bash
python src/experiments/run_experiment.py
```

2. Analyze results:
```bash
python src/analyze_results.py
```

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