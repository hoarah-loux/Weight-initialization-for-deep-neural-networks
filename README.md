# Deep Learning Weight Initialization Study

A comparative study of weight initialization strategies in deep neural networks. The project examines how initialization affects training stability and classification performance across multi-layer perceptron (MLP) and convolutional neural network (CNN) models on MNIST and CIFAR-10.

The accompanying [`Results-report.pdf`](./Results-report.pdf) describes the theoretical background, experimental setup, ablation studies, and findings.

## What this project compares

- **Weight initialization:** Normal (Gaussian), Uniform, Xavier (Glorot), and Kaiming (He)
- **Architectures:** MLP and CNN
- **Datasets:** MNIST and CIFAR-10
- **Metrics:** training loss, accuracy, macro F1-score, and test loss

The report finds that variance-aware initialization methods generally perform better than naive initialization, with Kaiming initialization performing especially well for ReLU-based networks. See the report for the full results and analysis.

## Project structure

```text
.
├── data/
│   ├── mnist.py            # MNIST download and DataLoader setup
│   └── cifar10.py          # CIFAR-10 download and DataLoader setup
├── init/
│   ├── init_normal.py      # Normal/Gaussian initialization
│   ├── init_uniform.py     # Uniform initialization
│   ├── init_xavier.py      # Xavier/Glorot initialization
│   └── init_kaiming.py     # Kaiming/He initialization
├── models/
│   ├── mlp.py              # Multi-layer perceptron
│   └── cnn.py              # Convolutional neural network
├── main.py                 # Experiment configuration and entry point
├── train.py                # Training loop and training metrics
├── test.py                 # Test-set evaluation
├── requirements.txt        # Python dependencies
├── CMPE460_FinalReport.pdf # Final course report
└── Results-report.pdf      # Detailed results report
```

## Requirements

- Python 3.9 or newer
- PyTorch
- Torchvision

## Setup

From the project directory, create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate it with:

```powershell
.venv\Scripts\activate
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

For GPU support, install the PyTorch build that matches your CUDA setup using the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/).

## Run an experiment

Run the entry point from the project directory:

```bash
python main.py
```

The script selects CUDA when available and otherwise uses the CPU. MNIST or CIFAR-10 is downloaded automatically the first time it is selected.

To choose an experiment, edit the configuration near the top of `main.py`:

```python
dataset_name = "cifar10"  # "mnist" or "cifar10"
model_name = "mlp"        # "mlp" or "cnn"
batch_size = 128
epochs = 10
lr = 0.01

activation = nn.ReLU
init_fn = init_uniform
```

Select an initialization function imported from `init`:

```python
init_normal
init_uniform
init_xavier
init_kaiming
```

Run the program again for each configuration you want to compare. The current entry point runs one dataset/model/initialization combination at a time.

## Output

Training loss, accuracy, and macro F1-score are printed after each epoch. After training, test loss, accuracy, and macro F1-score are printed, and the trained model's state dictionary is saved to `model.pth` in the project directory.
