# Neural Networks from Scratch

A comprehensive, hands-on repository dedicated to building and understanding neural networks, autograd engines, and language models from the absolute ground up. This repository documents the journey of implementing foundational deep learning concepts using only mathematical first principles, Python, NumPy, and PyTorch.

---

## 📁 Repository Structure

```directory
├── Makemore/
│   ├── build_makemore.ipynb      # Character-level language modeling (Bigram, MLP)
│   └── names.txt                 # Dataset containing 32,000 common names
├── Micrograd/
│   └── Micrograd_from _scratch.ipynb # Tiny autograd engine & MLP implementation
├── attachments/                  # Embedded architecture diagrams and visualizations
├── .vscode/                      # Editor workspace snippets and configuration
├── build_neuron.ipynb            # Coding dense layers, activations, and forward passes
└── README.md                     # Project documentation
```

---

## 🚀 Core Modules

### 1. Foundational Neurons & Dense Layers (`build_neuron.ipynb`)
This module focuses on the raw building blocks of neural networks using Python and **NumPy** for vectorization. It implements the forward pass of dense layers from scratch and covers:
- **Neuron Mathematics**: Implementing single neurons with 3 and 4 inputs by calculating dot products ($y = \sum w_i x_i + b$).
- **Layer Vectorization**: Expanding to a complete `Layer_Dense` class that handles batches of inputs using matrix multiplication ($Y = XW^T + B$).
- **Activation Functions**: Implementing the Rectified Linear Activation Function (`ReLU`) from scratch.
- **Spiral Dataset Integration**: Visualizing and fitting non-linear synthetic spiral datasets (generated via the `nnfs` library) to verify layer and activation function behavior.

### 2. Micrograd: Autograd Engine (`Micrograd/`)
A complete implementation of a scalar-valued autograd engine (inspired by Andrej Karpathy's `micrograd`). It builds the foundation of backpropagation:
- **`Value` Class**: Implements forward and backward operations for scalar values, tracking the computational Directed Acyclic Graph (DAG).
- **Mathematical Operations**: Built-in autograd support for addition, multiplication, powers, exponential, and activation functions like `tanh` and `ReLU`.
- **Dynamic Backpropagation**: Automatically computes partial derivatives (gradients) across arbitrary algebraic expressions using the chain rule.
- **Neural Network Library**: Implements `Neuron`, `Layer`, and `MLP` classes purely on top of the scalar `Value` object.
- **Visualization**: Generates beautiful computational graph traces using **Graphviz** to visually debug gradients and node connections.

### 3. Makemore: Autoregressive Language Modeling (`Makemore/`)
An exploration into character-level autoregressive language models using the `names.txt` dataset:
- **Dataset Preprocessing**: Tokenizing names, building character-to-index mappings, and analyzing dataset statistics.
- **Statistical Bigram Model**: Building a probability table (count matrix) of character transitions and generating new names by sampling from the distribution.
- **Neural Network Bigram Model**: 
  - Formulating the bigram model as a single-layer neural network.
  - Using **One-Hot Encoding** for character representation.
  - Designing a forward pass with the **Softmax** activation function to output next-character probability distributions.
  - Optimizing the network parameters by minimizing average Negative Log-Likelihood (NLL) via Stochastic Gradient Descent (SGD).

---

## 🛠️ Environment Setup & Installation

### Prerequisites
Make sure you have Python 3.9+ installed. It is recommended to run these notebooks inside a virtual environment.

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SouvikGhorui/Neural-Networks-from-scratch.git
   cd Neural-Networks-from-scratch
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # On Windows
   source .venv/bin/activate   # On macOS/Linux
   ```

3. **Install Dependencies**:
   ```bash
   pip install numpy torch matplotlib nnfs graphviz
   ```

   *Note: To visualize the autograd DAG in the Micrograd notebook, you will need to install the [Graphviz system binary](https://graphviz.org/download/) and add it to your system path.*

---

## 💡 Acknowledgements & References
This repository is built as an educational resource, drawing inspiration and methodologies from:
- **Neural Networks: Zero to Hero** series by Andrej Karpathy (implementations of Micrograd and Makemore).
- **Neural Networks from Scratch in Python (NNFS)** by Harrison Kinsley and Daniel Kukieła.
