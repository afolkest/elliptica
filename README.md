# Elliptica

![Electrogenesis](examples/Folkestad_electrogeneis_elliptica.jpg)

![Elliptica UI](examples/ui_example.png)

## What is this?

Elliptica is a program for making 2d visual arts using electrostatics (the Laplace equation).  You place objects on a canvas, impose voltage boundary conditions on them, and then solve a partial differential equation system, and visualize the resulting vector fields using a technique known as Line Integral Convolution (LIC).

The software contains various useful things that lets you iterate on your pieces without leaving the software, such as OKLCH color palette creation and postprocessing effects.

## Not actively developed, buggy

This was my first attempt at a larger project 100% coded by LLMs and it shows.

I'm doing a complete low level rewrite of this software in Metal, since the framework choices made here (DearPyGui + PyTorch) are fundamentally incompatible with real time interactivity, given the 
algorithms we work with here. I will likely not be adding any more features in this prototype, although I will happily fix reported bugs (of which there are many). 

## Quick Start

```bash
pip install -r requirements.txt
python -m elliptica.ui.dpg
```

## System Requirements

- **Python 3.11+**
- **GPU** (optional): Apple Silicon (MPS) or NVIDIA (CUDA) - falls back to CPU (probably quite slow)

## Installation

```bash
git clone https://github.com/afolkest/elliptica.git
cd elliptica
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m elliptica.ui.dpg
```

## License

AGPL-3.0. See [LICENSE](LICENSE).
