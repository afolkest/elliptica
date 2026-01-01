![Electrogenesis](examples/Folkestad_electrogeneis_elliptica.jpg)

# Elliptica

Elliptica is a modular framework for making 2d visual arts using laws of physics,
specifically elliptic boundary value problems.  You place objects on a canvas,
impose boundary conditions on these, and then solve a partial differential
equation system (like the Poisson equation for electric fields), and visualize
the resulting vector fields using high-quality Line Integral Convolution (LIC).

Making good art requires rapid iteration, so the software is made with
this in mind, trying to let you focus on the aesthetic aspects. That said, it is
computationally heavy to run some of these algorithms.  At 1024 x 1024, expect
to have to wait a few seconds per render.

This software is not about trying to give maximally scientifically accurate
visualizations of physics, although accurate physics is a side effect. This
software is all about making beautiful things. The laws of physics happen to be
extremely helpful.

The above image is created in Elliptica. So is this one:

<img src="examples/Folkestad_portrait_of_you_and_me_elliptica.jpg" width="600" alt="Portrait of You and Me">

## This is an alpha version
This is an alpha version, and I am actively working on this software. Expect bugs,
and frequent non-backwards compatible changes. I have so far only tested the software
on my own M1 Silicon Mac, although I'm aiming to make this run on other systems.

The UI is janky. I will improve it. 

Note: this project is heavily LLM coded (thanks Claude).  

## Quick Start

```bash
pip install -r requirements.txt
python -m elliptica.ui.dpg
```

## System Requirements

- **Python 3.11+**
- **OS**: macOS, Linux (X11/Wayland), Windows 10/11
- **GPU** (optional): Apple Silicon (MPS) or NVIDIA (CUDA) - falls back to CPU (not recommended)
- **Display**: GUI app, no headless support

## Installation

```bash
git clone https://github.com/your-username/elliptica.git
cd elliptica
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m elliptica.ui.dpg
```

## License

AGPL-3.0. See [LICENSE](LICENSE).
