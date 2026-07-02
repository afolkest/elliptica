# Elliptica

![Electrogenesis](examples/Folkestad_electrogeneis_elliptica.jpg)

![Elliptica UI](examples/ui_example.png)

## What is this?

Elliptica is a program for making 2d visual arts using electrostatics (the Laplace equation).  You place objects on a canvas, impose voltage boundary conditions on them, solve a partial differential equation, and then visualize the resulting vector fields using a technique known as Line Integral Convolution (LIC).

## This is a prototype: superseded, buggy, but fun

This was my first attempt at a larger project 100% coded by LLMs and if you look at the code it shows.

I've moved on to doing a complete low level rewrite in Metal, since the framework choices made here (DearPyGui + PyTorch) are fundamentally incompatible with real time interactivity (given the 
algorithms we're working with here). I will not be adding any more features in the prototype, although I'm open to fix reported bugs. 

## Quick Start

```bash
pip install -r requirements.txt
python -m elliptica.ui.dpg
```
