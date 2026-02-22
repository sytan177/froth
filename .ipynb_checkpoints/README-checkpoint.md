# froth: superbubble identification with graph neural networks

Identify supernova driven bubbles or just underdense regions for simulation snapshots or astronomical galaxy images using trained graph neural networks and either gas data or image intensity as input features. Currently tested on high-resolution AREPO galaxy simulations with a spatial resolution of 10 pc. 20 pc and 100 pc and JWST galaxy images in F770W and F1130W bands. 

This package is implements the method described in the paper:
paper citation here ...

---

## Currently Available Models
- 
- aerate_3D for simulation snapshot, density and temperature aware
- aerate_3D_den for simulation snapshot, density aware only
- aerate_3D_hot for simulation snapshot, density and temperature aware, hard negatives for temperature below 10e5.5K

---

## Installation

pip install froth_gnn

Or you can install from source:

```bash
git clone https://github.com/sytan177/froth.git
cd froth
pip install -e .

### Requirements
- Python 3.9+

### Core Dependencies
- numpy
- torch
- torch-geometric
- scikit-learn
- h5py
- numba
- joblib

### Optional
for galaxy image preprocessing
- photutils
