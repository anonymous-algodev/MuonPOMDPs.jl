# MuonPOMDPs.jl

Muon tomography modeled as a POMDP using POMDPs.jl

Uses the _inversion variational autoencoder_ (I-VAE) developed for this project ([anonymous-algodev/I-VAE](https://github.com/anonymous-algodev/I-VAE))

See the [`notebooks/i-vae.ipynb`](./notebooks/i-vae.ipynb) notebook for usage.

Drill Locations | Cross Sections
:---------------:|:----:
<kbd> <img src="./img/muon-tomography-drilling.png"> </kbd> | <kbd> <img src="./img/muon-tomography-sections.png"> </kbd>

Belief Updating |
:---------------:|
<kbd> <img src="./img/plot_true_intrusion.png"> </kbd>
<kbd> <img src="./img/plot_final_belief_mean.png"> </kbd>
<kbd> <img src="./img/plot_final_belief_std.png"> </kbd>

Radiography |
:---------------:|
<kbd> <img src="./img/muon-tomography-radiography.png"> </kbd>


## Setup

Install Python dependencies:

```bash
pip install -r python/requirements.txt
``
