<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/DeTraffic/detraffic/blob/main/assets/icon-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/DeTraffic/detraffic/blob/main/assets/icon-light.svg">
    <img alt="DeTraffic logo" src="https://github.com/DeTraffic/detraffic/blob/main/assets/icon-teal.svg" width=70>
  </picture>
</div>
<h1 align="center">
  DeTraffic
</h1>

DeTraffic is a multi-agent deep reinforcement learning model to de-traffic our lives.

## Table Of Contents

1. [Installation](#installation)
2. [Development](#development)
3. [Running experiments](#running_experiments)
4. [Acknowledgement](#acknowledgement)
3. [License](#license)

## Installation

### SUMO
You have to install SUMO beforehand.

```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```
Don't forget to set SUMO_HOME variable (default sumo installation path is /usr/share/sumo)
```bash
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc
```
Important: for a huge performance boost (~8x) with Libsumo, you can declare the variable:
```bash
export LIBSUMO_AS_TRACI=1
```
Notice that you will not be able to run with sumo-gui or with multiple simulations in parallel if this is active ([more details](https://sumo.dlr.de/docs/Libsumo.html)).

### DeTraffic
If you do not have `poetry` installed:

```bash
pip install poetry
```

For testing:

```bash
poetry install --without-dev # for newer versions of poetry
poetry install --no-dev # for older versions of poetry
```

For development purposes:

```bash
poetry install
```

And then you can dive into the environment with:

```bash
poetry shell
```

## Development

After the installation, there are several steps to follow for development.

### pdoc

### pre-commit

```bash
pre-commit install
```

### pytest

### mypy

### ruff

### refurb

## Running experiments

You can check predefined experiments at `experiments` and models at `models`, or define your own experiments or models.

```bash
poetry shell
python detraffic/benchmark.py
```

## Acknowledgement

This repository contains code from [PyTorch Reinforcement Learning (DQN) Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) and [efficient-kan](https://github.com/Blealtan/efficient-kan/). Also containts SUMO installation steps from [sumo-rl](https://github.com/LucasAlegre/sumo-rl/).