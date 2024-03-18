# detraffic

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

1. [Installation](#Installation)
2. [Development](#Development)
3. [License](#license)

## Installation

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
