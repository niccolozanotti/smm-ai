# Statistical and Mathematical Methods for AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)


This repository contains implementations of the Homework Assignments (see [here](assignments/)) for the [course](https://www.unibo.it/it/studiare/dottorati-master-specializzazioni-e-altra-formazione/insegnamenti/insegnamento/2024/446599)
“Statistical and Mathematical Methods for AI” ('24/'25) at [unibo](https://www.unibo.it/en).

The class material can be found [here](https://devangelista2.github.io/statistical-mathematical-methods/intro.html).

## Project Structure

```
smmAI/
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── README.md
├── .gitignore
├── ruff.toml
├── .pre-commit-config.yaml
├── assignments/                  
│   ├── hw1.md
│   ├── hw2.md
│   ├── hw3.md
│   └── hw4.md
├── apps/                  
│   ├── hw1_linear_systems.py
│   ├── hw2_svd_pca.py
│   ├── hw3_optimization.py
│   └── hw4_mle_map.py
└── notebooks/                    
    ├── hw1.py
    ├── hw2.py
    ├── hw3.py
    └── hw4.py
```

## Installation

You can install this project using either uv (recommended) or pip.

### Option 1: Using uv (Recommended)

1. First, install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and activate a new virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Unix-like systems
# OR
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
# Install main dependencies
uv pip install .

# Install with development dependencies (for contributing)
uv pip install ".[dev]"
```

### Option 2: Using pip

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Unix-like systems
# OR
.venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
# Install main dependencies
pip install -r requirements.txt

# Install development dependencies (for contributing)
pip install -r requirements-dev.txt
```

## Running the Notebooks

After installation, you can run the marimo notebooks:

```bash
marimo edit notebooks/hw1.py
```

## License

This Project is licensed under the [MIT License](LICENSE).




