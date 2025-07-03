# MIDST Toolkit

----------------------------------------------------------------------------------------

[![code checks](https://github.com/VectorInstitute/midst-toolkit/actions/workflows/code_checks.yml/badge.svg)](https://github.com/VectorInstitute/midst-toolkit/actions/workflows/code_checks.yml)
[![integration tests](https://github.com/VectorInstitute/midst-toolkit/actions/workflows/integration_tests.yml/badge.svg)](https://github.com/VectorInstitute/midst-toolkit/actions/workflows/integration_tests.yml)
[![docs](https://github.com/VectorInstitute/midst-toolkit/actions/workflows/docs.yml/badge.svg)](https://github.com/VectorInstitute/midst-toolkit/actions/workflows/docs.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A toolkit for facilitating MIA resiliency testing on diffusion-model-based synthetic tabular data. Many of the attacks
included in this toolkit are based on the most successful ones used in the
[2025 SaTML MIDST Competition](https://vectorinstitute.github.io/MIDST/).

## 🧑🏿‍💻 Developing

### Installing dependencies

The development environment can be set up using [uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation).
Hence, make sure it is installed and then run the following:

```bash
uv sync
source .venv/bin/activate
```

In order to install dependencies for testing (codestyle, unit tests, integration tests),
run:

```bash
uv sync --dev
source .venv/bin/activate
```

In order to exclude installation of packages from a specific group (e.g. docs),
run:

```bash
uv sync --no-group docs
```
