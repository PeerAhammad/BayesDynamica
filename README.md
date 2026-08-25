# BayesDynamica

> A Python research codebase for Bayesian inference and filtering in linear, nonlinear, and switching state-space models.

## Overview

**BayesDynamica** is a research-oriented implementation of probabilistic inference methods for dynamical systems.

The repository brings together several classical and modern state-space inference techniques, with separate modules for **models** and **inference algorithms**.

The current codebase includes implementations for:

* Kalman filtering
* Kalman smoothing
* Extended Kalman Filtering (EKF)
* Unscented Kalman Filtering (UKF)
* Expectation-Maximization (EM)
* Switching Kalman filtering
* Variational switching inference
* Linear state-space models
* Nonlinear state-space models
* Switching state-space models

---

## Architecture

The project is organized into two main components:

```text
BayesDynamica/
│
├── src/
│   ├── models/
│   │   ├── linear_ssm.py
│   │   ├── nonlinear_ssm.py
│   │   └── switching_ssm.py
│   │
│   └── inference/
│       ├── kalman_filter.py
│       ├── kalman_smoother.py
│       ├── ekf.py
│       ├── ukf.py
│       ├── em_algorithm.py
│       ├── switching_kalman_filter.py
│       └── variational_switching_kt.py
│
└── .gitignore
```

### Models

The `models/` package contains state-space model definitions:

* **Linear SSM** — linear state-space dynamics
* **Nonlinear SSM** — nonlinear state-space dynamics
* **Switching SSM** — dynamical systems with switching regimes

### Inference

The `inference/` package contains algorithms for estimating hidden states and model parameters:

* **Kalman Filter**
* **Kalman Smoother**
* **Extended Kalman Filter**
* **Unscented Kalman Filter**
* **EM Algorithm**
* **Switching Kalman Filter**
* **Variational Switching Inference**

---

## Research Direction

BayesDynamica provides a modular foundation for studying inference in dynamical systems, particularly where the underlying process may be:

* Linear
* Nonlinear
* Regime-switching
* Partially observed

The modular separation between **state-space models** and **inference methods** is intended to make it easier to experiment with different combinations of models and inference algorithms.

---

## Project Status

🚧 **Research / Development**

The repository is actively being developed as a research codebase. Additional models, inference methods, experiments, and validation are expected to evolve over time.

---

## Technologies

* Python
* Probabilistic modeling
* State-space methods
* Bayesian inference
* Numerical methods
* Scientific computing

---

## Related Research

This repository is part of my broader research interests in:

**Bayesian inference · Dynamical systems · Probabilistic modeling · State-space models · Machine learning**

### Research Profiles

* [ORCID](https://orcid.org/0009-0008-6686-9223)
* [OpenReview](https://openreview.net/profile?id=%7EPeerAhammad_M_Bagawan3)
* [GitHub](https://github.com/PeerAhammad)

---

## Author

**PeerAhammad M Bagawan**

B.Tech Computer Science & Engineering
Visvesvaraya Technological University (VTU)
