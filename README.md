# ER-DD-SAA: EV Charging Station Location Probelm

This repository contains the Python code and datasets for numerical experiments on electric vehicle (EV) charging station location optimization with decision-dependent demand learning.

## Code Description

The experiments include synthetic data analysis and a real-world case study using New York State EV registration and charging station data.

| File | Description |
|---|---|
| `synthetic_onestep` | One-step estimation and neural network experiments using synthetic data. |
| `synthetic_twostep` | Two-step regression and optimization experiments using synthetic data. |
| `case_study_onestep.py` | One-step regression for the real-world case study. |
| `case_study_twostep.py` | Two-step regression and optimization for the real-world case study. |

## Data

- `data/`: New York data used in the real-world experiments.
- `original registration/`: EV registration and charging station data.

## Requirements

Python packages include NumPy, Pandas, SciPy, scikit-learn, Matplotlib, Gurobi, and Gurobi Machine Learning.

A valid Gurobi license is required for optimization.

## Contact

For questions about the code, please contact the repository author.
