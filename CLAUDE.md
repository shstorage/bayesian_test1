# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Small collection of Python scripts for **Bayesian inference visualizations**, specifically demonstrating coin-flip (binomial) Bayesian updating with discrete theta values.

## Running Scripts

```bash
python ba_2.py
```

Scripts produce matplotlib plots (e.g., `pmf_vs_likelihood.png`).

## Dependencies

- numpy
- matplotlib
- scipy (scipy.stats.binom)

## Architecture

- `ba_1.py` — Core `BernGrid` function: Bernoulli likelihood grid approximation with HDI (`hdi_of_grid`) and visualization (prior/likelihood/posterior as vertical bar plots). Accepts raw binary Data arrays. Runs a demo with triangular prior and `[0,0,0,1]` data.
- `ba_2.py` — Bayesian coin-flip analysis: computes prior (triangular), binomial likelihood (`scipy.stats.binom`), and posterior over 1001 discrete θ values. Compares two experiments (N=4,z=1 vs N=40,z=10) in a 3×2 subplot grid with MAP estimation.

## Notes

- Comments are in Korean.
- θ is discretized as `np.linspace(0, 1, 1001)` across all scripts.
