# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python scripts for **Bayesian inference** demonstrations: coin-flip (binomial) Bayesian updating and corrosion rate estimation with hierarchical Bayesian models.

## Running Scripts

```bash
python ba_1.py        # BernGrid demo → saves plot
python ba_2.py        # Coin-flip comparison → saves pmf_vs_likelihood.png
python test1.py       # Empirical Bayes corrosion → saves empirical_bayes_result.png
python test2.py       # MCMC hierarchical Bayes corrosion → saves mcmc_bayes_result.png
```

## Dependencies

- numpy, matplotlib, scipy
- pandas (test1/test2)
- pymc, arviz (test2 only — uses `numpyro` NUTS sampler)

## Architecture

**Coin-flip Bayesian updating (ba_1.py, ba_2.py):**
- `ba_1.py` — `BernGrid` function: grid approximation of Bernoulli likelihood with HDI (`hdi_of_grid`) and bar-plot visualization. Accepts raw binary data arrays.
- `ba_2.py` — Compares two experiments (N=4,z=1 vs N=40,z=10) in a 3×2 subplot; uses `scipy.stats.binom` for binomial likelihood.

**Corrosion rate estimation (test1.py, test2.py):**
- Domain: 12×13 grid of measurement points, each with thickness readings at t=1,3,5. OLS linear regression per point gives corrosion rate (β). Goal: estimate the worst-case (minimum β) across all points.
- `test1.py` — **Empirical Bayes** (closed-form): normal-normal conjugate update; uses Monte Carlo order statistics to find the distribution of `min(β)`.
- `test2.py` — **Hierarchical MCMC** via PyMC: partial pooling with hyperpriors on `mu_beta`, `sigma_beta`; extracts order statistic from MCMC posterior samples directly.

Both corrosion scripts produce a 2×2 subplot: (a) shrinkage scatter, (b) worst-point posterior, (c) order statistic distribution, (d) t=5 thickness prediction with 95% CI.

## Notes

- Comments are in Korean.
- θ is discretized as `np.linspace(0, 1, 1001)` in ba_1/ba_2.
- Font set to `Malgun Gothic` in test1/test2 (Korean labels); may need adjustment on non-Windows systems.
- `np.random.seed(42)` used in test1/test2 for reproducibility.
