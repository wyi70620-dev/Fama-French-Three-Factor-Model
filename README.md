# Fama-French Three-Factor Model Extension for Stock Excess Return Modeling

This repository contains a research-oriented quantitative finance project based on the Fama-French three-factor framework.  
The goal is to model stock excess returns and explore practical extensions through time-varying effects, alternative loss functions, and nonlinear modeling with GAM (Generalized Additive Models).

## Resume Highlights

- Built an end-to-end factor modeling workflow for stock excess return analysis.
- Implemented multiple modeling strategies: baseline FF3, time-varying design, robust loss-based variants, and GAM-based nonlinear extensions.
- Produced evaluation outputs and visual diagnostics to compare model behavior across approaches.
- Organized reproducible research code and documentation suitable for academic and portfolio presentation.

## Method Overview

- **Baseline factor modeling**: estimate relationships between returns and Fama-French factors.
- **Time-varying extension**: capture changing factor sensitivities over time.
- **Loss-function exploration**: test alternative objective functions for improved robustness.
- **GAM extension**: model potential nonlinear effects beyond linear factor assumptions.

## Repository Structure

- `Python代码示例/` - main Python scripts for data loading, preprocessing, analysis, modeling, and visualization.
- `备用代码/` - alternative implementations and backup versions.
- `数据/` - sample datasets and prediction outputs.
- `研究+方案/` - research notes, design plans, and supporting documents.

> Note: Some folder and file names are in Chinese. The code and workflow are still straightforward to run by following the script order below.

## Quick Start

1. Use Python 3.9+.
2. Install dependencies:
   - `pandas`
   - `numpy`
   - `statsmodels`
   - `matplotlib`
3. Run scripts in `Python代码示例/` in this order:
   - `数据读取与预处理.py` (data loading and preprocessing)
   - `数据特征分析.py` (feature analysis)
   - `时变方案.py` / `损失函数方案.py` / `GAM方案.py` (model variants)
   - `统计指标与可视化.py` (evaluation metrics and visualization)

## Project Purpose

This repository is intended for research demonstration and portfolio use (e.g., resume GitHub link), showcasing quantitative modeling, statistical analysis, and research engineering practices.
