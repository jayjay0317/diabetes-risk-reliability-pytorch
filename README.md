# Diabetes Risk Reliability Analysis with PyTorch

## Overview

This project extends a previous scikit-learn based diabetes risk prediction pipeline into a PyTorch-focused machine learning research project.

The primary goal is not only to improve predictive performance, but also to investigate model reliability, calibration, and uncertainty estimation in clinical risk prediction under severe class imbalance.

This repository explores how deep learning models behave in preventative healthcare screening settings where minimizing false negatives is critically important.

---

## Previous End-to-End Deployment Project

This work builds upon a previously deployed diabetes risk prediction system developed using scikit-learn, Flask, Streamlit, Docker, and AWS infrastructure.

Previous project repository:
- [Diabetes ML Pipeline Analysis](https://github.com/jayjay0317/Diabetes-ML-Pipeline-Analysis)

The earlier project focused primarily on:
- End-to-end deployment engineering
- Clinical risk threshold optimization
- Class imbalance handling
- Production API infrastructure

In contrast, this repository focuses on:
- PyTorch deep learning models
- Reliability and calibration analysis
- Uncertainty estimation
- Research-oriented experimentation

---

## Research Motivation

Clinical risk prediction models often operate under highly imbalanced conditions where standard accuracy metrics can be misleading.

In the original project, traditional machine learning models achieved high overall accuracy while failing to correctly identify minority-class patients.

This project investigates whether PyTorch-based neural networks can provide:

- Better recall-focused prediction performance
- Improved probability calibration
- More reliable uncertainty estimation
- Better robustness under class imbalance

---

## Dataset

This project uses the BRFSS 2015 Diabetes Health Indicators dataset.

- Source: CDC Behavioral Risk Factor Surveillance System (BRFSS)
- Task: Binary diabetes risk prediction
- Features include:
  - BMI
  - General Health
  - Physical Health
  - Mental Health
  - Blood Pressure
  - Cholesterol
  - Lifestyle and socioeconomic indicators

Due to severe class imbalance, the original multiclass target was strategically converted into a binary clinical risk prediction task.

---

## Baseline Machine Learning Models

Traditional machine learning baselines include:

- Logistic Regression
- Balanced Logistic Regression
- Random Forest
- Threshold Optimization using Youden’s J Statistic

The baseline pipeline includes:

- Scikit-learn Pipelines
- ColumnTransformer preprocessing
- Log transformation for skewed variables
- StandardScaler normalization
- Cross-validated ROC-AUC evaluation
- Recall-focused threshold tuning

---

## PyTorch Deep Learning Model

This project implements a PyTorch-based multilayer perceptron (MLP) for tabular clinical risk prediction.

Planned components include:

- PyTorch Dataset and DataLoader
- Custom training loop
- BCEWithLogitsLoss with class weighting
- Validation monitoring
- Learning curve analysis
- Model comparison against Random Forest baseline

---

## Reliability & Calibration Analysis

This repository focuses heavily on model reliability rather than raw accuracy alone.

Planned evaluation methods include:

- ROC-AUC
- Precision / Recall tradeoff analysis
- Calibration curves
- Brier Score
- Threshold sensitivity analysis
- Error analysis for high-confidence failures

---

## Uncertainty Estimation

To better understand model confidence in clinical prediction settings, this project explores uncertainty-aware deep learning techniques.

Planned experiments include:

- Monte Carlo Dropout
- Prediction variance analysis
- Confidence distribution visualization
- Uncertain sample identification

---

## Project Structure

```text
├── notebooks/
├── src/
├── models/
├── results/
├── data/
```

---

## Current Status

Project initialization and baseline migration are currently in progress.

Upcoming steps:

- Reproduce scikit-learn baseline
- Implement PyTorch MLP
- Add calibration analysis
- Explore uncertainty estimation techniques