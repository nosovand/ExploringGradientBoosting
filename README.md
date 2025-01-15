# Exploring Gradient Boosting

## Overview

This repository contains the code and results from the semester project "Exploring Gradient Boosting," which focuses on implementing and analyzing various ensemble learning algorithms such as random forests, and gradient boosting. The project was carried out to gain a deeper understanding of these techniques by implementing them from scratch and comparing them to established models in the Scikit-learn library.

## Repository Structure

```
.
├── algorithms
│   ├── __init__.py                  # Package initializer
│   ├── random_forest.py             # Random Forest implementation
│   ├── tree.py                      # Decision Tree implementation
│   ├── gradient_boosting.py         # Gradient Boosting implementation
│   ├── fast_parameter_search.py     # Hyperparameter tuning functions
├── datasets
│   ├── winequality-red.csv          # Wine Quality Dataset
├── gradient_boosting_comparison.ipynb
├── forest_regressor_comparison.ipynb
├── tree_regressor_comparison.ipynb
├── tree_classifier_comparison.ipynb
├── forest_classifier_comparison.ipynb
├── requirements.txt                 # Dependencies for the project
├── Exploring_gradient_boosting_report.pdf # Detailed report
├── README.md                        # Project documentation (this file)
```

## Project Goals

- Understand and implement decision trees, random forests, and gradient boosting models from scratch.
- Validate the custom implementations by comparing them to Scikit-learn models using benchmark datasets.
- Analyze the performance improvements provided by ensemble learning techniques.
- Provide practical insights into how ensemble methods improve predictive accuracy.

## Datasets

### Wine Quality Dataset
- **Source:** UCI Machine Learning Repository
- **Purpose:** Classification tasks
- **Description:** Contains wine samples with 11 features and 6 classes.

### Scikit Diabetes Toy Dataset
- **Source:** Scikit-learn
- **Purpose:** Regression tasks
- **Description:** A small dataset with 10 features and a continuous target variable.

## Notebooks

- `tree_classifier_comparison.ipynb`: Analysis and comparison of decision tree classifiers.
- `tree_regressor_comparison.ipynb`: Analysis and comparison of decision tree regressors.
- `forest_classifier_comparison.ipynb`: Evaluation of random forest classifiers.
- `forest_regressor_comparison.ipynb`: Evaluation of random forest regressors.
- `gradient_boosting_comparison.ipynb`: Evaluation of gradient boosting models.

## Key Implementations

### Custom Algorithms
- **Decision Trees:** Custom implementation of decision tree classifiers and regressors.
- **Random Forests:** Implementation of bagging techniques using custom and Scikit-learn trees.
- **Gradient Boosting:** Sequential ensemble learning using custom and Scikit-learn trees.

### Highlights
- Models are tested for overfitting, random parameters, and hyperparameter-tuned scenarios.
- Metrics such as accuracy (classification) and R² (regression) are used for evaluation.
- Results validate that custom implementations closely mimic Scikit-learn models.

## Dependencies

Install required Python packages using:

```bash
pip install -r requirements.txt
```

## Documentation

The project report is available in [Exploring_gradient_boosting_report.pdf](Exploring_gradient_boosting_report.pdf). It details the theoretical background, methodologies, results, and conclusions.