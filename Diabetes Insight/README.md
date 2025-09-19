# Diabetes Insight Report Generator

This project loads the **scikit-learn diabetes dataset**, performs **exploratory data analysis (EDA)**, builds a **linear regression model**, and automatically generates a **comprehensive PDF report** for presentation.

## Features
- Loads and cleans real diabetes dataset.
- Displays a **preview table** of the first 10 rows.
- Generates **correlation heatmap** to identify top predictors.
- Creates **scatter plots** for top 3 correlated features vs. disease progression.
- Fits a **linear regression model** (using top 5 features).
- Produces **diagnostic plots**:
  - Predicted vs Actual values
  - Residuals distribution
  - Coefficient bar chart
- Generates a **multi-page PDF report** with:
  - Title & metadata page
  - Visualizations & model results
  - Executive summary & recommendations
  - Appendix with statistics

## Requirements
- Python 3.x
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `reportlab`

## Usage
Run:
```bash
python diabetes.py