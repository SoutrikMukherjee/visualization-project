# Machine Learning Data Visualization Project

## Overview
This comprehensive project demonstrates advanced data visualization techniques across the entire machine learning lifecycle, showcasing skills in:

- **Exploratory Data Analysis (EDA)** with both static and interactive visualizations
- **Model Training and Evaluation** using visual performance metrics
- **Communication of Results** through executive-style dashboards

The project focuses on credit card fraud detection, using visualization as a tool for understanding data patterns, evaluating model performance, and communicating results effectively.

## Features

### 1. Exploratory Data Analysis
- Missing value visualization
- Class distribution analysis and imbalance visualization
- Feature distributions and correlations
- Dimensionality reduction visualization (PCA & t-SNE)
- 3D interactive scatter plots for complex relationships

### 2. Model Evaluation Visualizations
- ROC curves with AUC comparison
- Precision-Recall curves
- Confusion matrices
- Feature importance visualization
- Learning curves for detecting overfitting
- Interactive model comparison dashboards

### 3. Executive Summary Dashboard
- Interactive Plotly dashboard for non-technical stakeholders
- Key performance metrics visualization
- Model comparison in business context

## Libraries Used
- **Data Manipulation:** NumPy, Pandas
- **Machine Learning:** Scikit-learn
- **Visualization:**
  - Static: Matplotlib, Seaborn
  - Interactive: Plotly
  - Missing data: Missingno
- **Dimensionality Reduction:** PCA, t-SNE

## Project Structure
```
ml-visualization-project/
│
├── README.md                  # This file
├── ml_visualization.py        # Main project script
├── requirements.txt           # Dependencies
│
└── visualizations/            # Output directory
    ├── missing_values.png
    ├── class_distribution.png
    ├── feature_distributions.png
    ├── correlation_matrix.png
    ├── pca_visualization.png
    ├── tsne_visualization.png
    ├── 3d_scatter_interactive.html
    ├── roc_curves.png
    ├── precision_recall_curves.png
    ├── confusion_matrices.png
    ├── feature_importances.png
    ├── learning_curves.png
    ├── model_comparison_dashboard.html
    └── executive_summary.html
```

## Installation & Usage

1. Clone this repository:
```bash
git clone https://github.com/your-username/ml-visualization-project.git
cd ml-visualization-project
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Download the dataset:
   - Get the Credit Card Fraud Detection dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   - Place `creditcard.csv` in the project root directory

4. Run the project:
```bash
python ml_visualization.py
```

5. View the results:
   - Static visualizations: Check the PNG files in the `visualizations` directory
   - Interactive dashboards: Open the HTML files in a web browser

## Requirements
```
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
plotly>=5.3.0
missingno>=0.5.0
```

## Key Visualization Takeaways

### For Data Understanding
- Visualizing class imbalance helps understand the need for appropriate evaluation metrics
- Dimensionality reduction plots reveal hidden patterns in high-dimensional data
- Interactive 3D plots enable exploration of complex feature relationships

### For Model Evaluation
- ROC and PR curves provide nuanced performance understanding beyond accuracy
- Learning curves help diagnose overfitting/underfitting
- Feature importance visualizations provide model interpretability

### For Communication
- Interactive dashboards make technical results accessible to stakeholders
- Summary visualizations highlight key findings without overwhelming with details

## License
MIT

## Author
Soutrik Mukherjee

---

*This project was created to demonstrate proficiency in data visualization for machine learning as required for ML internship positions.*
