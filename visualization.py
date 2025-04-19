# ML Data Visualization Project
# Author: [Your Name]

"""
This project demonstrates advanced data visualization techniques for machine learning,
covering the entire ML lifecycle from exploratory data analysis to model evaluation
and results communication.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import missingno as msno
import os
import warnings
warnings.filterwarnings('ignore')

# Set plot aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Create output directory for visualizations
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# Part 1: Load and Explore the Dataset
# Using a credit card fraud dataset as an example
def load_and_explore_data():
    print("Loading and exploring dataset...")
    
    # Data can be downloaded from: 
    # https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    df = pd.read_csv('creditcard.csv')
    
    # Basic dataset information
    print(f"Dataset shape: {df.shape}")
    print("\nDataset info:")
    print(df.info())
    
    print("\nClass distribution:")
    print(df['Class'].value_counts())
    
    return df

# Part 2: Exploratory Data Analysis (EDA)
def perform_eda(df):
    print("\nPerforming exploratory data analysis...")
    
    # 2.1: Check for missing values
    plt.figure(figsize=(10, 6))
    msno.matrix(df)
    plt.title('Missing Value Analysis')
    plt.tight_layout()
    plt.savefig('visualizations/missing_values.png')
    
    # 2.2: Distribution of target variable (imbalanced dataset visualization)
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Class', data=df)
    plt.title('Class Distribution (0: Normal, 1: Fraud)')
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    # Add percentage labels
    total = len(df)
    for p in plt.gca().patches:
        percentage = f'{100 * p.get_height() / total:.2f}%'
        plt.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height() + 0.1),
                     ha='center', va='baseline')
    
    plt.tight_layout()
    plt.savefig('visualizations/class_distribution.png')
    
    # 2.3: Distribution of features
    plt.figure(figsize=(15, 12))
    for i, feature in enumerate(df.columns[:6]):  # First 6 features
        plt.subplot(2, 3, i+1)
        sns.histplot(df[feature], kde=True)
        plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.savefig('visualizations/feature_distributions.png')
    
    # 2.4: Feature correlations
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.iloc[:, :-1].corr()
    mask = np.triu(correlation_matrix)
    sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('visualizations/correlation_matrix.png')
    
    # 2.5: PCA visualization for dimensionality reduction
    pca = PCA(n_components=2)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.iloc[:, :-1])
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[df['Class']==0, 0], X_pca[df['Class']==0, 1], 
                alpha=0.2, s=10, label='Normal', c=colors[0])
    plt.scatter(X_pca[df['Class']==1, 0], X_pca[df['Class']==1, 1], 
                alpha=0.8, s=50, label='Fraud', c=colors[1])
    plt.title('PCA: 2D Projection of the Dataset')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('visualizations/pca_visualization.png')
    
    # 2.6: t-SNE visualization for non-linear dimensionality reduction
    # Using a smaller subset due to computational constraints
    subset_indices = np.random.choice(df.shape[0], size=5000, replace=False)
    subset_df = df.iloc[subset_indices]
    
    X_subset = subset_df.iloc[:, :-1].values
    y_subset = subset_df.iloc[:, -1].values
    
    X_subset_scaled = scaler.fit_transform(X_subset)
    X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X_subset_scaled)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[y_subset==0, 0], X_tsne[y_subset==0, 1], 
                alpha=0.2, s=10, label='Normal', c=colors[0])
    plt.scatter(X_tsne[y_subset==1, 0], X_tsne[y_subset==1, 1], 
                alpha=0.8, s=50, label='Fraud', c=colors[1])
    plt.title('t-SNE: Non-linear 2D Projection')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('visualizations/tsne_visualization.png')
    
    # 2.7: Advanced visualization using Plotly (interactive)
    fraud = df[df['Class'] == 1]
    normal = df[df['Class'] == 0].sample(n=len(fraud)*5)  # Downsample for better visualization
    
    fig = px.scatter_3d(
        pd.concat([fraud, normal]), 
        x='V14', y='V12', z='V10',
        color='Class', 
        opacity=0.8,
        color_discrete_map={0: colors[0], 1: colors[1]},
        title='3D Scatter Plot of Key Features'
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title='V14',
            yaxis_title='V12',
            zaxis_title='V10'
        )
    )
    
    fig.write_html('visualizations/3d_scatter_interactive.html')
    
    return X_scaled, df['Class'].values, scaler

# Part 3: Model Training and Evaluation
def train_and_evaluate_models(X, y, scaler):
    print("\nTraining and evaluating models...")
    
    # 3.1: Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 3.2: Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    # 3.3: Train models and store results
    results = {}
    feature_importances = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'model': model,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
        # Store feature importances if available
        if hasattr(model, 'feature_importances_'):
            feature_importances[name] = model.feature_importances_
    
    # 3.4: Visualize model performance
    visualize_model_performance(results, feature_importances)
    
    return results

# Part 4: Visualize Model Performance
def visualize_model_performance(results, feature_importances):
    print("\nVisualizing model performance...")
    
    # 4.1: ROC curves
    plt.figure(figsize=(10, 8))
    
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(result['y_test'], result['y_prob'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('visualizations/roc_curves.png')
    
    # 4.2: Precision-Recall curves
    plt.figure(figsize=(10, 8))
    
    for name, result in results.items():
        precision, recall, _ = precision_recall_curve(result['y_test'], result['y_prob'])
        plt.plot(recall, precision, lw=2, label=f'{name}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('visualizations/precision_recall_curves.png')
    
    # 4.3: Confusion matrices
    fig, axes = plt.subplots(1, len(results), figsize=(15, 5))
    
    for i, (name, result) in enumerate(results.items()):
        cm = confusion_matrix(result['y_test'], result['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'{name} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('visualizations/confusion_matrices.png')
    
    # 4.4: Feature importances
    if feature_importances:
        plt.figure(figsize=(12, 10))
        
        for i, (name, importances) in enumerate(feature_importances.items()):
            plt.subplot(len(feature_importances), 1, i+1)
            indices = np.argsort(importances)[::-1]
            plt.barh(range(10), importances[indices[:10]], align='center')
            plt.yticks(range(10), [f'V{indices[j]}' for j in range(10)])
            plt.title(f'{name} Feature Importance')
            plt.xlabel('Importance')
            plt.tight_layout()
        
        plt.savefig('visualizations/feature_importances.png')
    
    # 4.5: Learning curves to detect overfitting
    fig, axes = plt.subplots(1, len(results), figsize=(18, 6))
    
    for i, (name, result) in enumerate(results.items()):
        model = result['model']
        
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train, cv=5, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='roc_auc'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        axes[i].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                           alpha=0.1, color=colors[0])
        axes[i].fill_between(train_sizes, test_mean - test_std, test_mean + test_std, 
                           alpha=0.1, color=colors[1])
        axes[i].plot(train_sizes, train_mean, 'o-', color=colors[0], label='Training score')
        axes[i].plot(train_sizes, test_mean, 'o-', color=colors[1], label='Cross-validation score')
        axes[i].set_title(f'{name} Learning Curve')
        axes[i].set_xlabel('Training examples')
        axes[i].set_ylabel('ROC AUC Score')
        axes[i].grid(True)
        axes[i].legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('visualizations/learning_curves.png')
    
    # 4.6: Interactive Model Comparison Dashboard with Plotly
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Model Comparison: Accuracy', 'Model Comparison: Precision', 
                       'Model Comparison: Recall', 'Model Comparison: F1 Score')
    )
    
    metrics = {}
    for name, result in results.items():
        report = classification_report(result['y_test'], result['y_pred'], output_dict=True)
        metrics[name] = {
            'Accuracy': report['accuracy'],
            'Precision': report['1']['precision'],
            'Recall': report['1']['recall'],
            'F1 Score': report['1']['f1-score']
        }
    
    # Convert to DataFrame for easier plotting
    metrics_df = pd.DataFrame(metrics)
    
    # Plot each metric
    x = list(metrics.keys())
    
    # Accuracy
    fig.add_trace(go.Bar(
        x=x,
        y=metrics_df.loc['Accuracy'],
        marker_color=colors[0],
        name='Accuracy'
    ), row=1, col=1)
    
    # Precision
    fig.add_trace(go.Bar(
        x=x,
        y=metrics_df.loc['Precision'],
        marker_color=colors[1],
        name='Precision'
    ), row=1, col=2)
    
    # Recall
    fig.add_trace(go.Bar(
        x=x,
        y=metrics_df.loc['Recall'],
        marker_color=colors[2],
        name='Recall'
    ), row=2, col=1)
    
    # F1 Score
    fig.add_trace(go.Bar(
        x=x,
        y=metrics_df.loc['F1 Score'],
        marker_color=colors[3],
        name='F1 Score'
    ), row=2, col=2)
    
    fig.update_layout(
        height=800,
        width=1000,
        title_text='Model Performance Metrics Comparison',
        showlegend=False
    )
    
    fig.write_html('visualizations/model_comparison_dashboard.html')

# Part 5: Create an Executive Summary Dashboard
def create_executive_summary():
    print("\nCreating executive summary dashboard...")
    
    # This would typically be created with BI tools like Tableau or Power BI
    # Here we'll simulate it with a Plotly dashboard
    
    # Placeholder data for executive summary
    models = ['Logistic Regression', 'Random Forest', 'Gradient Boosting']
    accuracy = [0.92, 0.97, 0.98]
    precision = [0.74, 0.85, 0.89]
    recall = [0.68, 0.82, 0.84]
    f1 = [0.71, 0.83, 0.86]
    auc = [0.84, 0.91, 0.93]
    
    exec_fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "bar"}, {"type": "pie"}],
               [{"type": "scatter", "colspan": 2}, None]],
        subplot_titles=('Model Performance Comparison', 'Fraud vs Normal Transactions', 
                        'Performance Metrics by Model')
    )
    
    # Model Performance Comparison
    exec_fig.add_trace(
        go.Bar(x=models, y=auc, name='AUC', marker_color=colors[0]),
        row=1, col=1
    )
    
    # Class Distribution
    exec_fig.add_trace(
        go.Pie(
            labels=['Normal (99.83%)', 'Fraud (0.17%)'],
            values=[99.83, 0.17],
            marker_colors=[colors[2], colors[3]]
        ),
        row=1, col=2
    )
    
    # Performance Metrics by Model
    for i, model in enumerate(models):
        exec_fig.add_trace(
            go.Scatter(
                x=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
                y=[accuracy[i], precision[i], recall[i], f1[i], auc[i]],
                mode='lines+markers',
                name=model
            ),
            row=2, col=1
        )
    
    exec_fig.update_layout(
        height=800,
        width=1000,
        title_text='Credit Card Fraud Detection: Executive Summary',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    exec_fig.write_html('visualizations/executive_summary.html')

# Part 6: Main Function to Run the Project
def main():
    # Welcome message
    print("=" * 80)
    print("ML Data Visualization Project - Demonstrating ML Visualization Skills")
    print("=" * 80)
    
    # Step 1: Data Loading and Exploration
    df = load_and_explore_data()
    
    # Step 2: Exploratory Data Analysis
    X_scaled, y, scaler = perform_eda(df)
    
    # Step 3: Model Training and Evaluation
    results = train_and_evaluate_models(X_scaled, y, scaler)
    
    # Step 4: Create Executive Summary
    create_executive_summary()
    
    print("\nProject completed successfully!")
    print(f"Visualizations saved to: {os.path.abspath('visualizations')}")
    print("\nTo view interactive visualizations, open the HTML files in a web browser.")

if __name__ == "__main__":
    main()
