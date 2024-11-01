import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_fairness_report(normal_csv, transformed_csv):
    """Load the fairness reports for normal and transformed datasets."""
    df_normal = pd.read_csv(normal_csv)
    df_transformed = pd.read_csv(transformed_csv)
    return df_normal, df_transformed

def plot_fairness_metrics_comparison(df_normal, df_transformed, group_column='Group_code', metrics=['Demographic Parity Difference', 'Equal Opportunity Difference', 'Disparate Impact']):
    """Plot the comparison of fairness metrics with side-by-side bars (left: normal, right: transformed)."""
    
    # Merge the two dataframes on the group column
    df_comparison = df_normal[[group_column] + metrics].merge(df_transformed[[group_column] + metrics], on=group_column, suffixes=('_normal', '_transformed'))
    
    # Plotting comparison for each metric
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        
        # Prepare data for side-by-side bar plots
        df_melted = df_comparison.melt(id_vars=group_column, value_vars=[f'{metric}_normal', f'{metric}_transformed'], var_name='Dataset', value_name=metric)
        df_melted['Dataset'] = df_melted['Dataset'].apply(lambda x: 'Normal' if 'normal' in x else 'Transformed')
        
        # Plot side-by-side bars
        sns.barplot(x=group_column, y=metric, hue='Dataset', data=df_melted, palette=['blue', 'orange'], dodge=True)
        
        plt.title(f'Comparison of {metric} (Normal vs. Transformed)')
        plt.xticks(rotation=90, ha='right')
        plt.ylabel(metric)
        
        # Add dashed line at 0.8 for Disparate Impact (DI)
        if metric == 'Disparate Impact':
            plt.axhline(0.8, color='red', linestyle='--', label='DI Threshold (0.8)')
        
        plt.legend(title="Dataset", loc='upper left')
        plt.tight_layout()
        plt.show()

# Example usage:
df_normal, df_transformed = load_fairness_report('results/compas/fairness_analysis.csv', 'results/compas/fairness_analysis_transformed.csv')
plot_fairness_metrics_comparison(df_normal, df_transformed)
