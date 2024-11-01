import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def highlight_biased_groups(fairness_df, dpd_threshold=0.2, eod_threshold=0.2, di_threshold=0.8):
    """
    Highlight biased groups by checking if they exceed the fairness thresholds.
    :param fairness_df: DataFrame containing fairness analysis results.
    :param dpd_threshold: Threshold for Demographic Parity Difference (DPD).
    :param eod_threshold: Threshold for Equal Opportunity Difference (EOD).
    :param di_threshold: Threshold for Disparate Impact (DI).
    :return: DataFrame with an additional column indicating bias status.
    """
    # Identify biased groups
    fairness_df['is_biased'] = (
        (fairness_df['Demographic Parity Difference'].abs() > dpd_threshold) |
        (fairness_df['Equal Opportunity Difference'].abs() > eod_threshold) |
        (fairness_df['Disparate Impact'] < di_threshold)
    )
    return fairness_df

def save_bias_visualizations(dataset_name, transformed=False, show_all_groups=False):
    # Load the fairness analysis results
    if transformed:
        fairness_df = pd.read_csv(f'results/{dataset_name}/fairness_analysis_transformed.csv')
    else:
        fairness_df = pd.read_csv(f'results/{dataset_name}/fairness_analysis.csv')
    
    # Set thresholds for fairness concerns
    dpd_threshold = 0.2
    eod_threshold = 0.2
    di_threshold = 0.8
    
    print(fairness_df)
    
    # Highlight biased groups (either filter or show all)
    fairness_df = highlight_biased_groups(fairness_df, dpd_threshold, eod_threshold, di_threshold)

    if show_all_groups:
        bias_groups = fairness_df  # Show all groups, highlighting biased ones
    else:
        # Filter groups with bias concerns
        bias_groups = fairness_df[fairness_df['is_biased']]

    # Rename columns to abbreviations for better plot readability
    bias_groups = bias_groups.rename(columns={
        'Demographic Parity Difference': 'DPD',
        'Equal Opportunity Difference': 'EOD',
        'Disparate Impact': 'DI'
    })

    # Ensure the results directory exists
    results_dir = f'results/{dataset_name}'
    os.makedirs(results_dir, exist_ok=True)

    # Helper function to plot and save
    def plot_bar_metric(metric, title, ylabel, threshold=None, show_all=False):
        plt.figure(figsize=(12, 6))
        if show_all:
            # Only apply hue if you're visualizing all groups with highlights
            sns.barplot(x='Group_code', y=metric, data=bias_groups, hue='is_biased', dodge=False, palette={False: 'gray', True: 'red'})
            plt.legend(title='Biased', loc='upper right')
        else:
            sns.barplot(x='Group_code', y=metric, data=bias_groups)

        plt.axhline(0, color='black', linewidth=1)
        if threshold:
            plt.axhline(threshold, color='red', linestyle='--', label=f'{metric} Threshold ({threshold})')

        plt.title(title)
        plt.ylabel(ylabel)
        plt.xticks(rotation=90, ha='right')
        plt.tight_layout()

        # Save the plot
        filename = f"{results_dir}/{('transformed_' if transformed else '')}{metric.lower()}_by_group.png"
        plt.savefig(filename)
        plt.close()

    # Plot and save visualizations
    plot_bar_metric('DPD', 'Demographic Parity Difference (DPD) by Group', 'DPD', show_all=show_all_groups)
    plot_bar_metric('EOD', 'Equal Opportunity Difference (EOD) by Group', 'EOD', show_all=show_all_groups)
    plot_bar_metric('DI', 'Disparate Impact (DI) by Group', 'DI', threshold=di_threshold, show_all=show_all_groups)

    # Save Heatmap combining all fairness metrics
    plt.figure(figsize=(10, 8))
    heatmap_data = bias_groups.set_index('Group_code')[['DPD', 'EOD', 'DI']]
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', center=0)
    plt.title('Fairness Metrics Heatmap by Group')
    plt.tight_layout()

    # Save heatmap
    heatmap_filename = f"{results_dir}/{('transformed_' if transformed else '')}fairness_metrics_heatmap.png"
    plt.savefig(heatmap_filename)
    plt.close()

    print(f"Bias visualizations saved in {results_dir}")

if __name__ == "__main__":
    save_bias_visualizations('compas', show_all_groups=True)
    save_bias_visualizations('census', show_all_groups=True)
    save_bias_visualizations('credit', show_all_groups=True)
    
    save_bias_visualizations('compas', transformed=True, show_all_groups=True)
    save_bias_visualizations('census', transformed=True, show_all_groups=True)
    save_bias_visualizations('credit', transformed=True, show_all_groups=True)

    
