from data.data_preprocessing import preprocess_and_save_datasets, split_transformed_datasets
from di_remover import disparate_impact_remover
from itemset_generation import save_frequent_itemsets
from model_training import run_model_for_dataset
from analysis import run_fairness_analysis
from plot import save_bias_visualizations

def preprocess_for_all():
    
    # split and produce processed datasets from raw
    print("Preprocess for datasets")
    # preprocess_and_save_datasets()
    
    print("Genereate DI Remover transformed datasets")
    # generate transformed dataset
    disparate_impact_remover('census', ['Race', 'Gender', 'age_cat'])
    disparate_impact_remover('compas', ['race', 'sex', 'age_cat'])
    disparate_impact_remover('credit', ['sex', 'age_cat'])
    
    split_transformed_datasets()
    print("Dataset splitting complete.")

def generate_frequent_itemsets():
    """Generate and save frequent itemsets for all datasets."""
    
    print("Generating frequent itemsets...")
    save_frequent_itemsets('compas', min_support=0.02)
    save_frequent_itemsets('census', min_support=0.02)
    save_frequent_itemsets('credit', min_support=0.02)
    print("Frequent itemsets generation complete.")
    
def train_and_save_models():
    """Train models on both processed and transformed datasets and save predictions."""
    
    # print("Training and evaluating models on non-transformed datasets...")
    # processed_path = 'data/processed'
    # run_model_for_dataset('compas', processed_path)
    # run_model_for_dataset('census', processed_path)
    # run_model_for_dataset('credit', processed_path)
    
    print("\nTraining and evaluating models on transformed datasets...")
    transformed_path = 'data/transformed'
    run_model_for_dataset('compas', transformed_path)
    run_model_for_dataset('census', transformed_path)
    run_model_for_dataset('credit', transformed_path)
    
    print("Model training complete.")
    
def perform_fairness_analysis():
    """Run fairness analysis on both processed and transformed datasets."""
    
    # print("Running fairness analysis on non-transformed datasets...")
    # run_fairness_analysis('compas')
    # run_fairness_analysis('census')
    # run_fairness_analysis('credit')
    
    print("\nRunning fairness analysis on transformed datasets...")
    run_fairness_analysis('compas', True)
    run_fairness_analysis('census', True)
    run_fairness_analysis('credit', True)
    
    print("Fairness analysis complete.")

def generate_bias_visualizations():
    """Generate and save bias visualizations for all datasets."""
    
    # print("Generating bias visualizations...")
    # save_bias_visualizations('compas', show_all_groups=True)
    # save_bias_visualizations('census', show_all_groups=True)
    # save_bias_visualizations('credit', show_all_groups=True)
    
    print("\nGenerating bias visualizations for transformed datasets...")
    save_bias_visualizations('compas', transformed=True, show_all_groups=True)
    save_bias_visualizations('census', transformed=True, show_all_groups=True)
    save_bias_visualizations('credit', transformed=True, show_all_groups=True)
    
    print("Bias visualizations complete.")
    
def main():
    # Step 1: Preprocess datasets
    preprocess_for_all()
    
    # Step 2: Generate frequent itemsets
    # generate_frequent_itemsets()
    
    # Step 3: Train models and save predictions
    train_and_save_models()
    
    # Step 4: Perform fairness analysis
    perform_fairness_analysis()
    
    # Step 5: Generate bias visualizations
    generate_bias_visualizations()
    
    print("Workflow completed!")
    
if __name__ == "__main__":
    main()