import pandas as pd
from .census import load_and_preprocess_census, split_data as split_census_data
from .compas import load_and_preprocess_compas, split_data as split_compas_data
from .credit import load_and_preprocess_german_credit, split_data as split_credit_data

def preprocess_and_save_datasets():
    """Preprocess both Census Income and COMPAS datasets and save the processed files."""

    # Preprocess Census Income dataset
    print("Processing Census Income dataset...")
    census_file_path = 'data/raw/adult.csv'
    census_df = load_and_preprocess_census(census_file_path)

    # Save preprocessed Census data
    census_processed_path = 'data/processed/census_processed.csv'
    census_df.to_csv(census_processed_path, index=False)
    print(f"Census Income dataset saved to {census_processed_path}")

    # Split Census Income data into training and test sets
    X_train_census, X_test_census, y_train_census, y_test_census = split_census_data(census_df)
    print(f"Census Income - Training data shape: {X_train_census.shape}, Test data shape: {X_test_census.shape}")

    # Preprocess COMPAS dataset
    print("Processing COMPAS dataset...")
    compas_file_path = 'data/raw/compas.csv'
    compas_df = load_and_preprocess_compas(compas_file_path)

    # Save preprocessed COMPAS data
    compas_processed_path = 'data/processed/compas_processed.csv'
    compas_df.to_csv(compas_processed_path, index=False)
    print(f"COMPAS dataset saved to {compas_processed_path}")

    # Split COMPAS data into training and test sets
    X_train_compas, X_test_compas, y_train_compas, y_test_compas = split_compas_data(compas_df)
    print(f"COMPAS - Training data shape: {X_train_compas.shape}, Test data shape: {X_test_compas.shape}")

    # Preprocess German Credit dataset
    print("Processing German Credit dataset...")
    german_credit_df = load_and_preprocess_german_credit()

    # Save preprocessed German Credit data
    german_credit_processed_path = 'data/processed/credit_processed.csv'
    german_credit_df.to_csv(german_credit_processed_path, index=False)
    print(f"German Credit dataset saved to {german_credit_processed_path}")

    # Split German Credit data into training and test sets
    X_train_credit, X_test_credit, y_train_credit, y_test_credit = split_credit_data(german_credit_df)
    print(f"German Credit - Training data shape: {X_train_credit.shape}, Test data shape: {X_test_credit.shape}")

    # Save the training and test sets for all datasets
    pd.DataFrame(X_train_census).to_csv('data/processed/X_train_census.csv', index=False)
    pd.DataFrame(X_test_census).to_csv('data/processed/X_test_census.csv', index=False)
    pd.DataFrame(y_train_census).to_csv('data/processed/y_train_census.csv', index=False)
    pd.DataFrame(y_test_census).to_csv('data/processed/y_test_census.csv', index=False)

    pd.DataFrame(X_train_compas).to_csv('data/processed/X_train_compas.csv', index=False)
    pd.DataFrame(X_test_compas).to_csv('data/processed/X_test_compas.csv', index=False)
    pd.DataFrame(y_train_compas).to_csv('data/processed/y_train_compas.csv', index=False)
    pd.DataFrame(y_test_compas).to_csv('data/processed/y_test_compas.csv', index=False)

    pd.DataFrame(X_train_credit).to_csv('data/processed/X_train_credit.csv', index=False)
    pd.DataFrame(X_test_credit).to_csv('data/processed/X_test_credit.csv', index=False)
    pd.DataFrame(y_train_credit).to_csv('data/processed/y_train_credit.csv', index=False)
    pd.DataFrame(y_test_credit).to_csv('data/processed/y_test_credit.csv', index=False)

    print("Preprocessing complete.")
    
def split_and_save_dataset(dataset_name, processed_path, transformed_path, split_function):
    """
    Helper function to split a dataset into training and test sets and save them.
    
    :param dataset_name: The name of the dataset (e.g., 'census', 'compas', 'credit').
    :param processed_path: Path to the processed CSV file.
    :param transformed_path: Path where the split data (train/test) will be saved.
    :param split_function: The function that splits the dataset (e.g., split_census_data, split_compas_data).
    """
    print(f"Splitting {dataset_name} dataset...")
    df = pd.read_csv(processed_path)
    
    # Split the data
    X_train, X_test, y_train, y_test = split_function(df)
    print(f"{dataset_name.capitalize()} - Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    
    # Save the training and test sets
    pd.DataFrame(X_train).to_csv(f'{transformed_path}/X_train_{dataset_name}.csv', index=False)
    pd.DataFrame(X_test).to_csv(f'{transformed_path}/X_test_{dataset_name}.csv', index=False)
    pd.DataFrame(y_train).to_csv(f'{transformed_path}/y_train_{dataset_name}.csv', index=False)
    pd.DataFrame(y_test).to_csv(f'{transformed_path}/y_test_{dataset_name}.csv', index=False)
    print(f"{dataset_name.capitalize()} data splitting complete.")
    
def split_transformed_datasets():
    """Split the transformed CSV files for Census, COMPAS, and German Credit datasets into training and test sets."""

    # Define paths for transformed and processed data
    transformed_dir = 'data/transformed/'
    
    # Split and save the Census dataset
    split_and_save_dataset('census', 'data/transformed/census_processed.csv', transformed_dir, split_census_data)

    # Split and save the COMPAS dataset
    split_and_save_dataset('compas', 'data/transformed/compas_processed.csv', transformed_dir, split_compas_data)

    # Split and save the German Credit dataset
    split_and_save_dataset('credit', 'data/transformed/credit_processed.csv', transformed_dir, split_credit_data)
    
# preprocess_and_save_datasets()