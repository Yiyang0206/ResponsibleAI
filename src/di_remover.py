from aif360.algorithms.preprocessing import DisparateImpactRemover
import pandas as pd
import os
import aif360.datasets

def create_intersectional_attribute(df, sensitive_attributes):
    """
    Combine multiple sensitive attributes into a single intersectional attribute.
    :param df: DataFrame containing the dataset.
    :param sensitive_attributes: List of sensitive attribute column names (e.g., ['Race', 'Gender', 'age_cat']).
    :return: DataFrame with a new 'intersectional_attr' column.
    """
    df['intersectional_attr'] = df[sensitive_attributes].astype(str).agg('_'.join, axis=1)
    return df

def match_intersectional_with_na(intersectional_value, privileged_combinations):
    """
    Custom function to match intersectional values with wildcard ('NA') privileged groups.
    :param intersectional_value: The intersectional attribute value from the dataset (e.g., '4_0_1').
    :param privileged_combinations: List of privileged group combinations (e.g., ['0_0_2', 'NA_1_0']).
    :return: Boolean indicating whether the intersectional value matches one of the privileged groups.
    """
    intersectional_parts = intersectional_value.split('_')
    
    for privileged_group in privileged_combinations:
        privileged_parts = privileged_group.split('_')
        match = True
        for i, part in enumerate(privileged_parts):
            if part != 'NA' and part != intersectional_parts[i]:
                match = False
                break
        if match:
            return 1
    return 0

def disparate_impact_remover(dataset_name, sensitive_attributes):
    """
    Applies Disparate Impact Remover to an intersectional attribute and saves the transformed data in a separate folder.
    Manually preserve the sensitive attributes (Race, Gender, age_cat).
    """
    # Create output directory for transformed data
    output_dir = f'data/transformed/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Define label and privileged combinations
    if dataset_name == "census":
        label = "Income"
        favorable_class = [1]
        privileged_combinations = ['4_0_1', 'NA_0_1', '4_0_0']
    elif dataset_name == "compas":
        label = "two_year_recid"
        favorable_class = [0]
        privileged_combinations = ['NA_1_2', '0_1_2', '2_1_2', '0_NA_2'] 
    elif dataset_name == "credit":
        label = "Risk"
        favorable_class = [1]
        privileged_combinations = ['3_1', '1_1']
    
    # Load original dataset
    df = pd.read_csv(f'data/processed/{dataset_name}_processed.csv')

    # Manually store the sensitive attributes to add them back later
    sensitive_attributes_df = df[sensitive_attributes].copy()

    # Create an intersectional attribute
    df = create_intersectional_attribute(df, sensitive_attributes)
    
    # Apply the wildcard matching to flag privileged groups
    df['is_privileged'] = df['intersectional_attr'].apply(lambda x: match_intersectional_with_na(x, privileged_combinations))

    # Remove sensitive attributes from the dataset before transformation
    df = df.drop(columns=sensitive_attributes)
    df = df.drop(columns=['intersectional_attr'])
    
    # Apply Disparate Impact Remover ONLY to the 'is_privileged' column
    dir = DisparateImpactRemover(repair_level=1.0)
    aif_data = aif360.datasets.StandardDataset(
        df,
        label_name=label,
        favorable_classes=favorable_class,
        protected_attribute_names=['is_privileged'],  # Use only the privileged flag column
        privileged_classes=[[1]],  # Privileged class is marked as 1 in 'is_privileged'
        categorical_features=[]
    )
    
    # Apply Disparate Impact Remover
    transformed_aif_data = dir.fit_transform(aif_data)
    
    # Convert back to pandas DataFrame
    df_transformed = transformed_aif_data.convert_to_dataframe()[0]

    # Reset index to ensure alignment before merging sensitive attributes back
    df_transformed = df_transformed.reset_index(drop=True)
    sensitive_attributes_df = sensitive_attributes_df.reset_index(drop=True)

    # Add the preserved sensitive attributes back to the transformed dataset
    df_transformed[sensitive_attributes] = sensitive_attributes_df
    print("After transformation:")
    print(df_transformed[sensitive_attributes].value_counts())
    
    # Save the transformed dataset
    df_transformed.to_csv(f'{output_dir}/{dataset_name}_processed.csv', index=False)
    print(f"Transformed {dataset_name} data with intersectional bias saved to {output_dir}")

# Example usage for Census dataset with intersectional bias (Race + Gender + age_cat)
disparate_impact_remover('census', ['Race', 'Gender', 'age_cat'])
disparate_impact_remover('compas', ['race', 'sex', 'age_cat'])
disparate_impact_remover('credit', ['sex', 'age_cat'])