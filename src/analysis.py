# import pandas as pd
# import json
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# import os

# def load_category_mappings(dataset_name):
#     """Load the category mappings for the dataset (e.g., race, sex, age_cat for COMPAS or Census)."""
#     mapping_file = f'data/mappings/{dataset_name}_mappings.json'
#     with open(mapping_file, 'r') as file:
#         mappings = json.load(file)
#     return mappings

# def generate_group_code_from_row(row, mappings, sensitive_attributes):
#     """Generate group code using encoded (mapped) values for each sensitive attribute in a row."""
#     group_code = []
#     for attr in sensitive_attributes:
#         value = row[attr]  # Get the value from the row
#         # Check if the value exists in the mapping for the attribute
#         if value in mappings[attr].values():
#             # If the value exists, find its corresponding encoded value
#             for k, v in mappings[attr].items():
#                 if v == value:
#                     group_code.append(str(v))  # Append the encoded value to the group code
#                     break
#         else:
#             group_code.append('NA')  # If no valid mapping found, append 'NA'
    
#     return '_'.join(group_code)

# def decode_group_code(group_code, mappings, sensitive_attributes):
#     """Convert the group code back to real labels using the mappings."""
#     group_values = group_code.split('_')
#     real_labels = []
    
#     for attr, value in zip(sensitive_attributes, group_values):
#         if value != 'NA':
#             # Reverse the mapping to get the real label
#             real_label = next((k for k, v in mappings[attr].items() if v == int(value)), 'NA')
#             real_labels.append(real_label)
#         else:
#             real_labels.append('NA')
    
#     return ', '.join(real_labels)

# def match_group_code(itemset_group_code, test_group_code):
#     """Match the frequent itemset's group code with the test set's group code, supporting partial matches."""
#     itemset_parts = itemset_group_code.split('_')
#     test_parts = test_group_code.split('_')
    
#     # Perform matching, allowing 'NA' to act as a wildcard
#     for i in range(len(itemset_parts)):
#         if itemset_parts[i] != 'NA' and itemset_parts[i] != test_parts[i]:
#             return False  # No match
#     return True  # Match found

# def calculate_fairness_metrics(y_true, y_pred, overall_ppr):
#     """Calculate various fairness metrics for the given group."""
#     positive_rate = y_pred.mean()  # Positive Prediction Rate
#     accuracy = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred, zero_division=1)
#     recall = recall_score(y_true, y_pred, zero_division=1)
#     fpr = ((y_true == 0) & (y_pred == 1)).mean()  # False Positive Rate

#     # New fairness metrics
#     dp_diff = positive_rate - overall_ppr  # Demographic Parity Difference
#     eod = recall - overall_ppr  # Equal Opportunity Difference
#     # Disparate Impact: Ratio of positive rates between groups
#     di = positive_rate / overall_ppr if overall_ppr > 0 else 0
    
#     return positive_rate, accuracy, precision, recall, fpr, dp_diff, eod, di

# def fairness_analysis_with_group_code(df_test, y_test, y_pred, mappings, sensitive_attributes, frequent_itemsets):
#     """Perform fairness analysis by matching group codes between X_test and frequent itemsets."""
    
#     # Create a column for group codes in the test set
#     df_test['group_code'] = df_test.apply(lambda row: generate_group_code_from_row(row, mappings, sensitive_attributes), axis=1)
    
#     overall_ppr = y_pred.mean()  # Calculate overall Positive Prediction Rate
#     results = []
    
#     for _, itemset in frequent_itemsets.iterrows():
#         group_code = itemset['group_code']
        
#         # Filter the test set for this group code
#         group_data = df_test[df_test['group_code'].apply(lambda x: match_group_code(group_code, x))]
        
#         if len(group_data) == 0:
#             continue  # Skip if no data matches this group code
        
#         # Calculate fairness metrics for this group
#         y_true_group = y_test.loc[group_data.index]
#         y_pred_group = y_pred.loc[group_data.index]
        
#         positive_rate, accuracy, precision, recall, fpr, dp_diff, eod, di = calculate_fairness_metrics(y_true_group, y_pred_group, overall_ppr)
        
#         real_label = decode_group_code(group_code, mappings, sensitive_attributes)

#         results.append({
#             'Group_code_code': group_code,
#             'Group_code': real_label,
#             'Group Size': len(group_data),
#             'Positive Prediction Rate': positive_rate,
#             # 'Accuracy': accuracy,
#             # 'Precision': precision,
#             # 'Recall': recall,
#             # 'FPR': fpr,
#             'Demographic Parity Difference': dp_diff,
#             'Equal Opportunity Difference': eod,
#             'Disparate Impact': di
#         })
    
#     # Convert results to DataFrame
#     fairness_df = pd.DataFrame(results)
#     return fairness_df

# def run_fairness_analysis(dataset_name, transformed=False):
    
#     if transformed:
#         path = 'transformed'
#     else:
#         path = 'processed'
#     # Load data
#     X_test = pd.read_csv(f'data/{path}/X_test_{dataset_name}.csv')
#     y_test = pd.read_csv(f'data/{path}/y_test_{dataset_name}.csv').squeeze()
#     y_pred = pd.read_csv(f'data/{path}/{dataset_name}_y_pred.csv').squeeze()

#     # Load frequent itemsets
#     frequent_itemsets = pd.read_csv(f'results/{dataset_name}/frequent_itemsets.csv')

#     # Load category mappings
#     mappings = load_category_mappings(dataset_name)
    
#     if dataset_name == 'compas':
#         sensitive_attributes = ['race', 'sex', 'age_cat']
#     elif dataset_name == 'census':
#         sensitive_attributes = ['Race', 'Gender', 'age_cat']
#     elif dataset_name == 'credit':
#         sensitive_attributes = ['sex', 'age_cat']
#     # Perform fairness analysis with group code matching
#     fairness_df = fairness_analysis_with_group_code(X_test, y_test, y_pred, mappings, sensitive_attributes, frequent_itemsets)

#     # Save the fairness analysis results
#     os.makedirs(f'results/{dataset_name}', exist_ok=True)

#     if transformed:
#         fairness_df.to_csv(f'results/{dataset_name}/fairness_analysis_transformed.csv', index=False)
#     else:    
#         fairness_df.to_csv(f'results/{dataset_name}/fairness_analysis.csv', index=False)
    
#     print(f"Fairness analysis results saved - {dataset_name} | {path}")

# if __name__ == "__main__":
#     run_fairness_analysis('compas')
#     run_fairness_analysis('census')
#     run_fairness_analysis('credit')
    
#     run_fairness_analysis('compas', True)
#     run_fairness_analysis('census', True)
#     run_fairness_analysis('credit', True)

import pandas as pd
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os

def load_category_mappings(dataset_name):
    """Load the category mappings for the dataset (e.g., race, sex, age_cat for COMPAS or Census)."""
    mapping_file = f'data/mappings/{dataset_name}_mappings.json'
    with open(mapping_file, 'r') as file:
        mappings = json.load(file)
    return mappings

def generate_group_code_from_row(row, mappings, sensitive_attributes):
    """Generate group code using encoded (mapped) values for each sensitive attribute in a row."""
    group_code = []
    for attr in sensitive_attributes:
        value = row[attr]  # Get the value from the row
        # Check if the value exists in the mapping for the attribute
        if value in mappings[attr].values():
            # If the value exists, find its corresponding encoded value
            for k, v in mappings[attr].items():
                if v == value:
                    group_code.append(str(v))  # Append the encoded value to the group code
                    break
        else:
            group_code.append('NA')  # If no valid mapping found, append 'NA'
    
    return '_'.join(group_code)

def decode_group_code(group_code, mappings, sensitive_attributes):
    """Convert the group code back to real labels using the mappings."""
    group_values = group_code.split('_')
    real_labels = []
    
    for attr, value in zip(sensitive_attributes, group_values):
        if value != 'NA':
            # Reverse the mapping to get the real label
            real_label = next((k for k, v in mappings[attr].items() if v == int(value)), 'NA')
            real_labels.append(real_label)
        else:
            real_labels.append('NA')
    
    return ', '.join(real_labels)

def match_group_code(itemset_group_code, test_group_code):
    """Match the frequent itemset's group code with the test set's group code, supporting partial matches."""
    itemset_parts = itemset_group_code.split('_')
    test_parts = test_group_code.split('_')
    
    # Perform matching, allowing 'NA' to act as a wildcard
    for i in range(len(itemset_parts)):
        if itemset_parts[i] != 'NA' and itemset_parts[i] != test_parts[i]:
            return False  # No match
    return True  # Match found

def calculate_fairness_metrics(y_true, y_pred, overall_ppr, overall_recall):
    """Calculate various fairness metrics for the given group."""
    positive_rate = y_pred.mean()  # Positive Prediction Rate
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    fpr = ((y_true == 0) & (y_pred == 1)).mean()  # False Positive Rate

    # Fairness metrics relative to the overall model
    dp_diff = positive_rate - overall_ppr  # Demographic Parity Difference
    eod = recall - overall_recall  # Equal Opportunity Difference
    di = positive_rate / overall_ppr if overall_ppr > 0 else 0  # Disparate Impact
    
    return positive_rate, accuracy, precision, recall, fpr, dp_diff, eod, di

def fairness_analysis_with_group_code(df_test, y_test, y_pred, mappings, sensitive_attributes, frequent_itemsets):
    """Perform fairness analysis by matching group codes between X_test and frequent itemsets."""
    
    # Create a column for group codes in the test set
    df_test['group_code'] = df_test.apply(lambda row: generate_group_code_from_row(row, mappings, sensitive_attributes), axis=1)
    
    overall_ppr = y_pred.mean()  # Overall Positive Prediction Rate
    overall_recall = recall_score(y_test, y_pred, zero_division=1)  # Overall recall for the entire dataset
    results = []
    
    for _, itemset in frequent_itemsets.iterrows():
        group_code = itemset['group_code']
        
        # Filter the test set for this group code
        group_data = df_test[df_test['group_code'].apply(lambda x: match_group_code(group_code, x))]
        
        if len(group_data) == 0:
            continue  # Skip if no data matches this group code
        
        # Calculate fairness metrics for this group
        y_true_group = y_test.loc[group_data.index]
        y_pred_group = y_pred.loc[group_data.index]
        
        positive_rate, accuracy, precision, recall, fpr, dp_diff, eod, di = calculate_fairness_metrics(
            y_true_group, y_pred_group, overall_ppr, overall_recall
        )
        
        real_label = decode_group_code(group_code, mappings, sensitive_attributes)

        results.append({
            'Group_code_code': group_code,
            'Group_code': real_label,
            'Group Size': len(group_data),
            'Positive Prediction Rate': positive_rate,
            'Demographic Parity Difference': dp_diff,
            'Equal Opportunity Difference': eod,
            'Disparate Impact': di
        })
    
    # Convert results to DataFrame
    fairness_df = pd.DataFrame(results)
    return fairness_df

def run_fairness_analysis(dataset_name, transformed=False):
    
    if transformed:
        path = 'transformed'
    else:
        path = 'processed'
    
    # Load data
    X_test = pd.read_csv(f'data/{path}/X_test_{dataset_name}.csv')
    y_test = pd.read_csv(f'data/{path}/y_test_{dataset_name}.csv').squeeze()
    y_pred = pd.read_csv(f'data/{path}/{dataset_name}_y_pred.csv').squeeze()

    # Load frequent itemsets
    frequent_itemsets = pd.read_csv(f'results/{dataset_name}/frequent_itemsets.csv')

    # Load category mappings
    mappings = load_category_mappings(dataset_name)
    
    if dataset_name == 'compas':
        sensitive_attributes = ['race', 'sex', 'age_cat']
    elif dataset_name == 'census':
        sensitive_attributes = ['Race', 'Gender', 'age_cat']
    elif dataset_name == 'credit':
        sensitive_attributes = ['sex', 'age_cat']
    
    # Perform fairness analysis with group code matching
    fairness_df = fairness_analysis_with_group_code(X_test, y_test, y_pred, mappings, sensitive_attributes, frequent_itemsets)

    # Save the fairness analysis results
    os.makedirs(f'results/{dataset_name}', exist_ok=True)

    if transformed:
        fairness_df.to_csv(f'results/{dataset_name}/fairness_analysis_transformed.csv', index=False)
    else:    
        fairness_df.to_csv(f'results/{dataset_name}/fairness_analysis.csv', index=False)
    
    print(f"Fairness analysis results saved - {dataset_name} | {path}")

if __name__ == "__main__":
    run_fairness_analysis('compas')
    run_fairness_analysis('census')
    run_fairness_analysis('credit')
    
    run_fairness_analysis('compas', True)
    run_fairness_analysis('census', True)
    run_fairness_analysis('credit', True)
