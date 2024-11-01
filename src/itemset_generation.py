import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
import json

def load_category_mappings(dataset_name):
    """Load the category mappings for the dataset (e.g., race, sex, age_cat for COMPAS or Census)."""
    mapping_file = f'data/mappings/{dataset_name}_mappings.json'
    with open(mapping_file, 'r') as file:
        mappings = json.load(file)
    return mappings

def decode_sensitive_attributes(df, mappings, sensitive_attributes):
    """Convert encoded sensitive attributes back to their original category names."""
    for attr in sensitive_attributes:
        mapping = mappings[attr]
        reverse_mapping = {v: k for k, v in mapping.items()}  # Reverse the mapping (int -> category)
        df[attr] = df[attr].map(reverse_mapping)  # Map back to original category names
    return df

def preprocess_sensitive_attributes(df, sensitive_attributes):
    """Preprocess sensitive attributes for frequent itemset generation."""
    df_sensitive = df[sensitive_attributes].apply(lambda x: x.astype(str))
    
    te = TransactionEncoder()
    transactions = df_sensitive.values.tolist()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    return df_encoded

def find_frequent_itemsets(df_sensitive, min_support=0.02):
    """Apply the Apriori algorithm to find frequent itemsets and filter out sets with fewer than 2 valid attributes."""
    frequent_itemsets = apriori(df_sensitive, min_support=min_support, use_colnames=True)
    
    # Filter out itemsets with fewer than 2 valid (non-NA, non-empty) attributes
    frequent_itemsets = frequent_itemsets[
        frequent_itemsets['itemsets'].apply(lambda x: len([item for item in x if item != 'NA' and item != '']) > 1)
    ]
    
    return frequent_itemsets

def split_itemsets(itemset, sensitive_attributes, df):
    """Split the itemsets into separate columns for each sensitive attribute."""
    itemset_dict = {attr: '' for attr in sensitive_attributes}  # Initialize all as empty
    for item in itemset:
        # Assign the item to the correct sensitive attribute based on a match, skipping NA or empty values
        for attr in sensitive_attributes:
            if item != 'NA' and item != '' and item in df[attr].unique():
                itemset_dict[attr] = item
    return itemset_dict


def generate_group_code_from_mapping(itemset_dict, mappings, sensitive_attributes):
    """Generate a group code using the encoded (mapped) values of the sensitive attributes."""
    group_code = []
    for attr in sensitive_attributes:
        if itemset_dict[attr] != '':  # Only process non-empty values
            encoded_value = mappings[attr].get(itemset_dict[attr], 'NA')
        else:
            encoded_value = 'NA'
        group_code.append(str(encoded_value))  # Use mapped value or 'NA' if missing
    return '_'.join(group_code)

def save_frequent_itemsets(dataset_name, min_support):
    df = pd.read_csv(f'data/processed/{dataset_name}_processed.csv')
    mappings = load_category_mappings(dataset_name)
    
    if dataset_name == 'compas':
        sensitive_attributes = ['race', 'sex', 'age_cat']
    elif dataset_name == 'census':
        sensitive_attributes = ['Race', 'Gender', 'age_cat']
    elif dataset_name == 'credit':
        sensitive_attributes = ['sex', 'age_cat']
    
    df = decode_sensitive_attributes(df, mappings, sensitive_attributes)
    df_sensitive = preprocess_sensitive_attributes(df, sensitive_attributes)
    frequent_itemsets = find_frequent_itemsets(df_sensitive, min_support=min_support)
    
    # Calculate group size
    total_rows = len(df)
    frequent_itemsets['group_size'] = (frequent_itemsets['support'] * total_rows).astype(int)
    
    # Split the itemsets into separate columns for each sensitive attribute
    split_data = frequent_itemsets['itemsets'].apply(lambda x: pd.Series(split_itemsets(x, sensitive_attributes, df)))
    
    # Generate a group code using the encoded values from the mappings
    frequent_itemsets['group_code'] = split_data.apply(lambda row: generate_group_code_from_mapping(row, mappings, sensitive_attributes), axis=1)
    
    # Combine the split columns with support, group_size, and group_code
    frequent_itemsets = pd.concat([frequent_itemsets[['support', 'group_size', 'group_code']], split_data], axis=1)
    
    print(f"Frequent itemsets with group size, group code (mapped values), and split attributes:\n{frequent_itemsets}")
    
    # Save to CSV
    frequent_itemsets.to_csv(f'results/{dataset_name}/frequent_itemsets.csv', index=False)

# if __name__ == "__main__":
#     save_frequent_itemsets('compas', min_support = 0.02)
#     save_frequent_itemsets('census', min_support = 0.02)
#     save_frequent_itemsets('credit', min_support = 0.02)

