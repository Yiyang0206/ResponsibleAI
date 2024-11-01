import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import json

def categorize_age(census_df, bins=None, labels=None):
    """Convert numerical age into categorical age bins."""
    if bins is None:
        bins = [0, 25, 45, 100]  # Modify the bins as needed
    if labels is None:
        labels = ['Less than 25', '25 - 45', 'Greater than 45']  # Labels corresponding to bins

    # Create a new age category column
    census_df['age_cat'] = pd.cut(census_df['Age'], bins=bins, labels=labels, include_lowest=True)
    return census_df

def save_category_mappings(census_df, output_file='data/mappings/census_mappings.json'):
    """Save the original category mappings for sensitive attributes, including age categories, to a JSON file."""
    sensitive_columns = ['Race', 'Gender', 'age_cat']
    
    encoders = {}
    for column in sensitive_columns:
        le = LabelEncoder()
        census_df[column] = le.fit_transform(census_df[column])
        encoders[column] = {str(class_label): int(encoded_val) for class_label, encoded_val in zip(le.classes_, le.transform(le.classes_))}

    # Save the mappings for sensitive attributes to a JSON file
    with open(output_file, 'w') as json_file:
        json.dump(encoders, json_file, indent=4)
    
    print(f"Category mappings, including age, saved to {output_file}")

def load_and_preprocess_census():
    """Load and preprocess the Census Income dataset."""
    
    # Load the dataset
    census_df = pd.read_csv(file_path)

    # Select only the relevant columns
    selected_columns = [
        'Age', 'Workclass', 'Education', 'Marital Status', 'Occupation', 'Relationship', 'Race', 'Gender', 
        'Capital Gain', 'capital loss', 'Hours per Week', 'Native Country', 'Income'
    ]
    census_df = census_df[selected_columns]
    
    # Encode the target column 'Income' to 0 and 1
    income_mapping = {' <=50K': 0, ' >50K': 1}
    census_df['Income'] = census_df['Income'].map(income_mapping)

    # Categorize Age into bins
    census_df = categorize_age(census_df)

    # Save the category mappings for sensitive attributes
    save_category_mappings(census_df)

    # Impute missing values for numerical columns with mean
    numerical_columns = ['Age', 'Capital Gain', 'capital loss', 'Hours per Week']
    imputer_num = SimpleImputer(strategy='mean')
    census_df[numerical_columns] = imputer_num.fit_transform(census_df[numerical_columns])

    # Impute missing values for categorical columns with the most frequent value
    categorical_columns = ['Workclass', 'Education', 'Marital Status', 'Occupation', 'Relationship', 'Race', 'Gender', 'Native Country']
    imputer_cat = SimpleImputer(strategy='most_frequent')
    census_df[categorical_columns] = imputer_cat.fit_transform(census_df[categorical_columns])

    # Encode categorical columns using Label Encoding
    le = LabelEncoder()
    for column in categorical_columns:
        census_df[column] = le.fit_transform(census_df[column])

    # Scale numerical columns
    scaler = StandardScaler()
    census_df[numerical_columns] = scaler.fit_transform(census_df[numerical_columns])

    return census_df

def split_data(census_df):
    """Split the dataset into training and testing sets."""
    X = census_df.drop('Income', axis=1)  # Features
    y = census_df['Income']               # Target

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test