from ucimlrepo import fetch_ucirepo
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def categorize_age(df):
    """Convert numerical age into categorical age bins."""
    bins = [0, 25, 45, 100]  # Define age bins
    labels = ['Less than 25', '25 - 45', 'Greater than 45']
    df['age_cat'] = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True)
    return df

def save_category_mappings(df, output_file='data/mappings/credit_mappings.json'):
    """Save the category mappings for sensitive attributes, including 'sex' and 'age_cat'."""
    sex_mapping = {
        'A91': 0,  # male: divorced/separated
        'A92': 1,  # female: divorced/separated/married
        'A93': 2,  # male: single
        'A94': 3,  # male: married/widowed
    }

    # Create mapping for age categories
    age_mapping = {
        'Less than 25': 0,
        '25 - 45': 1,
        'Greater than 45': 2
    }

    # Apply the mappings to the dataframe
    df['sex'] = df['sex'].map(sex_mapping)
    df['age_cat'] = df['age_cat'].map(age_mapping)

    sex_mapping = {
        'male_divorced_separated': 0,
        'female_divorced_separated_married': 1,
        'male_single': 2,
        'male_married_widowed': 3,
    }
    
    # Save the mappings to a JSON file
    mappings = {
        'sex': sex_mapping,
        'age_cat': age_mapping
    }

    with open(output_file, 'w') as json_file:
        json.dump(mappings, json_file, indent=4)
    
    print(f"Category mappings saved to {output_file}")
    return df

def load_and_preprocess_german_credit():
    """Fetch and preprocess the German Credit dataset."""
    # Fetch dataset
    german_credit_data = fetch_ucirepo(id=144)
    
    # Extract features (X) and target (y)
    X = german_credit_data.data.features
    y = german_credit_data.data.targets

    # Rename attributes based on their actual meaning
    X = X.rename(columns={
        'Attribute1': 'Status_of_checking_account',
        'Attribute2': 'Duration_in_month',
        'Attribute3': 'Credit_history',
        'Attribute4': 'Purpose',
        'Attribute5': 'Credit_amount',
        'Attribute6': 'Savings_account_bonds',
        'Attribute7': 'Present_employment_since',
        'Attribute8': 'Installment_rate',
        'Attribute9': 'sex',  # This is the 'Personal status and sex' attribute
        'Attribute10': 'Other_debtors',
        'Attribute11': 'Present_residence_since',
        'Attribute12': 'Property',
        'Attribute13': 'Age',
        'Attribute14': 'Other_installment_plans',
        'Attribute15': 'Housing',
        'Attribute16': 'Number_of_existing_credits',
        'Attribute17': 'Job',
        'Attribute18': 'Number_of_people_liable',
        'Attribute19': 'Telephone',
        'Attribute20': 'Foreign_worker',
        'class': 'Risk'
    })
    
    # Add target column 'class' as 'Risk' for clarity
    risk_mapping = {1: 1, 2: 0}  # 1 = Good, 2 = Bad
    X['Risk'] = y
    X['Risk'] = X['Risk'].map(risk_mapping)

    # Categorize age into bins
    X = categorize_age(X)

    # Process and save category mappings
    X = save_category_mappings(X)
    
    numerical_columns = ['Duration_in_month', 'Credit_amount', 'Installment_rate', 'Age']
    categorical_columns = ['Status_of_checking_account', 'Credit_history', 'Purpose', 'Savings_account_bonds', 
                           'Present_employment_since', 'Other_debtors', 'Present_residence_since', 
                           'Property', 'Other_installment_plans', 'Housing', 'Job', 'Telephone', 'Foreign_worker', 
                           'age_cat']
    
    # Encode categorical columns using Label Encoding
    le = LabelEncoder()
    for column in categorical_columns:
        X[column] = le.fit_transform(X[column])

    # Scale numerical columns
    scaler = StandardScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

    return X

def split_data(df):
    """Split the German Credit dataset into training and testing sets."""
    X = df.drop('Risk', axis=1)  # Features
    y = df['Risk']               # Target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
