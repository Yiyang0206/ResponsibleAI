import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import json

def save_category_mappings(compas_df, output_file='data/mappings/compas_mappings.json'):
    """Save the original category mappings for race, sex, and age category to a JSON file."""
    # Initialize LabelEncoder
    le_race = LabelEncoder()
    le_sex = LabelEncoder()
    le_age_cat = LabelEncoder()  # Adding age category

    # Fit the LabelEncoders on the raw data
    le_race.fit(compas_df['race'])
    le_sex.fit(compas_df['sex'])
    le_age_cat.fit(compas_df['age_cat'])  # Fit on age_cat

    # Get the mappings for race, sex, and age_cat
    race_mapping = dict(zip(map(str, le_race.classes_), map(int, le_race.transform(le_race.classes_))))
    sex_mapping = dict(zip(map(str, le_sex.classes_), map(int, le_sex.transform(le_sex.classes_))))
    age_cat_mapping = dict(zip(map(str, le_age_cat.classes_), map(int, le_age_cat.transform(le_age_cat.classes_))))  # Mapping for age_cat

    # Combine all mappings into a dictionary
    mappings = {
        "race": race_mapping,
        "sex": sex_mapping,
        "age_cat": age_cat_mapping  # Adding age_cat to the mappings
    }

    # Save the mappings to a JSON file
    with open(output_file, 'w') as json_file:
        json.dump(mappings, json_file, indent=4)

    print(f"Category mappings saved to {output_file}")

def load_and_preprocess_compas(file_path="responsible_ai_thesis/data/raw/compas.csv"):
    """Load and preprocess the COMPAS dataset, including additional sensitive attributes."""
    
    # Load the dataset
    compas_df = pd.read_csv(file_path)

    # Select only the relevant columns (including age_cat)
    selected_columns = [
        'sex', 'race', 'c_charge_degree', 'score_text', 'v_type_of_assessment', 'age_cat',
        'age', 'priors_count', 'juv_fel_count', 'decile_score', 'days_b_screening_arrest', 'type_of_assessment',
        'two_year_recid'
    ]
    compas_df = compas_df[selected_columns]

    # Save the category mappings including age_cat
    save_category_mappings(compas_df)

    # Impute missing values for numerical columns with mean
    numerical_columns = ['age', 'priors_count', 'juv_fel_count', 'decile_score', 'days_b_screening_arrest']
    imputer_num = SimpleImputer(strategy='mean')
    compas_df[numerical_columns] = imputer_num.fit_transform(compas_df[numerical_columns])

    # Impute missing values for categorical columns with the most frequent value
    categorical_columns = ['sex', 'race', 'c_charge_degree', 'score_text', 'v_type_of_assessment', 'age_cat', 'type_of_assessment']
    imputer_cat = SimpleImputer(strategy='most_frequent')
    compas_df[categorical_columns] = imputer_cat.fit_transform(compas_df[categorical_columns])

    # Encode categorical columns using Label Encoding
    le = LabelEncoder()
    for column in categorical_columns:
        compas_df[column] = le.fit_transform(compas_df[column])

    # Scale numerical columns
    scaler = StandardScaler()
    compas_df[numerical_columns] = scaler.fit_transform(compas_df[numerical_columns])

    return compas_df

def split_data(compas_df):
    """Split the dataset into training and testing sets."""
    # Define the target and features
    X = compas_df.drop('two_year_recid', axis=1)  # Features
    y = compas_df['two_year_recid']               # Target

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
