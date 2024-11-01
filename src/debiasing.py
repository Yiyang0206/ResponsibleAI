import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # Disable TensorFlow 2.x behavior to use session-based execution
import aif360
from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.datasets import StandardDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

def create_intersectional_attribute(df, sensitive_attributes):
    """
    Combine multiple sensitive attributes into a single intersectional attribute.
    """
    df['intersectional_attr'] = df[sensitive_attributes].astype(str).agg('_'.join, axis=1)
    return df

def adversarial_debiasing(dataset_name, sensitive_attributes):
    """
    Applies Adversarial Debiasing to a dataset with intersectional attributes.
    """
    # Define label and privileged combinations
    if dataset_name == "census":
        label = "Income"
        favorable_class = [1]
        privileged_groups = [{'intersectional_attr': '4_0_1'}]
        privileged_classes = [['4_0_1']]
        unprivileged_groups = [{'intersectional_attr': '4_0_2'}]
    elif dataset_name == "compas":
        label = "two_year_recid"
        favorable_class = [0]
        privileged_groups = [{'intersectional_attr': '0_1_2'}]
        privileged_classes = [['0_1_2']]
        unprivileged_groups = [{'intersectional_attr': '2_0_1'}]
    elif dataset_name == "credit":
        label = "Risk"
        favorable_class = [1]
        privileged_groups = [{'intersectional_attr': '3_1'}]
        privileged_classes = [['3_1']]
        unprivileged_groups = [{'intersectional_attr': '1_0'}]

    # Load original dataset
    df = pd.read_csv(f'data/processed/{dataset_name}_processed.csv')

    # Create an intersectional attribute
    df = create_intersectional_attribute(df, sensitive_attributes)

    # Split the data into features (X) and label (y)
    X = df.drop(columns=[label])
    y = df[label]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pd.DataFrame(X_train).to_csv('data/debiased/X_train_census.csv', index=False)
    pd.DataFrame(X_test).to_csv('data/debiased/X_test_census.csv', index=False)
    pd.DataFrame(y_train).to_csv('data/debiased/y_train_census.csv', index=False)
    pd.DataFrame(y_test).to_csv('data/debiased/y_test_census.csv', index=False)
    
    # Combine X_train and y_train into a DataFrame for AIF360 dataset
    train_df = X_train.copy()
    train_df[label] = y_train
    test_df = X_test.copy()
    test_df[label] = y_test

    # Convert to AIF360 StandardDataset
    dataset_train = StandardDataset(train_df, label_name=label, favorable_classes=favorable_class,
                                    protected_attribute_names=['intersectional_attr'],
                                    privileged_classes=privileged_classes)

    dataset_test = StandardDataset(test_df, label_name=label, favorable_classes=favorable_class,
                                   protected_attribute_names=['intersectional_attr'],
                                   privileged_classes=privileged_classes)

    # Adversarial Debiasing Model
    sess = tf.Session()  # Create a session for TensorFlow 1.x compatibility
    debiasing_model = AdversarialDebiasing(privileged_groups=privileged_groups,
                                           unprivileged_groups=unprivileged_groups,
                                           scope_name='debias_classifier', debias=True,
                                           sess=sess)

    # Train the model
    debiasing_model.fit(dataset_train)

    # Make predictions
    pred_dataset = debiasing_model.predict(dataset_test)

    # Convert predictions to pandas DataFrame for evaluation
    y_pred = pred_dataset.labels
    y_test = dataset_test.labels

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"Adversarial Debiasing - {dataset_name} Results:")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    pd.DataFrame(y_pred, columns=['y_pred']).to_csv(f'data/debiased/{dataset_name}_y_pred.csv', index=False)

    return debiasing_model, y_pred

# Example usage for Census dataset with intersectional bias (Race + Gender + age_cat)
adversarial_debiasing('census', ['Race', 'Gender', 'age_cat'])
# adversarial_debiasing('compas', ['race', 'sex', 'age_cat'])
# adversarial_debiasing('credit', ['sex', 'age_cat'])
