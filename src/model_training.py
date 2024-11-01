import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_neural_network(X_train, X_test, y_train, y_test):
    """Train a Neural Network model and evaluate it."""
    
    # Normalize/Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build a simple neural network model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2)
    
    # Make predictions on the test set
    y_pred_proba = model.predict(X_test_scaled)
    y_pred = (y_pred_proba > 0.5).astype(int)  # Convert probabilities to binary class labels (0 or 1)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Print evaluation metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return model, y_pred

def run_model_for_dataset(dataset_name, path):
    if dataset_name == 'compas':
        print("Loading and preprocessing COMPAS dataset...")
        X_train = pd.read_csv(f'{path}/X_train_compas.csv')
        X_test = pd.read_csv(f'{path}/X_test_compas.csv')
        y_train = pd.read_csv(f'{path}/y_train_compas.csv').squeeze()
        y_test = pd.read_csv(f'{path}/y_test_compas.csv').squeeze()
    elif dataset_name == 'census':
        print("Loading and preprocessing Census dataset...")
        X_train = pd.read_csv(f'{path}/X_train_census.csv')
        X_test = pd.read_csv(f'{path}/X_test_census.csv')
        y_train = pd.read_csv(f'{path}/y_train_census.csv').squeeze()
        y_test = pd.read_csv(f'{path}/y_test_census.csv').squeeze()
    elif dataset_name == 'credit':
        print("Loading and preprocessing German Credit dataset...")
        X_train = pd.read_csv(f'{path}/X_train_credit.csv')
        X_test = pd.read_csv(f'{path}/X_test_credit.csv')
        y_train = pd.read_csv(f'{path}/y_train_credit.csv').squeeze()
        y_test = pd.read_csv(f'{path}/y_test_credit.csv').squeeze()

    # Train and evaluate the neural network model
    trained_model, y_pred = train_neural_network(X_train, X_test, y_train, y_test)
    
    # Save predictions for fairness analysis
    pd.DataFrame(y_pred, columns=['y_pred']).to_csv(f'{path}/{dataset_name}_y_pred.csv', index=False)
    print(f"Saved predictions for {dataset_name} dataset.")

# if __name__ == "__main__":
#     # Process for non-transformed datasets
#     print("Training and evaluating models on non-transformed datasets...")
#     processed_path = 'data/processed'
#     run_model_for_dataset('compas', processed_path)
#     run_model_for_dataset('census', processed_path)
#     run_model_for_dataset('credit', processed_path)
    
#     # Process for transformed datasets
#     print("\nTraining and evaluating models on transformed datasets...")
#     transformed_path = 'data/transformed'
#     run_model_for_dataset('compas', transformed_path)
#     run_model_for_dataset('census', transformed_path)
#     run_model_for_dataset('credit', transformed_path)