import pandas as pd
from prefect import flow, task
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- Task 1: Load and Preprocess ---
@task
def load_and_preprocess_data(file_path: str):
    """Loads and preprocesses the credit card data."""
    print(f"Loading and preprocessing data from {file_path}...")
    df = pd.read_csv(file_path)

    # Scale Time and Amount
    scaler = StandardScaler()
    df['scaled_Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    df = df.drop(['Time', 'Amount'], axis=1)

    print("Data loaded and preprocessed.")
    return df

# --- Task 2: Split Data ---
@task
def split_data(df: pd.DataFrame):
    """Splits data into 70% train and 30% test sets."""
    print("Splitting data into X and y...")
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Split data: 70% train, 30% test
    # We use 'stratify=y' to make sure both train and test sets
    # get the same tiny percentage of fraud cases. This is critical!
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print("Data split complete.")
    return X_train, X_test, y_train, y_test

# --- Task 3: Train Model ---
@task
def train_model(X_train, y_train, model):
    """Trains a given machine learning model."""
    print(f"Training model: {model.__class__.__name__}...")
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model

# --- Task 4: Evaluate Model (MLOps Logging) ---
@task
def evaluate_model(model, X_test, y_test):
    """Evaluates the model and logs the 4 required metrics."""
    model_name = model.__class__.__name__
    print(f"Evaluating model: {model_name}...")

    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # This is the logging step for Sub-objective 2.4
    # Prefect automatically captures all 'print' statements as logs.
    print(f"--- {model_name} Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("---------------------------------")

# --- Define Your ML Flow ---
@flow(name="ML Training and Evaluation Pipeline")
def ml_pipeline_flow():
    """
    The main flow to train and evaluate two ML models.
    """
    # --- Algorithm 1: Logistic Regression ---
    df = load_and_preprocess_data(file_path='creditcard.csv')
    X_train, X_test, y_train, y_test = split_data(df)

    lr_model = LogisticRegression(random_state=42)
    trained_lr_model = train_model(X_train, y_train, lr_model)
    evaluate_model(trained_lr_model, X_test, y_test)

    # --- Algorithm 2: Random Forest ---
    # Note: We can re-use the split data
    # rf_model = RandomForestClassifier(random_state=42)
    rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    trained_rf_model = train_model(X_train, y_train, rf_model)
    evaluate_model(trained_rf_model, X_test, y_test)

# --- Deploy the Flow ---
if __name__ == "__main__":
    ml_pipeline_flow.serve(name="ml-pipeline-deployment")