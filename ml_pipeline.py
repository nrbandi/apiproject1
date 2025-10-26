import os
import pandas as pd
import requests
import json
from dotenv import load_dotenv
from prefect import flow, task
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

load_dotenv()

@task
def send_discord_notification(metrics_log: str):
    """Sends a notification to a Discord webhook."""
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url: # Check if it loaded correctly
        print("!!! ERROR: DISCORD_WEBHOOK_URL not found in .env file.")
        return

    data = {
        "content": "🚀 ML Pipeline Run COMPLETE!",
        "embeds": [{
            "title": "Model Performance Metrics",
            "description": metrics_log
        }]
    }

    try:
        response = requests.post(
            webhook_url, data=json.dumps(data),
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status() # Raise an error for bad responses
        print("Successfully sent Discord notification.")
    except Exception as e:
        print(f"Error sending Discord notification: {e}")

# --- Main Task 1: Load Preprocess Data---
@task
def load_preprocessed_data(file_path: str): # Renamed for clarity
    """Loads the preprocessed CSV file."""
    print(f"Loading preprocessed data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows of preprocessed data.")
    return df

# --- Task 2: Split Data ---
@task
def split_data(df: pd.DataFrame, test_size: float):
    """Splits data into train and test sets."""
    print(f"Splitting data with test_size={test_size}...")
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Split data based  on the test_size parameter
    # We use 'stratify=y' to make sure both train and test sets
    # get the same tiny percentage of fraud cases. This is critical!
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
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
    # Create a log string
    log_message = (
        f"--- {model_name} Metrics ---\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1 Score: {f1:.4f}\n"
        "---------------------------------"
    )
    print(log_message) # Keep printing for the dashboard
    return log_message # Return the string

# --- Define Your ML Flow ---
@flow(name="ML Training and Evaluation Pipeline")
def ml_pipeline_flow(test_size: float = 0.3, file_path: str = 'preprocessed_creditcard.csv'):
    """
    The main flow to train and evaluate two ML models.
    'test_size' and 'file_path' are parameters that can be set from the UI.
    """
    print(f"Starting ML pipeline with test_size = {test_size} and file_path = {file_path}")

    # --- Algorithm 1: Logistic Regression ---
    df = load_preprocessed_data(file_path=file_path)
    X_train, X_test, y_train, y_test = split_data(df, test_size)

    lr_model = LogisticRegression(random_state=42)
    trained_lr_model = train_model(X_train, y_train, lr_model)
    # Capture the logs
    lr_logs = evaluate_model(trained_lr_model, X_test, y_test)

    # --- Algorithm 2: Random Forest ---
    # Note: We can re-use the split data
    # rf_model = RandomForestClassifier(random_state=42)
    rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    trained_rf_model = train_model(X_train, y_train, rf_model)
    rf_logs = evaluate_model(trained_rf_model, X_test, y_test)

    # Send one final notification with all metrics
    final_log = f"{lr_logs}\n{rf_logs}"
    send_discord_notification(final_log)

# --- Deploy the Flow ---
if __name__ == "__main__":
    ml_pipeline_flow.serve(name="ml-pipeline-deployment")