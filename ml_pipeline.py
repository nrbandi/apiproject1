import pandas as pd
import requests
import json
from prefect import flow, task
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

@task
def send_discord_notification(metrics_log: str):
    """Sends a notification to a Discord webhook."""
    webhook_url = "https://discord.com/api/webhooks/1430192759595335771/OZkP4g1aicCD5twjyLZmIcROfqa-tfBZ-eLe3TQ_Cf-KgO6YuGFGSXE9WlwWg1AeRLZb"

    if not webhook_url.startswith("http"):
        print("Discord webhook URL not set, skipping notification.")
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

# --- Main Task 1: Load and Preprocess ---
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
    # Prefect automatically captures all 'print' statements as logs.
    print(f"--- {model_name} Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("---------------------------------")

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
def ml_pipeline_flow(test_size: float = 0.3):
    """
    The main flow to train and evaluate two ML models.
    'test_size' is a parameter that can be set from the UI.
    """
    print(f"Starting ML pipeline with test_size = {test_size}")

    # --- Algorithm 1: Logistic Regression ---
    df = load_and_preprocess_data(file_path='creditcard.csv')
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