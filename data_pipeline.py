import pandas as pd
from sklearn.preprocessing import StandardScaler
from prefect import flow, task

# --- Define Your Tasks ---
# A task is a single, observable step in your pipeline.

@task
def load_data(file_path: str):
    """Loads the CSV file from the given path."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows.")
    return df

@task
def preprocess_data(df: pd.DataFrame):
    """Scales the 'Time' and 'Amount' columns."""
    print("Preprocessing data: Scaling 'Time' and 'Amount'...")
    scaler = StandardScaler()

    df['scaled_Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

    df = df.drop(['Time', 'Amount'], axis=1)

    print("Preprocessing complete.")
    return df

# --- Define Your Flow ---
# A flow is the main function that orchestrates your tasks.

@flow(name="Data Processing Pipeline (Fraud Detection)")
def data_pipeline_flow():
    """
    The main flow to load and preprocess the credit card data.
    """
    # Call your tasks in order
    df = load_data(file_path='creditcard.csv')
    preprocessed_df = preprocess_data(df)

    # In a real project, you would save this preprocessed data.
    # For this assignment, just printing is fine.
    print("Data pipeline finished successfully!")
    print("--- First 5 rows of preprocessed data ---")
    print(preprocessed_df.head())


# --- Run the Flow ---
# This 'if' block makes the script runnable
# if __name__ == "__main__":
#    data_pipeline_flow()

    # --- Deploy the Flow ---
# This 'if' block makes the script runnable as a deployment
if __name__ == "__main__":
    # .serve() creates a deployment that connects to Prefect Cloud
    # and starts a worker to listen for flow run requests.
    # data_pipeline_flow.serve(name="data-pipeline-deployment")
    data_pipeline_flow.serve(name="data-pipeline-deployment",
                         cron="*/2 * * * *")