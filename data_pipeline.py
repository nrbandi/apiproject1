import pandas as pd
from sklearn.preprocessing import StandardScaler
from prefect import flow, task, get_run_logger

# --- Define Tasks ---
@task
def load_data(file_path: str):
    """Loads the CSV file from the given path."""
    logger = get_run_logger() # <-- Use Prefect logger
    logger.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} rows.")
    return df

@task
def preprocess_data(df: pd.DataFrame):
    """Scales the 'Time' and 'Amount' columns."""
    logger = get_run_logger()
    logger.info("Preprocessing data: Scaling 'Time' and 'Amount'...")
    scaler = StandardScaler()

    df['scaled_Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

    df = df.drop(['Time', 'Amount'], axis=1)

    logger.info("Preprocessing complete.")
    return df

@task
def save_data(df: pd.DataFrame, output_path: str):
    """Saves the DataFrame to a CSV file."""
    logger = get_run_logger()
    logger.info(f"Saving preprocessed data to {output_path}...")
    df.to_csv(output_path, index=False)
    logger.info(f"Data saved successfully.")
    return output_path # <-- Return the path

# --- Define Your Flow ---
@flow(name="Data Processing Pipeline (Fraud Detection)")
def data_pipeline_flow(input_path: str = 'creditcard.csv', # <-- Parameterize input
                       output_path: str = 'preprocessed_creditcard.csv'): # <-- Parameterize output
    """
    The main flow to load, preprocess, and save the credit card data.
    'input_path' and 'output_path' are parameters.
    """
    logger = get_run_logger()

    # Call the tasks in order
    df = load_data(file_path=input_path)
    preprocessed_df = preprocess_data(df)

    # Save the result
    final_path = save_data(preprocessed_df, output_path) # <-- Call save task
    
    logger.info(f"Data pipeline finished successfully! Output at: {final_path}")
    # You can remove the print(head) if you want

# --- Run the Flow ---
if __name__ == "__main__":
    # .serve() creates a deployment that connects to Prefect Cloud
    # and starts a worker to listen for flow run requests.
    # data_pipeline_flow.serve(name="data-pipeline-deployment")
    # the parameter corn make it run every 2 mins
    data_pipeline_flow.serve(name="data-pipeline-deployment",
                         cron="*/2 * * * *")