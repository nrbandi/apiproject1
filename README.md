# CCZG506: API-Driven Cloud-Native Application

This project is a complete Data Science and MLOps application for **Credit Card Fraud Detection**, built for the CCZG506 assignment.

It uses **Prefect** to create and monitor two cloud-native pipelines:
1.  A **Data Pipeline** that runs on a 2-minute schedule to process data.
2.  An **ML Pipeline** that trains and evaluates two models (Logistic Regression and Random Forest) to detect fraud.

## How to Run

1.  **Clone the repo and set up the environment:**
    ```bash
    git clone https://github.com/nrbandi/apiproject1.git
    cd apiproject1
    python3 -m venv apienv
    source apienv/bin/activate
    pip3 install -r requirements.txt 
    ```
2.  **Log in to Prefect Cloud:**
    ```bash
    prefect cloud login
    prefect cloud workspace set -n "default"
    ```
3.  **Run the workers (in two separate terminals):**
    ```bash
    # In Terminal 1
    python3 data_pipeline.py

    # In Terminal 2
    python3 ml_pipeline.py
    ```
4.  **Run the API check script:**
    ```bash
    python3 check_api.py
    ```

## Model Performance

Metrics from the last ML pipeline run:

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.9992 | 0.8585 | 0.6149 | 0.7165 |
| Random Forest | 0.9995 | 0.9492 | 0.7568 | 0.8421 |
