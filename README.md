# API-Driven Cloud-Native Application

This project implements a complete Data Science and MLOps application for **Credit Card Fraud Detection**.

It leverages **Prefect 3** to create, orchestrate, and monitor two distinct, cloud-native pipelines:
1.  A **Data Pipeline** (`data_pipeline.py`) that runs on a 2-minute schedule to preprocess the raw data and save the result.
2.  An **ML Pipeline** (`ml_pipeline.py`) that loads the preprocessed data, trains two models (Logistic Regression and Random Forest), evaluates them using appropriate metrics for imbalanced data, and logs the results.

The system demonstrates an **API-driven architecture** through:
* **Prefect Cloud:** Centralized monitoring and control via its API.
* **Discord Bot Integration:** An external bot (`bot.py`) triggers the ML pipeline on-demand via an API call in response to a chat command.
* **API Consumption:** A script (`check_api.py`) uses the Prefect Python client to query deployment details directly from the API.

---

## Project Setup

Follow these steps to set up the project locally.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/nrbandi/apiproject1.git](https://github.com/nrbandi/apiproject1.git)
    cd apiproject1
    ```

2.  **Create Python Environment:**
    ```bash
    python3 -m venv apienv
    source apienv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip3 install -r requirements.txt
    ```

4.  **Set Up Kaggle API:**
    * Go to your Kaggle account settings ([`https://www.kaggle.com/account`](https://www.kaggle.com/account)) and download your `kaggle.json` API token.
    * Place the token in the correct location:
        ```bash
        mkdir -p ~/.kaggle
        mv ~/Downloads/kaggle.json ~/.kaggle/
        chmod 600 ~/.kaggle/kaggle.json
        ```

5.  **Download Dataset:**
    ```bash
    kaggle datasets download -d mlg-ulb/creditcardfraud
    unzip creditcardfraud.zip
    # This creates creditcard.csv in your project folder
    ```

6.  **Create `.env` File for Secrets:**
    * Create a file named `.env` in the root of the `apiproject1` directory.
    * Add your Discord Bot Token to this file. **This file is ignored by Git and should NOT be committed.**
        ```dotenv
        # Contents of .env file
        DISCORD_BOT_TOKEN="YOUR_DISCORD_BOT_TOKEN_HERE"
        ```

7.  **Set Up Discord Bot (Optional - for API Trigger):**
    * Go to the [Discord Developer Portal](https://discord.com/developers/applications/).
    * Create a "New Application" (e.g., `MyAPIBot`).
    * Go to the "Bot" tab, "Reset Token," and copy the token into your `.env` file (see step 6).
    * **Enable the "MESSAGE CONTENT INTENT"** under "Privileged Gateway Intents."
    * Go to "OAuth2" -> "URL Generator," select `bot` scope, and grant `Send Messages` & `Read Message History` permissions. Use the generated URL to invite the bot to your Discord server.

8.  **Connect to Prefect Cloud:**
    ```bash
    prefect cloud login
    prefect cloud workspace set
    ```

---

## How to Run

You need **three separate terminals** running simultaneously. Ensure you `cd apiproject1` and `source apienv/bin/activate` in each one.

1.  **Terminal 1: Data Pipeline Worker**
    ```bash
    python3 data_pipeline.py
    ```
    *(This worker runs the scheduled data processing every 2 minutes and creates `preprocessed_creditcard.csv`.)*

2.  **Terminal 2: ML Pipeline Worker**
    ```bash
    python3 ml_pipeline.py
    ```
    *(This worker waits for jobs triggered via the UI or the Discord bot.)*

3.  **Terminal 3: Discord Bot Worker**
    ```bash
    python3 bot.py
    ```
    *(This worker logs into Discord and listens for the `!run-pipeline` command.)*

---

## Testing API-Driven Features

1.  **Trigger via Discord:**
    * Go to your Discord server/channel where the bot is.
    * Type the message: `!run-pipeline`
    * Observe Terminal 2 as the ML pipeline starts.

2.  **Check Application Details via API:**
    * Open a **fourth terminal**.
    * `cd apiproject1`
    * `source apienv/bin/activate`
    * Run the script: `python3 check_api.py`
    * Observe the output showing details for both deployments retrieved via the Prefect API.

---

## Model Performance

Metrics captured from the `ML Training and Evaluation Pipeline` logs in Prefect:

| Model                | Accuracy | Precision | Recall | F1 Score |
| :------------------- | :------- | :-------- | :----- | :------- |
| Logistic Regression  | 0.9992   | 0.8585    | 0.6149 | 0.7165   |
| Random Forest        | 0.9995   | 0.9492    | 0.7568 | 0.8421   |

*(Note: Random Forest performance significantly improved by using `n_jobs=-1` for parallel processing using all processors in the system)*

---