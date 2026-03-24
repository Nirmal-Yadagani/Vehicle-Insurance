# 🚗 Vehicle Insurance Subscription Predictor

![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)
![AWS](https://img.shields.io/badge/AWS-EC2%20%7C%20S3%20%7C%20ECR-FF9900?logo=amazonaws)
![MongoDB](https://img.shields.io/badge/Database-MongoDB%20Atlas-47A248?logo=mongodb)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=github-actions)

## 📌 Project Overview
This project is an end-to-end Machine Learning Operations (MLOps) pipeline designed to predict whether a customer will subscribe to vehicle insurance. It utilizes a modular, component-based architecture to ensure scalability, easy maintenance, and seamless continuous integration and deployment (CI/CD).

## 🏗️ Architecture & Pipeline Flow
The project is strictly structured into modular pipelines:

1. **Data Ingestion:** Connects to MongoDB Atlas, fetches raw data in key-value pairs, converts it to a pandas DataFrame, and splits it into `train.csv` and `test.csv`.
2. **Data Validation:** Validates the ingested data against a predefined schema (`schema.yaml`) to check for data drift and anomalies.
3. **Data Transformation:** Applies feature engineering and preprocessing, saving the transformed arrays (`.npy`) and transformation objects (`preprocessing.pkl`).
4. **Model Trainer:** Trains the machine learning model, evaluates its performance, and saves the best model object (`model.pkl`).
5. **Model Evaluation & Pusher:** Compares the newly trained model's accuracy against the active model stored in **AWS S3**. If the new model beats the defined threshold (e.g., `0.02` improvement), it is pushed to the S3 model registry.
6. **Prediction Pipeline:** A Flask-based web application (`app.py`) that serves the model for real-time predictions.

## 🚀 Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **Database:** MongoDB Atlas
* **Cloud Platform:** Amazon Web Services (AWS)
    * **S3:** Model Registry
    * **ECR:** Docker Image Repository
    * **EC2:** Production Application Hosting (Ubuntu)
* **DevOps:** Docker, GitHub Actions (Self-Hosted Runner)

## 💻 Local Setup & Installation

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/vehicle-insurance.git](https://github.com/yourusername/vehicle-insurance.git)
cd vehicle-insurance
```

**2. To create virtual environment and install dependencies (UV is modern python package manager its much faster than pip)**
```bash
uv sync
```

**3. Set up Environment Variables**
You will need to set up your MongoDB connection string and AWS credentials.

For Bash/Linux/macOS:

```bash
export MONGODB_URL="mongodb+srv://<username>:<password>@cluster..."
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_DEFAULT_REGION="us-east-1"
```

For PowerShell/Windows:

```powerShell
$env:MONGODB_URL="mongodb+srv://<username>:<password>@cluster..."
$env:AWS_ACCESS_KEY_ID="your_access_key"
$env:AWS_SECRET_ACCESS_KEY="your_secret_key"
$env:AWS_DEFAULT_REGION="us-east-1"
```

**4. Train and save model to S3 bucket**
```bash
uv run main.py
```

**5. Run the application**

```bash
uv run app.py
```
Access the web UI at http://localhost:5000 or trigger the training pipeline via the /train route.

⚙️ CI/CD Deployment
This project features an automated deployment pipeline triggered on pushes to the main branch.

Build: A Docker image is built from the source code.

Push: The image is pushed to AWS Elastic Container Registry (ECR).

Deploy: A GitHub Actions self-hosted runner on an AWS EC2 instance pulls the latest ECR image and serves it on port 5000.

To access the live production server, navigate to: http://<EC2-PUBLIC-IP>:5000

<details>
<summary>📂 View Project Directory Structure</summary>

```text
.
├── app.py
├── artifact/
├── config/
│   ├── model.yaml
│   └── schema.yaml
├── Dockerfile
├── LICENSE
├── main.py
├── notebook/
├── pyproject.toml
├── src/
│   ├── cloud_storage/
│   ├── components/
│   ├── configuration/
│   ├── constants/
│   ├── data_access/
│   ├── entity/
│   ├── exception/
│   ├── logger/
│   ├── pipline/
│   └── utils/
├── static/
└── templates/
```
</details>