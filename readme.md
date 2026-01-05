# MLOps End-to-End Machine Learning Pipeline

## Project Overview

This project demonstrates a complete **end-to-end MLOps workflow**, covering the entire lifecycle of a machine learning model — from data ingestion and preprocessing to model training, evaluation, versioning, and deployment readiness.

The objective of this project is to highlight **production-oriented machine learning practices**, focusing on reproducibility, modular design, experiment tracking, and deployment preparedness rather than only model accuracy.

This repository is suitable for:
- Academic demonstrations of MLOps concepts
- Teaching machine learning deployment workflows
- Industry-ready portfolio projects and interviews

---

## Key Objectives

- Design a structured and modular machine learning pipeline
- Apply best practices for experiment tracking and model versioning
- Ensure reproducibility and scalability
- Prepare models for real-world deployment scenarios
- Demonstrate practical implementation of MLOps concepts

---

## Project Workflow

The project follows a standard MLOps pipeline:

1. **Data Ingestion**
   - Load and validate raw datasets
   - Handle missing values and data inconsistencies

2. **Data Preprocessing**
   - Feature engineering
   - Encoding and normalization
   - Train-test split

3. **Model Training**
   - Train machine learning models
   - Hyperparameter tuning (if applicable)

4. **Model Evaluation**
   - Evaluate model performance using appropriate metrics
   - Compare different model versions

5. **Experiment Tracking and Versioning**
   - Track experiments, parameters, and metrics
   - Store trained models and artifacts

6. **Deployment Readiness**
   - Export finalized models
   - Separate training and inference logic

---

## Tech Stack

- Python
- NumPy
- Pandas
- Scikit-learn
- MLflow (optional)
- Docker (optional)
- FastAPI (optional)

---

## Repository Structure

```
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── models/
├── artifacts/
├── requirements.txt
└── README.md
```

---

## How to Run the Project

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd <repository-name>
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Train the Model
```bash
Jupyter Notebook and run the cells
```
### Step 4: Dockerize the Model
```bash
docker build -t heart-api:latest
```
### Step 5: Run the Container Model
```bash
docker run -p 8000:8000 heart-api:latest
```

### Step 6: Evaluate the Model
```bash
Navigate to localhost:8080/docs and check for swagger docs and for REST end points check the curl commands in curl.txt
```

---

## Demo Video

A complete walkthrough of the project, including code explanation and execution, is available below:

**Google Drive Demo Video:**  
<p><a href="https://drive.google.com/file/d/1kQOyEbTIoTND07u6Vxwaiqc8kMf9SgQn/view?usp=sharing">Video Link</a></p>

---

## Learning Outcomes

- Hands-on understanding of MLOps principles
- Experience in building production-ready ML pipelines
- Exposure to experiment tracking and model versioning
- Clear separation of data processing, training, and inference stages

---

## Future Enhancements

- Implement CI/CD pipelines for automated training and testing
- Add model monitoring and data drift detection
- Containerize the application using Docker
- Deploy the pipeline on cloud platforms such as AWS, Azure, or GCP

---

## License

This project is intended for educational and demonstration purposes.
