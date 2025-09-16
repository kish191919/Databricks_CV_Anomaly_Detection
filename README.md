# Databricks_CV_Anomaly_Detection

## 👁️ Databricks + Computer Vision Anomaly Detection & Model Deployment  
_A complete guide to anomaly detection with Databricks and Apache Spark_  

> "From data ingestion to real-time serving — build and deploy scalable computer vision anomaly detection models."


![alt text](image.png)
---

### 📌 One-Line Summary
This project provides a full pipeline for **computer vision–based anomaly detection**, covering **data ingestion, preprocessing, model training, deployment, and REST API serving** — all within **Databricks** and powered by **Apache Spark**.

---

## 1️⃣ How It Was Built

### **1. Utilities ([00_utils.ipynb](https://github.com/kish191919/Databricks_CV_Anomaly_Detection/blob/master/Databricks_Code/00_utils.ipynb))**
- Common helper functions for preprocessing and visualization  
- Reusable utilities to streamline workflows  

---

### **2. Data Ingestion & ETL ([01_Ingestion_ETL.ipynb](https://github.com/kish191919/Databricks_CV_Anomaly_Detection/blob/master/Databricks_Code/01_Ingestion_ETL.ipynb))**
- Ingested large-scale image datasets into Databricks  
- Implemented Spark-based ETL for scalability  
- Optimized storage and partitioning for performance and cost efficiency  

---

### **3. Deep Learning Training ([02_HF_Deep_Learning.ipynb](https://github.com/kish191919/Databricks_CV_Anomaly_Detection/blob/master/Databricks_Code/02_HF_Deep_Learning.ipynb))**
- Applied image preprocessing and augmentation  
- Trained models using **PyTorch + Hugging Face**  
- Evaluated performance with metrics like **Accuracy**, **Loss**, and **PR-AUC**  

---

### **4. Model Deployment ([03_Model_Deployment.ipynb](https://github.com/kish191919/Databricks_CV_Anomaly_Detection/blob/master/Databricks_Code/03_Model_Deployment.ipynb))**
- Registered trained models in **MLflow**  
- Managed versions for reproducibility  
- Optimized inference pipelines for deployment  

---

### **5. Model Serving ([04_Model_Serving.ipynb](https://github.com/kish191919/Databricks_CV_Anomaly_Detection/blob/master/Databricks_Code/04_Model_Serving.ipynb))**
- Deployed models with **Databricks Model Serving**  
- Exposed REST API endpoints for real-time predictions  
- Integrated anomaly detection into external systems  

---

## 2️⃣ Optimization & Best Practices
- Spark optimizations for large-scale image data  
- Databricks cluster configuration for **cost efficiency**  
- Strategies for balancing performance and resource usage  

---

## 🛠 Technologies Used
| Step              | Technology                   |
|-------------------|------------------------------|
| Data Processing   | Apache Spark, Databricks     |
| Deep Learning     | PyTorch, Hugging Face        |
| Experiment Mgmt   | MLflow                       |
| Deployment        | Databricks Model Registry    |
| Serving           | REST API, Databricks Serving |

---

## 💡 Key Learnings
- Full lifecycle ML on Databricks: ingestion → training → deployment → serving  
- How to optimize Databricks for **low-cost, high-performance workflows**  
- Practical experience with model versioning, reproducibility, and API integration  
