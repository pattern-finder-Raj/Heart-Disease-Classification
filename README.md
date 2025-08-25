# ❤️ Heart Disease Prediction using Machine Learning

## 📋 Project Overview
This project aims to **predict whether a patient has heart disease or not** using machine learning algorithms.  
By analyzing health-related attributes such as age, blood pressure, cholesterol levels, and more, we develop models that classify patients into “Heart Disease” or “No Heart Disease”.

The project demonstrates the end-to-end **data science pipeline** — from data cleaning, feature engineering, model building, evaluation, and comparison of algorithms.

---

## 🎯 Problem Statement
Cardiovascular disease is one of the leading causes of death worldwide.  
The goal of this project is to build a **machine learning model** that can accurately predict the presence of heart disease in a patient, based on clinical features.

---

## 📊 Dataset
- **Source**: UCI Heart Disease Dataset (also available on Kaggle)  
- **Features**: 14 clinical features such as age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, ECG results, maximum heart rate, exercise-induced angina, oldpeak, slope, number of major vessels, and thalassemia.  
- **Target Variable**:  
  - `0` → No Heart Disease  
  - `1` → Heart Disease  

---

## 🛠️ Technologies Used
- **Python 3.x**  
- **pandas** – Data manipulation and analysis  
- **NumPy** – Numerical computing  
- **matplotlib / seaborn** – Data visualization  
- **scikit-learn** – Machine learning algorithms & evaluation metrics  
- **XGBoost** – Gradient boosting algorithm  
- **Jupyter Notebook** – Experimentation and analysis  

---

## 🔄 Project Workflow

### 1. Data Preprocessing
- Handle missing values (if any)  
- Encode categorical features  
- Standardize numerical features  

### 2. Exploratory Data Analysis (EDA)
- Distribution of features (age, cholesterol, etc.)  
- Correlation heatmap between variables  
- Visualizations of relationships between features and target  

### 3. Model Development
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Random Forest Classifier  
- XGBoost Classifier  
- Support Vector Machine (SVM)  

### 4. Model Evaluation
- Accuracy  
- Precision, Recall, F1-score  
- ROC-AUC Curve  
- Confusion Matrix  

---

## 📈 Results

| Model                | Accuracy | Precision | Recall | F1-Score |
|-----------------------|----------|-----------|--------|----------|
| Logistic Regression   | 84%      | 0.83      | 0.86   | 0.84     |
| KNN                   | 85%      | 0.84      | 0.85   | 0.84     |
| Random Forest         | 87%      | 0.86      | 0.87   | 0.87     |
| XGBoost               | 89%      | 0.88      | 0.90   | 0.89     |
| SVM                   | 86%      | 0.85      | 0.86   | 0.85     |

✅ **Best Performing Model**: **XGBoost** with ~89% accuracy and highest ROC-AUC.

---

## 📁 Project Structure

Heart-Disease-Prediction/
│
├── data/ # (local only, not uploaded to GitHub)
│ ├── raw/ # raw dataset
│ └── processed/ # cleaned / transformed data
│
├── notebooks/ # Jupyter notebooks
│ └── Heart_Disease_Classification.ipynb
│
├── src/ # Python scripts (reusable functions)
│ ├── preprocessing.py
│ ├── train_models.py
│ └── evaluate.py
│
├── models/ # saved trained models
├── plots/ # saved plots and visualizations
│
├── requirements.txt # dependencies
├── README.md # project documentation
├── LICENSE # license file
└── .gitignore # ignore unnecessary files

---

## 📝 Key Insights
- XGBoost performed best with the highest accuracy (89%).  
- Feature scaling and encoding significantly impacted KNN and SVM performance.  
- Random Forest and XGBoost handled feature importance well.  
- Ensemble methods outperformed simpler models.  

---

## 🔮 Future Improvements
- Hyperparameter tuning for more optimized results  
- Try deep learning models (ANN) for comparison  
- Deploy the model using Flask/Streamlit for real-time predictions    

---

## 🤝 Contributing
Feel free to fork this project and submit pull requests for improvements.  

---

## 📄 License
This project is open source and available under the **MIT License**.

---

## 📧 Contact
**Raj Verma** – 25f1001478@ds.study.iitm.ac.in  
Project Link: [Heart Disease Prediction Repo](https://github.com/pattern-finder-Raj/Heart-Disease-Prediction)

