# 🎓 BurnoutIQ 2.0

### Student Burnout Risk Analytics & Intelligence Platform

An interactive data science application for exploring student wellbeing patterns, analysing academic and lifestyle factors, and demonstrating machine learning techniques using synthetic student data.

**Live Application:** https://student-burnout-risk-analysis.streamlit.app/

## 📌 Project Overview

BurnoutIQ 2.0 is a machine learning and analytics platform developed using Python and Streamlit.

The project combines exploratory data analysis, supervised classification, unsupervised clustering, and interactive scenario comparison within a unified intelligence workspace.

The application is designed as an educational research prototype and is not intended for clinical diagnosis or real-world student risk assessment.

## 🚀 Key Features

### 1. Command Center

* Interactive student wellbeing dashboard
* Burnout distribution analysis
* Academic and lifestyle visualizations
* Department-level comparisons

### 2. Student Explorer

* Individual student record exploration
* Comparison with dataset median values
* Academic and lifestyle profile analysis

### 3. Risk Intelligence

* Machine learning-based burnout category estimation
* Low, Medium, and High category classification
* Interactive student profile inputs
* Model-estimated category probabilities

### 4. What-If Lab

* Compare hypothetical student profiles
* Explore model responses to changes in selected inputs
* Visualize differences between baseline and alternative scenarios

Scenario comparisons demonstrate model sensitivity, not causal effects.

### 5. Research Studio

* Classification model evaluation
* Confusion matrix visualization
* Feature correlation analysis
* Exploratory student segmentation
* Dataset filtering and CSV export

## 🧠 Machine Learning Techniques

**Supervised Learning**

* Logistic Regression
* Classification pipeline
* Missing-value preprocessing
* Model comparison and evaluation

**Unsupervised Learning**

* K-Means Clustering
* Student segmentation
* Silhouette score evaluation

## 📊 Model Performance

| Metric                      | Result |
| --------------------------- | ------ |
| Classification Accuracy     | 70.00% |
| Macro F1 Score              | 69.17% |
| Clustering Silhouette Score | 0.056  |
| Dataset Records             | 800    |

The classification metrics were obtained from a held-out test split of synthetic data.

The low silhouette score indicates weak cluster separation. Student segments should therefore be interpreted as exploratory groupings rather than validated student profiles.

## 🛠️ Technology Stack

* **Programming:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Visualization:** Plotly
* **Web Application:** Streamlit
* **Model Serialization:** Joblib
* **Version Control:** Git and GitHub
* **Deployment:** Streamlit Community Cloud

## 📂 Dataset

The project uses a synthetic dataset containing 800 student records.

The dataset includes academic, demographic, lifestyle, and self-reported wellbeing variables.

**Target variable:** `Burnout_Level`

**Classification categories:** Low, Medium, High

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/anuvarna-r/student-burnout-risk-analysis.git
```

Navigate to the project folder:

```bash
cd student-burnout-risk-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the application:

```bash
streamlit run app.py
```

## ⚠️ Limitations

* The dataset is synthetic and does not represent a validated student population.
* Model predictions have not been externally validated.
* Estimated classification probabilities are not clinical risk probabilities.
* Exploratory clusters exhibit weak separation.
* What-if comparisons do not establish causal relationships.
* The application must not be used for medical diagnosis, student screening, or institutional decisions.

## 🔮 Future Improvements

* Evaluation using suitable real-world datasets with appropriate consent and governance
* Improved model calibration and validation
* Explainable AI techniques
* Enhanced data quality checks
* More robust segmentation approaches

## 👩‍💻 Project Type

**End-to-End Data Science & Machine Learning Portfolio Project**

Developed to demonstrate practical skills in data preprocessing, exploratory analysis, supervised and unsupervised learning, model evaluation, interactive application development, and cloud deployment.
