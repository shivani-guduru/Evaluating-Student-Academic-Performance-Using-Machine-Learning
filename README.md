
# **Classifying Student Academic Performance Using Machine Learning**

## **Project Overview**

This project aims to build and evaluate machine learning models to predict student performance based on academic, behavioral, and socio-economic features. The objective is to predict the final grade (G3) of students using multiple regression models.

### **Goal**

The goal of this project is to predict student performance (Final Grade, G3) using machine learning models. The dataset includes various features such as study time, internet access, and family background.

### **Key Steps**

1. **Import the Dataset**: The dataset is imported directly from the **UCI Machine Learning Repository**.
2. **Exploratory Data Analysis (EDA)**: Initial exploration of the dataset, including missing value handling, summary statistics, and visualization of data distributions.
3. **Feature Engineering**: Binned target grades into **Low**, **Medium**, and **High** categories based on the final grade.
4. **Model Selection & Training**: Various regression models, including **Ridge Regression**, **Support Vector Regression (SVR)**, and **K-Nearest Neighbors Regressor (KNN)**, are trained on the dataset.
5. **Model Evaluation**: Models are evaluated using metrics like **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, **R-squared**, and **Adjusted R-squared**.
6. **Hyperparameter Tuning**: **GridSearchCV** is used to optimize the hyperparameters of the models.
7. **Feature Importance**: Feature importance is extracted to interpret which features are the most influential in predicting student performance.

---

## **Libraries and Dependencies**

The following libraries are used in this project:

- `ucimlrepo` - For fetching datasets from the UCI repository.
- `pandas`, `numpy` - For data manipulation.
- `matplotlib`, `seaborn` - For data visualization.
- `sklearn` - For machine learning models, evaluation, and preprocessing.

### **Required Libraries**

You can install all necessary dependencies by running:

```bash
pip install -r requirements.txt
```

Where the `requirements.txt` file contains the following:

```plaintext
ucimlrepo
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## **Dataset**

- **Dataset**: **Student Performance Data** (ID = 320) from the **UCI Machine Learning Repository**.
  - **Description**: This dataset contains information about students' academic performance based on various features such as subject grades, parental education, and socio-economic status.

---

## **Modeling**

In this project, the following machine learning models are applied:

- **Ridge Regression** (`Ridge`)
- **Support Vector Regression** (`SVR`)
- **K-Nearest Neighbors Regressor** (`KNN`)

These models are trained to predict the final grade (G3) of students.

---

## **Data Preprocessing**

- **Handling Missing Values**: Missing data is handled during preprocessing using techniques like imputation.
- **Feature Scaling**: Numerical features are scaled using `StandardScaler`.
- **One-Hot Encoding**: Categorical features are one-hot encoded.

---

## **Model Evaluation**

Models are evaluated using the following metrics:

- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R-squared (R²)**
- **Adjusted R-squared (Adj R²)**

Additionally, **GridSearchCV** is used to optimize the models by tuning hyperparameters.

---

## **Feature Importance and Insights**

- **Feature Coefficients**: The most influential features are extracted from the trained models, particularly **Ridge Regression**, and their impact on the final grade (G3) is analyzed.
- **Visualization**: A bar plot visualizes the features with the most impact on student performance.
- **Actionable Insights**: A table is presented with the top features that affect student grades, providing actionable insights for early intervention.

---

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
