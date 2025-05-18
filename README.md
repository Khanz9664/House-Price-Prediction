# 🏠 House Price Prediction using Machine Learning

This project leverages supervised machine learning techniques to predict house prices based on the **Ames Housing Dataset**. It walks through a complete data science pipeline — including data preprocessing, feature engineering, model building, and evaluation.

---

## 📌 Overview

The goal is to predict the final selling price of a house using various features like quality, size, location, and amenities. The dataset contains 80+ variables describing different aspects of residential homes in Ames, Iowa.

---

## 📂 Project Structure

```
House-Price-Prediction/
├── House Price Prediction Model.ipynb # Main Jupyter notebook
├── model.pkl # Trained model (optional)
├── selected_features.pkl # Selected top 50 features
└── README.md # Project overview
```

---

## 🧪 Tools & Technologies

- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn
- Joblib
- XgBoost
- Jupyter Notebook

---

## 📊 Workflow Summary

### 1. **Exploratory Data Analysis (EDA)**
- Visualized feature relationships
- Identified missing data and outliers

### 2. **Data Cleaning**
- Dropped features with excessive missing values
- Imputed remaining missing values using mean/mode strategy
- Converted categorical values to numerical using **Label Encoding**

### 3. **Feature Engineering**
- Selected the top **50 features** based on correlation with `SalePrice`
- Ensured all features were numeric and compatible with modeling
- Scaled the dataset appropriately when needed

### 4. **Modeling**
- Applied **Linear Regression, RandomForestRegressor, Ridge, Lasso, XgBoost**
- Split dataset: 80% for training, 20% for testing
- Evaluated using **Root Mean Squared Error (RMSE)**

---

## 📈 Model Performance

- **Model Used**: `Linear Regression, RandomForestRegressor, Ridge, Lasso, XgBoost`
- **Evaluation Metric**: RMSE, R Squared, and MAE
- **Result**: RandomForest Performed better among all the trained models
- **Feature Importance**: Derived from the model to explain influential factors

---

## ✅ Key Learnings

- Handling real-world data often requires extensive cleaning and preparation
- Feature selection based on correlation can improve both accuracy and efficiency
- Proper encoding of categorical variables is crucial for model compatibility

---

## 🧠 Future Improvements

- Hyperparameter tuning using GridSearchCV
- Advanced feature engineering (polynomial features, interaction terms)
- Robust outlier detection methods

---

## 👤 Author

**Shahid Ul Islam**  
_Machine Learning & Data Science Enthusiast_

