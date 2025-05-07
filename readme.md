# 🧬 Breast Cancer lumps Classification using Logistic Regression and CatBoost

This repository presents a machine learning pipeline for classifying breast lumps as **malignant** or **benign** using the Wisconsin Breast Cancer dataset. I explored two classification models:

- Logistic Regression (as a baseline)
- CatBoostClassifier (advanced boosting model)

---

## 📁 Dataset Overview

- **Source:** University of Wisconsin Hospitals, Madison
- **Target Variable:** `diagnosis` (0 = Benign, 1 = Malignant)
- **Features:** Various statistical measures (e.g., mean radius, mean area, mean texture, etc.)

---

## 📊 Exploratory Data Analysis

### 🔹 Pair Plot of Selected Features

![Pairplot](images/pairplot.png)

> This plot helps visualize the **pairwise relationships** between selected numerical features colored by the `diagnosis` class.
>
> - Diagonal plots show the **distribution (KDE)** of each feature.
> - Off-diagonal plots show **scatterplots with regression lines**.
> - From the plot:
>   - `mean radius`, `mean perimeter`, and `mean area` show strong linear relationships and class separation.
>   - Benign tumors (label 0) tend to cluster separately from malignant tumors (label 1), especially in the size-based features.

### 🔹 Correlation Heatmap

![Correlation Heatmap](images/heatmap.png)

> Strong positive correlation observed between `mean perimeter`, `mean area`, and `mean radius`, indicating potential multicollinearity.

### 🔹 Scatter Plots

#### • Mean Perimeter vs Mean Radius

![Perimeter vs Radius](images/scatter_1.png)

> Clear linear relationship observed; higher values often correspond to benign tumors.

#### • Mean Area vs Mean Texture

![Area vs Texture](images/scatter_2.png)

> Less distinct class separation; highlights the importance of using multiple features.

#### • Mean Smoothness vs Mean Texture

![Smoothness vs Texture](images/scatter_3.png)

> Overlap in classes indicates these features may contribute more when combined with others.

---

## ⚙️ Logistic Regression Model

### 🔹 Preprocessing:

- scaled the data using StandardScaler
- Split the data to training and test sets. Training set was 75%, and 25% for test set.

### 🔹 Grid Search:

- For the logistic regression model I tuned `C`, `penalty`, `solver`, `max_iter` using `GridSearchCV` from scikit-learn.

### 🔹 Evaluation:

- **Test Accuracy:** The optimal model achieved an accuracy score of `91%`
- **AUC:** The area under curve (AUC) was `0.98`

#### • Confusion Matrix:

![Logistic Confusion Matrix](images/logistic_confusion_matrix.png)

> Logistic regression performed well but had 27 false positives.which is a huge number. The false positives got were 0 which is a good sign in predicting if breast lumps are benign or malignant.

#### • ROC Curve:

![Logistic ROC](images/logistic_AUC.png)

> AUC of 0.98 shows excellent discriminative power of the model.

---

## 🚀 CatBoostClassifier model

### 🔹 Grid Search:

- For CatBoostClassifier I tuned `depth`, `iterations`, `learning_rate`, `l2_leaf_reg` of the model.

### 🔹 Evaluation:

- **Test Accuracy:** The model achieved an accuracy score of `90.9%`.
- **AUC:** The area under curve (AUC) was `0.98`

#### • Confusion Matrix:

![CatBoost Confusion Matrix](images/catboost_confusion_matrix.png)

> For CatBoostClassifier there was a slightly improved recall, and reduced false positives from 27 to 6. There was an increase in false negatives but it is a more consistent model.

#### • ROC Curve:

![CatBoost ROC](images/catboost_AUC.png)

> Also achieved an AUC of 0.98, indicating excellent performance similar to logistic regression.

---

## 💾 Model Saving and Reuse

Saved the model using joblib.dump() and loade it to test its accuracy in predicting the first row from the dataframe and the prediction was accurate.

---

## ✅ Conclusion

Both models achieved excellent performance:

- **Logistic Regression** had slightly better precision.
- **CatBoostClassifier** slightly reduced **false negatives**, which is crucial in medical diagnosis.

I saved the model `CatBoostClassifier` ready for streamlit app development.

# 🧬 Breast Cancer Prediction Streamlit Web App

This Streamlit web application allows users to input details of a breast tumor (e.g., measurements such as radius, perimeter, texture, etc.) and predicts whether the tumor is **benign** or **malignant** using a pre-trained CatBoost machine learning model.

---

## 🚀 Features

- 🔢 Text input fields for tumor measurements
- 🔮 Instant prediction of diagnosis (Benign or Malignant)
- 🎞️ Centered GIF loading animation on prediction
- 🧠 Model powered by `CatBoostClassifier`
- ✅ User-friendly and responsive interface

---

## 🖼️ Live Loading Animation

When the "Predict Diagnosis" button is clicked, a centered GIF appears temporarily to simulate a real-time prediction experience:

![Loading GIF Sample](https://media.tenor.com/1tPR5fhN_AsAAAAm/world-cancer-day-survivor.webp)

---

## 🧩 Files

- `app.py` – The Streamlit application
- `utils/cat_boost_model.pkl` – Trained CatBoost model
- `utils/columns.pkl` – List of features used in training
- `GIF` - 'https://media.tenor.com/1tPR5fhN_AsAAAAm/world-cancer-day-survivor.webp'

---

## 🧪 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/breast-cancer-app.git
cd breast-cancer-app
```

### 2. Install required packages

Install required packages listed in the jupyter notebook

````

### 3. Run the Streamlit app

```bash
streamlit run app.py
````

---

## 📦 requirements.txt

```txt
streamlit
pandas
joblib
catboost
```

---

## 📸 Screenshot

![App Screenshot](images/app.png)

---

---

## 📌 Requirements

- Python 3.8+
- scikit-learn
- catboost
- pandas, matplotlib, seaborn
- joblib

---

### Author: Obadiah Kiptoo

---

## 📬 Acknowledgments

- Dataset courtesy of Dr. William H. Wolberg
- University of Wisconsin Hospitals, Madison
- Powered by CatBoost and Streamlit
