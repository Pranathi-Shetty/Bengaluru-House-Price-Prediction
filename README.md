# 🏠 Bengaluru House Price Prediction Using Machine Learning

### A Machine Learning Case Study on Real Estate Price Prediction

This project develops a machine learning-based regression system to predict residential property prices in Bengaluru, India, using historical housing data.

The study follows an end-to-end machine learning workflow involving data preprocessing, exploratory data analysis, feature engineering, model development, cross-validation, hyperparameter optimization, performance evaluation, and model serialization.

Multiple regression algorithms are evaluated to identify a suitable model for predicting house prices based on property characteristics such as location, area, number of bedrooms, bathrooms, balconies, and property type.

---

## 📌 1. Project Overview

Accurately estimating residential property prices is a challenging task because real estate values depend on several interconnected factors, including geographical location, property size, configuration, and market conditions.

Traditional property valuation methods often rely on manual assessment and historical comparisons.

This project investigates how supervised machine learning algorithms can model relationships between property characteristics and selling prices.

### Project Objectives

- Analyze residential property data from Bengaluru.
- Identify factors influencing housing prices.
- Clean and preprocess raw real estate data.
- Perform feature engineering and exploratory data analysis.
- Develop and compare multiple regression models.
- Optimize selected models using hyperparameter tuning.
- Evaluate predictive performance using standard regression metrics.
- Identify important features influencing predictions.
- Save the trained model for future inference.

---

## 📊 2. Dataset Description

**Dataset:** Bengaluru House Price Data  
**Source:** Kaggle  
**Original Records:** 13,320  
**Original Features:** 9  
**Prediction Target:** `price`

The dataset contains residential property listings from different localities in Bengaluru.

### Dataset Attributes

| Feature | Description |
|---|---|
| `area_type` | Type of property area |
| `availability` | Property availability status |
| `location` | Bengaluru locality |
| `size` | Property configuration, such as 2 BHK or 3 BHK |
| `society` | Residential society name |
| `total_sqft` | Total property area |
| `bath` | Number of bathrooms |
| `balcony` | Number of balconies |
| `price` | Property price in lakhs of Indian rupees |

### Selected Model Features

The final model uses:

1. `area_type`
2. `location`
3. `total_sqft_clean`
4. `bhk`
5. `bath`
6. `balcony`

**Target variable:** `price` (₹ lakhs)

---

## 🛠️ 3. Technologies Used

### Programming Language
- Python

### Development Environment
- Jupyter Notebook
- Anaconda
- Git and GitHub

### Libraries
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

### Machine Learning Techniques
- Regression Analysis
- Feature Engineering
- One-Hot Encoding
- Feature Scaling
- Cross-Validation
- Hyperparameter Optimization
- Ensemble Learning

---

## 🔄 4. Machine Learning Workflow

The project follows the workflow:

**Data Collection → Data Cleaning → Exploratory Data Analysis → Feature Engineering → Data Splitting → Preprocessing → Model Training → Cross-Validation → Hyperparameter Tuning → Model Evaluation → Model Selection → Model Serialization**

### 4.1 Data Preprocessing

The original dataset contains missing values, duplicate records, inconsistent property-area representations, and potentially unrealistic property configurations.

The preprocessing stage includes:

- Handling missing values.
- Converting property-area measurements into square feet.
- Extracting bedroom counts from property configurations.
- Removing duplicate modelling records.
- Filtering invalid numerical values.
- Applying domain-based property configuration filters.
- Removing unnecessary attributes.

### 4.2 Feature Engineering

**Total Area Conversion**

The `total_sqft` feature contains numerical values, ranges, and alternative measurement units.

These values are converted into a consistent square-foot representation named `total_sqft_clean`.

**Bedroom Extraction**

The `size` feature is transformed into a numerical `bhk` attribute.

For example:

- `2 BHK` → `2`
- `3 BHK` → `3`
- `4 Bedroom` → `4`

**Domain-Based Filtering**

Property configurations are filtered using the following criteria:

- Minimum area per bedroom: 300 square feet.
- Maximum area per bedroom: 2,000 square feet.
- Maximum bedrooms: 10.
- Maximum bathrooms: 10, where bathroom counts are available.

These are heuristic data-cleaning rules and may exclude some legitimate properties.

After filtering, the dataset contains **11,811 records**.

### 4.3 Data Splitting

The cleaned dataset is divided into:

| Dataset | Records | Percentage |
|---|---:|---:|
| Training | 9,448 | 80% |
| Testing | 2,363 | 20% |
| Total | 11,811 | 100% |

**Random State:** 42

### 4.4 Feature Preprocessing

Numerical and categorical features are processed using a Scikit-learn `ColumnTransformer`.

**Numerical Features**
- Missing-value imputation using the median.
- Standardization using `StandardScaler`.

**Categorical Features**
- Missing-value imputation using the most frequent category.
- One-hot encoding.
- Handling infrequent and previously unseen categories.

The preprocessing operations are integrated into machine learning pipelines to maintain consistent transformations during training and prediction.

---

## 🤖 5. Machine Learning Models

The project evaluates several regression approaches, with particular emphasis on:

### Ridge Regression

A regularized linear regression model that reduces coefficient magnitude using L2 regularization.

### Random Forest Regressor

An ensemble learning algorithm that combines predictions from multiple decision trees.

### Gradient Boosting Regressor

A sequential ensemble learning algorithm that builds decision trees to correct errors made by earlier models.

### Dummy Regressor

A baseline model used to establish a reference level of prediction performance.

---

## 📈 6. Baseline Model Comparison

The following results were obtained using five-fold cross-validation on the training dataset.

| Model | Mean R² | Mean MAE (₹ Lakhs) | Mean RMSE (₹ Lakhs) |
|---|---:|---:|---:|
| Gradient Boosting | 0.6629 | 33.7971 | 79.5497 |
| Ridge Regression | 0.6593 | 36.3925 | 79.5867 |
| Random Forest | 0.6341 | 32.9888 | 83.0328 |
| Dummy Regressor | -0.0010 | 73.7357 | 136.4332 |

**Observations:**

- Gradient Boosting achieved the lowest baseline cross-validation RMSE.
- Ridge Regression demonstrated competitive performance.
- Random Forest achieved the lowest baseline cross-validation MAE among these models.
- All three trained regression models substantially outperformed the Dummy Regressor.

---

## ⚙️ 7. Hyperparameter Optimization

Hyperparameter tuning was performed using `RandomizedSearchCV` with five-fold cross-validation.

The optimization objective was to minimize Root Mean Squared Error (RMSE).

### 7.1 Gradient Boosting — Best Hyperparameters

| Hyperparameter | Selected Value |
|---|---|
| `n_estimators` | 500 |
| `learning_rate` | 0.1 |
| `max_depth` | 5 |
| `min_samples_split` | 15 |
| `min_samples_leaf` | 4 |
| `subsample` | 0.9 |
| `max_features` | `log2` |

**Best Cross-Validation RMSE:** 78.5341 lakhs

### 7.2 Random Forest — Selected Hyperparameters

| Hyperparameter | Selected Value |
|---|---|
| `n_estimators` | 400 |
| `max_depth` | None |
| `min_samples_split` | 10 |
| `min_samples_leaf` | 1 |
| `max_features` | 0.8 |

**Best Cross-Validation RMSE:** 79.6788 lakhs

### Model Selection

The tuned Gradient Boosting Regressor achieved a lower cross-validation RMSE than the tuned Random Forest Regressor.

Therefore, **Gradient Boosting Regressor was selected as the final model**.

---

## 🏆 8. Final Model Performance

The selected Gradient Boosting model was evaluated on the held-out test dataset.

| Evaluation Metric | Result |
|---|---:|
| R² Score | 0.5853 |
| Mean Absolute Error (MAE) | 34.4603 lakhs |
| Mean Squared Error (MSE) | 9447.685 |
| Root Mean Squared Error (RMSE) | 97.1992 lakhs |

### Interpretation

The model explains approximately **58.53% of the variation in house prices** in the held-out test dataset.

The Mean Absolute Error indicates that predictions differ from actual property prices by approximately ₹34.46 lakhs on average.

The test RMSE is higher than the cross-validation RMSE, indicating weaker performance on unseen test observations.

The model provides a useful predictive baseline but should not be treated as a substitute for professional property valuation.

---

## 📉 9. Model Evaluation and Visualization

The project includes regression evaluation visualizations.

### Actual vs. Predicted Prices

Compares predicted property prices against actual property prices.

### Residual Analysis

Examines the difference between actual and predicted values to identify systematic errors.

### Prediction Error Distribution

Illustrates the distribution of prediction errors.

### Feature Importance

Identifies which features contribute most strongly to Gradient Boosting predictions.

### Grouped Feature Importance

Aggregates feature importance across numerical and categorical feature groups.

These visualizations support model interpretation and help identify limitations in predictive performance.

---

## 🔍 10. Feature Importance Analysis

The selected Gradient Boosting model produced the following grouped feature importance scores:

| Feature Group | Importance |
|---|---:|
| Numerical Features | 69.92% |
| Location | 24.17% |
| Area Type | 5.91% |

### Important Individual Features

| Feature | Importance |
|---|---:|
| Total Square Feet | 0.4162 |
| BHK | 0.1474 |
| Bathrooms | 0.1190 |
| Balconies | 0.0166 |

### Key Findings

- Property area is the most influential individual numerical predictor.
- Bedroom and bathroom counts contribute to estimated property prices.
- Location is an important categorical predictor.
- Area type contributes additional predictive information.

Feature importance measures the model's relative reliance on features and does not establish causal relationships.

---

## 🧪 11. Sample Prediction

The trained model was tested with the following sample property:

| Property Attribute | Value |
|---|---|
| Area Type | Super built-up Area |
| Location | Whitefield |
| Total Area | 1,200 sq. ft. |
| Bedrooms | 2 BHK |
| Bathrooms | 2 |
| Balconies | 1 |

### Predicted Property Price

**₹54.89 Lakhs**

This is an illustrative model prediction, not a verified market valuation.

---

## ⚠️ 12. Limitations

Although the model captures useful relationships in the dataset, several limitations remain.

**High-Value Property Prediction**

The model exhibits large prediction errors for some expensive properties.

**Location Complexity**

Property prices may vary considerably within the same locality.

**Missing Market Information**

The dataset does not explicitly include several potentially influential factors, such as:

- Property age
- Construction quality
- Floor level
- Nearby infrastructure
- Market conditions
- Transaction date

**Dataset Filtering**

Domain-based filtering may exclude unusual but legitimate properties.

**Generalization**

The difference between cross-validation and test performance suggests that further improvements are necessary.

---

## 🚀 13. Future Enhancements

Potential improvements include:

- Evaluating XGBoost, LightGBM, and CatBoost.
- Investigating more advanced feature engineering.
- Incorporating geographical coordinates.
- Applying improved outlier detection techniques.
- Studying alternative approaches for luxury properties.
- Expanding hyperparameter optimization.
- Adding explainability methods such as SHAP.
- Developing a Streamlit-based prediction interface.
- Deploying the trained model as a web application.
- Updating the dataset with more recent property information.

---

## 📁 14. Project Structure

The following is a suggested repository organization. Actual filenames may differ.

```text
Bengaluru-House-Price-Prediction/
│
├── data/
│   └── Bengaluru_House_Data.csv
│
├── notebooks/
│   └── Bengaluru_House_Price_Prediction_V2.ipynb
│
├── figures/
│   ├── figure_01_actual_vs_predicted.png
│   ├── figure_02_residual_analysis.png
│   ├── figure_03_prediction_error_distribution.png
│   ├── figure_04_top_20_feature_importance.png
│   └── figure_05_grouped_feature_importance.png
│
├── results/
│   ├── model_results.csv
│   ├── model_hyperparameters.csv
│   └── figure_captions.csv
│
├── saved_models/
│   └── README.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

Large serialized model files are excluded from Git tracking when they exceed repository hosting limits.

---

## 💻 15. Running the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/Pranathi-Shetty/Bengaluru-House-Price-Prediction.git
```

### Step 2: Navigate to the Project

```bash
cd Bengaluru-House-Price-Prediction
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

Otherwise, install the main dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib notebook
```

### Step 5: Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the project notebook and execute the cells sequentially.

**Note:** The dataset must be available at the path expected by the notebook. Model inference also requires access to the saved model artifact or retraining the model.

---

## 📚 16. Conclusion

This project demonstrates the application of supervised machine learning techniques to residential real estate price prediction.

The study includes data preprocessing, feature engineering, exploratory analysis, model comparison, cross-validation, hyperparameter tuning, and final evaluation.

Among the tuned models, the Gradient Boosting Regressor achieved the best cross-validation RMSE and was selected for final evaluation.

The final model achieved an R² score of **0.5853** and an RMSE of **97.1992 lakhs** on the held-out test dataset.

The findings demonstrate the importance of property area, location, and property configuration in predicting residential property prices while highlighting the challenges associated with expensive and atypical properties.

---

## 👩‍💻 Author

**Pranathi Shetty**

**Project:** Bengaluru House Price Prediction Using Machine Learning

**GitHub:** Pranathi-Shetty

---
