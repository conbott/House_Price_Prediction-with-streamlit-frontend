# House Price Prediction & Real Estate Valuation System

An end-to-end Machine Learning project to predict residential property prices using US real estate listing data. Features a complete Scikit-Learn preprocessing and modeling pipeline paired with an interactive Streamlit web dashboard.

---

## Project Overview
This project predicts fair market property valuations using continuous and categorical features (bedrooms, bathrooms, living area, lot size, and state). It fulfills all 12 project requirements: data cleaning, missing value handling, exploratory data analysis, pipeline feature preprocessing, multi-model evaluation, and deployment via Streamlit.

---

## Dataset Information
- **Dataset**: USA Real Estate Listings (`realtor-data.zip.csv`)
- **Total Cleaned Records**: Over 715,000 verified listings (comfortably exceeds the minimum requirement of 25,000 records)
- **Target Variable**: `price` (Continuous Numeric in USD)
- **Selected Predictors**:
  - `house_size` (Living area in square feet)
  - `bath` (Number of bathrooms)
  - `bed` (Number of bedrooms)
  - `acre_lot` (Lot size in acres)
  - `state` (Categorical location)

---

## Machine Learning Pipeline
1. **Data Cleaning**: Stripped identifiers (`brokered_by`, `street`, `zip_code`), filtered active `for_sale` properties, and eliminated non-sensical outliers (prices outside \$30k–\$5M, house sizes outside 300–12,000 sqft).
2. **Missing Value & Duplicate Handling**: Eliminated duplicate records; imputed numerical attributes with median values and categorical fields with the mode.
3. **Exploratory Data Analysis (EDA)**: Analyzed price distribution, feature correlations, and state-level median price benchmarks.
4. **Feature Engineering**: Built a Scikit-Learn `ColumnTransformer` utilizing `StandardScaler` for numeric variables and `OneHotEncoder(handle_unknown='ignore')` for location categories.
5. **Models Trained**:
   - **Model 1**: Linear Regression (Parametric Baseline)
   - **Model 2**: Random Forest Regressor (Non-linear Ensemble, 100 trees, max depth 16)

---

## Model Evaluation & Comparison
Both models were evaluated on an unseen 20% holdout test set:

| Model | Train $R^2$ | Test $R^2$ | Test MAE ($) | Test RMSE ($) |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | 0.4639 | 0.4671 | \$251,570 | \$429,208 |
| **Random Forest Regressor** | **0.8020** | **0.5040** | **\$223,415** | **\$414,080** |

### Decision & Justification:
**Random Forest Regressor** was selected as the production model because it achieved a higher Test $R^2$ (0.5040 vs 0.4671) and reduced average valuation error by ~\$28,000 per property compared to Linear Regression.

---

## How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Jupyter Notebook
Open and run `House_Price_Predictions.ipynb` to inspect the analysis and model training steps.

### 3. Launch the Streamlit Web Application
```bash
python -m streamlit run HousePrice_app.py
```
Open `http://localhost:8501` in your browser to interact with the prediction dashboard.

---

## Repository Structure
```
├── House_Price_Predictions.ipynb   # Complete 12-step Jupyter Notebook
├── HousePrice_app.py               # Streamlit interactive frontend
├── requirements.txt                # Project dependencies
├── README.md                       # Project documentation & results
└── saved_models/
    └── house_price_pipeline.pkl    # Serialized Random Forest model
```
