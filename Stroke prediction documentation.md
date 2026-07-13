# Stroke Prediction Project

## Overview
This project focuses on building machine learning models to predict the likelihood of stroke based on health and lifestyle factors. The dataset is highly imbalanced (only ~4.87% of cases are positive for stroke), so special attention is given to handling this imbalance using SMOTE and class-weighting techniques.

**Objective:** Develop and compare models that maximize recall (sensitivity) to identify as many true stroke cases as possible, while maintaining a reasonable balance with precision.

--Objective**: Build machine learning models to predict stroke likelihood.
  
**Challenge**: The dataset has a severe class imbalance (very few stroke cases compared to non-stroke cases). We will handle this using SMOTE and class weights to see which works better.

## Dataset
- **Source:** `healthcare-dataset-stroke-data.csv`
- **Size:** 5,110 rows × 12 columns
- **Features:**
  - `id`: unique identifier
  - `gender`: Male, Female, Other
  - `age`: continuous
  - `hypertension`: 0/1
  - `heart_disease`: 0/1
  - `ever_married`: Yes/No
  - `work_type`: Private, Self-employed, Govt_job, children, Never_worked
  - `Residence_type`: Urban/Rural
  - `avg_glucose_level`: continuous
  - `bmi`: continuous (contains 'N/A' strings)
  - `smoking_status`: never smoked, formerly smoked, smokes, Unknown
  - `stroke`: target variable (0 = no stroke, 1 = stroke)

### Class Distribution
- **No Stroke (0):** 4,861 samples (95.13%)
- **Stroke (1):** 249 samples (4.87%)

*Note:* The `bmi` column originally contained 201 `'N/A'` strings, which were converted to NaN and then filled with the median value (28.1).

## Methods

### 1. Data Preprocessing
- Dropped the `id` column (non‑predictive).
- Removed the single record where `gender == 'Other'`.
- Converted `bmi` to numeric, coercing errors to NaN, then filled missing values with the median.
- One‑hot encoded categorical variables (`gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`) with `drop_first=True`.
- Separated features (`X`) and target (`y`).
- Split data into training (80%) and test (20%) sets, stratified by the stroke label.
- Standardized numeric features (`age`, `avg_glucose_level`, `bmi`) using `StandardScaler` (fit on training data only).

### 2. Handling Class Imbalance
Two complementary strategies were explored:

**a. SMOTE (Synthetic Minority Oversampling Technique)**
- Applied only to the training set.
- Generated synthetic stroke samples until the classes were balanced (3,888 each).

**b. Class Weights**
- Computed `scale_pos_weight = #negative / #positive ≈ 19.52`.
- Passed this weight to models that support it (XGBoost) or used `class_weight='balanced'` for Logistic Regression and Random Forest.

### 3. Models Evaluated
| Model | Variant | Description |
|-------|---------|-------------|
| Logistic Regression | `LR_SMOTE` | Trained on SMOTE‑balanced data |
| Logistic Regression | `LR_ClassWeight` | Trained on original scaled data with `class_weight='balanced'` |
| Random Forest | `RF_SMOTE` | Trained on SMOTE‑balanced data |
| Random Forest | `RF_ClassWeight` | Trained on original scaled data with `class_weight='balanced'` |
| XGBoost | `XGB_SMOTE` | Trained on SMOTE‑balanced data |
| XGBoost | `XGB_ClassWeight` | Trained on original scaled data with `scale_pos_weight` |

All models used `random_state=42` for reproducibility. Logistic Regression used `max_iter=1000`. Random Forest used `n_estimators=100`. XGBoost used `n_estimators=100` and `eval_metric='logloss'`.

### 4. Evaluation
Models were assessed on the untouched test set using:
- **Precision** (positive predictive value)
- **Recall / Sensitivity** (ability to detect strokes)
- **F1‑Score** (harmonic mean of precision and recall)
- **ROC‑AUC** (ranking ability)

Because the problem is highly imbalanced, **recall** and **F1‑score** are prioritized over accuracy.

## Results

| Model | Precision | Recall | F1‑Score | ROC‑AUC |
|-------|-----------|--------|----------|---------|
| LR_SMOTE | 0.1383 | 0.78 | **0.2349** | 0.8213 |
| LR_ClassWeight | 0.1347 | 0.80 | 0.2305 | 0.8394 |
| XGB_ClassWeight | 0.1930 | 0.22 | 0.2056 | 0.7690 |
| RF_SMOTE | 0.1163 | 0.20 | 0.1471 | 0.7777 |
| RF_ClassWeight | 0.2273 | 0.10 | 0.1389 | 0.7987 |
| XGB_SMOTE | 0.1373 | 0.14 | 0.1386 | 0.7717 |

**Best model by F1‑Score:** `LR_SMOTE` (Logistic Regression with SMOTE).

### Interpretation
- The SMOTE‑based logistic regression achieved the highest recall (0.78) alongside a reasonable precision, yielding the best F1‑score.
- The class‑weighted logistic regression achieved the highest recall (0.80) but lower precision, resulting in a slightly lower F1.
- Despite the severe imbalance, models achieved ROC‑AUC scores in the high 0.70s–0.80s, indicating good ranking ability.

### Visualizations
- **Confusion matrices** for each model (saved to `outputs/confusion_matrices.png`).
- **Class distribution** plot (saved to `outputs/class_distribution.png`).

## Usage

### Requirements
```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn
```

### Running the Notebook
1. Place `healthcare-dataset-stroke-data.csv` in the same directory as `Stroke_Prediction.ipynb`.
2. Open the notebook in JupyterLab or Jupyter Notebook.
3. Execute all cells sequentially.

### Interactive Prediction
After running the notebook, you can use the provided function to test the best model on custom patient data:

```python
predict_stroke_interactive()
```

The function will prompt you to enter patient details (age, glucose level, BMI, etc.) and return a stroke‑risk prediction along with the predicted probability.

### Outputs
All generated plots and the model‑results CSV are saved in the `outputs/` directory:
- `outputs/class_distribution.png`
- `outputs/confusion_matrices.png`
- `outputs/model_results.csv`

## Conclusion
This project demonstrates that, even with a highly imbalanced medical dataset, effective stroke prediction is possible by combining proper preprocessing with resampling techniques (SMOTE) or algorithm‑level adjustments (class weights). Logistic Regression with SMOTE emerged as the best trade‑off between precision and recall, offering a recall of 78%—meaning it captures roughly 4 out of 5 actual stroke cases—while maintaining a usable precision for further clinical investigation.

Future work could explore:
- More sophisticated sampling techniques (e.g., SMOTE‑ENN, ADASYN).
- Ensemble methods or stacking.
- Feature engineering (e.g., interaction terms, risk scores).
- External validation on independent datasets.

---  
*Generated from the Stroke Prediction notebook (Project2).*