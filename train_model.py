import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# -----------------------------
# 1. Load and Initial Cleaning
# -----------------------------
print("Loading data...")
df = pd.read_csv("laptops.csv")
print(f"Initial dataset shape: {df.shape}")

# Remove rows with missing target
df = df.dropna(subset=['Price'])
print(f"After removing missing prices: {df.shape}")

# -----------------------------
# 2. Clean numeric columns
# -----------------------------
numeric_cols = ['num_cores', 'num_threads', 'ram_memory', 'primary_storage_capacity',
                'secondary_storage_capacity', 'display_size', 'resolution_width',
                'resolution_height', 'year_of_warranty']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Fill missing categorical values
categorical_cols = ['brand', 'processor_brand', 'processor_tier', 'primary_storage_type',
                    'secondary_storage_type', 'gpu_brand', 'gpu_type', 'OS', 'is_touch_screen']

for col in categorical_cols:
    df[col] = df[col].fillna("Unknown")

# -----------------------------
# 3. Feature Engineering
# -----------------------------
print("Creating engineered features...")
df['total_storage'] = df['primary_storage_capacity'] + df['secondary_storage_capacity']
df['resolution_total'] = df['resolution_width'] * df['resolution_height']
df['cores_per_thread_ratio'] = df['num_cores'] / (df['num_threads'] + 1)
df['ram_per_core'] = df['ram_memory'] / (df['num_cores'] + 1)
df['has_secondary_storage'] = (df['secondary_storage_capacity'] > 0).astype(int)

# Add engineered features to numeric columns
engineered_features = ['total_storage', 'resolution_total', 'cores_per_thread_ratio', 
                       'ram_per_core', 'has_secondary_storage']
numeric_cols_extended = numeric_cols + engineered_features

# -----------------------------
# 4. Outlier Removal
# -----------------------------
print("Removing outliers...")
Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Price'] >= Q1 - 1.5*IQR) & (df['Price'] <= Q3 + 1.5*IQR)]
print(f"After outlier removal: {df.shape}")

# -----------------------------
# 5. Features and target
# -----------------------------
features = numeric_cols_extended + categorical_cols
target = 'Price'

X = df[features]
y = df[target]

# -----------------------------
# 6. Split dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# -----------------------------
# 7. Preprocessing Pipeline
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols_extended),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

# -----------------------------
# 8. Create Model Pipeline
# -----------------------------
print("\nTraining base model...")
base_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    ))
])

# -----------------------------
# 9. Cross-Validation
# -----------------------------
print("Performing cross-validation...")
cv_scores = cross_val_score(base_model, X_train, y_train, 
                            cv=5, scoring='r2', n_jobs=-1)
print(f"CV R² Scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# -----------------------------
# 10. Hyperparameter Tuning (Optional - Comment out if too slow)
# -----------------------------
print("\nStarting hyperparameter tuning...")
param_grid = {
    'regressor__n_estimators': [500, 1000],
    'regressor__max_depth': [5, 6, 7],
    'regressor__learning_rate': [0.05, 0.1],
}

grid_search = GridSearchCV(
    base_model, param_grid, cv=3, 
    scoring='r2', n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV R² score: {grid_search.best_score_:.4f}")

# Use best model
model = grid_search.best_estimator_

# -----------------------------
# 11. Evaluate Model
# -----------------------------
print("\n" + "="*50)
print("FINAL MODEL EVALUATION")
print("="*50)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Training metrics
train_r2 = r2_score(y_train, y_pred_train)
train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))

# Test metrics
test_r2 = r2_score(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

# MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

test_mape = mean_absolute_percentage_error(y_test, y_pred_test)

print("\nTraining Metrics:")
print(f"  R² Score: {train_r2:.4f}")
print(f"  MAE: ₹{train_mae:,.2f}")
print(f"  RMSE: ₹{train_rmse:,.2f}")

print("\nTest Metrics:")
print(f"  R² Score: {test_r2:.4f}")
print(f"  MAE: ₹{test_mae:,.2f}")
print(f"  RMSE: ₹{test_rmse:,.2f}")
print(f"  MAPE: {test_mape:.2f}%")

# -----------------------------
# 12. Save Model and Metadata
# -----------------------------
print("\nSaving model and metadata...")

# Get unique values for categorical features
categorical_info = {}
for col in categorical_cols:
    categorical_info[col] = sorted(df[col].unique().tolist())

# Get feature ranges for numeric features
numeric_info = {}
for col in numeric_cols:
    numeric_info[col] = {
        'min': float(df[col].min()),
        'max': float(df[col].max()),
        'mean': float(df[col].mean())
    }

model_package = {
    'model': model,
    'feature_names': features,
    'numeric_cols': numeric_cols_extended,
    'categorical_cols': categorical_cols,
    'categorical_info': categorical_info,
    'numeric_info': numeric_info,
    'metrics': {
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_mape': test_mape
    }
}

joblib.dump(model_package, "laptop_price_model_complete.pkl")
print("✓ Model saved as 'laptop_price_model_complete.pkl'")

# -----------------------------
# 13. Visualizations
# -----------------------------
print("\nGenerating visualizations...")

# Figure 1: Prediction vs Actual
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Test set
axes[0].scatter(y_test, y_pred_test, alpha=0.5, s=30)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel("Actual Price (₹)", fontsize=12)
axes[0].set_ylabel("Predicted Price (₹)", fontsize=12)
axes[0].set_title(f"Test Set: Actual vs Predicted\nR² = {test_r2:.4f}", fontsize=14)
axes[0].grid(True, alpha=0.3)

# Training set
axes[1].scatter(y_train, y_pred_train, alpha=0.5, s=30, color='green')
axes[1].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[1].set_xlabel("Actual Price (₹)", fontsize=12)
axes[1].set_ylabel("Predicted Price (₹)", fontsize=12)
axes[1].set_title(f"Training Set: Actual vs Predicted\nR² = {train_r2:.4f}", fontsize=14)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('prediction_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Figure 2: Residual Analysis
residuals = y_test - y_pred_test

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Residual plot
axes[0].scatter(y_pred_test, residuals, alpha=0.5, s=30)
axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0].set_xlabel("Predicted Price (₹)", fontsize=12)
axes[0].set_ylabel("Residuals (₹)", fontsize=12)
axes[0].set_title("Residual Plot", fontsize=14)
axes[0].grid(True, alpha=0.3)

# Residual distribution
axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel("Residuals (₹)", fontsize=12)
axes[1].set_ylabel("Frequency", fontsize=12)
axes[1].set_title("Residual Distribution", fontsize=14)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('residual_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Figure 3: Feature Importance
ohe_features = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_features = numeric_cols_extended + list(ohe_features)

xgb_model = model.named_steps['regressor']
importances = xgb_model.feature_importances_
indices = np.argsort(importances)[-25:]  # top 25 features

plt.figure(figsize=(12, 10))
plt.barh(range(len(indices)), importances[indices], align='center', color='steelblue')
plt.yticks(range(len(indices)), [all_features[i] for i in indices], fontsize=10)
plt.xlabel("Feature Importance", fontsize=12)
plt.title("Top 25 Most Important Features", fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*50)
print("✓ Training complete!")
print("✓ Model saved successfully")
print("✓ Visualizations saved")
print("="*50)