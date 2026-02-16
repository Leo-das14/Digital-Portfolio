# ===============================
# 1. Import Required Libraries
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import isodate
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# ===============================
# 2. Load Dataset
# ===============================

data = pd.read_csv("youtube_channel_data.csv")

print("Dataset Info:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())


# ===============================
# 3. Data Cleaning
# ===============================

# Drop missing values
data = data.dropna()
data.reset_index(drop=True, inplace=True)

# Convert Video Duration (ISO 8601 format like PT5M30S) into seconds
if 'Video Duration' in data.columns:
    data['Video Duration'] = data['Video Duration'].apply(
        lambda x: isodate.parse_duration(x).total_seconds()
    )


# ===============================
# 4. Feature Engineering
# ===============================

# Revenue per View
data['Revenue per View'] = np.where(
    data['Views'] != 0,
    data['Estimated Revenue (USD)'] / data['Views'],
    0
)

# Engagement Rate (%)
data['Engagement Rate'] = np.where(
    data['Views'] != 0,
    (data['Likes'] + data['Shares'] + data['Comments']) / data['Views'] * 100,
    0
)


# ===============================
# 5. Exploratory Data Analysis
# ===============================

# Revenue Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data['Estimated Revenue (USD)'], bins=30, kde=True)
plt.title("Revenue Distribution")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# ===============================
# 6. Prepare Data for Model
# ===============================

features = [
    'Views',
    'Subscribers',
    'Likes',
    'Shares',
    'Comments',
    'Engagement Rate'
]

target = 'Estimated Revenue (USD)'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ===============================
# 7. Train Model
# ===============================

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# ===============================
# 8. Model Evaluation
# ===============================

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")


# ===============================
# 9. Feature Importance
# ===============================

importances = model.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title("Feature Importance")
plt.show()


# ===============================
# 10. Save Model
# ===============================

joblib.dump(model, "youtube_revenue_model.pkl")
print("Model Saved Successfully!")









