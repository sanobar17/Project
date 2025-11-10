# main.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# 1. Load Data
# -----------------------------
# Example dataset (replace with real macroeconomic data)
data = {
    "GDP_growth": [2.5, 3.0, 1.8, 2.2, 3.5, 2.9, 1.5, 2.7],
    "Inflation": [1.2, 1.5, 2.0, 1.8, 2.5, 2.1, 1.7, 2.3],
    "Unemployment": [5.0, 4.8, 5.2, 5.1, 4.7, 4.9, 5.3, 4.6],
    "Interest_rate": [0.5, 0.7, 1.0, 0.8, 1.2, 1.1, 0.9, 1.3],
    "Gov_spending": [200, 210, 190, 205, 220, 215, 195, 225]
}
df = pd.DataFrame(data)

# -----------------------------
# 2. Features & Target
# -----------------------------
X = df[["Inflation", "Unemployment", "Interest_rate", "Gov_spending"]]
y = df["GDP_growth"]

# -----------------------------
# 3. Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -----------------------------
# 4. Regression Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# 5. Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 6. Evaluation
# -----------------------------
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# -----------------------------
# 7. Visualization
# -----------------------------
plt.scatter(y_test, y_pred, color="blue")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
plt.xlabel("Actual GDP Growth")
plt.ylabel("Predicted GDP Growth")
plt.title("Forecasting Economic Growth using Regression")
plt.show()


# main.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# -----------------------------
# 1. Load Data
# -----------------------------
# Example dataset (replace with real bank + macroeconomic data)
data = {
    "Default": [0, 1, 0, 0, 1, 0, 1, 0],
    "Income": [3500, 2200, 5200, 4000, 1800, 6000, 2500, 4500],
    "Debt_to_income": [0.35, 0.65, 0.28, 0.40, 0.70, 0.25, 0.55, 0.30],
    "Credit_score": [690, 620, 720, 710, 580, 750, 600, 705],
    "Loan_amount": [12000, 8000, 15000, 10000, 7000, 20000, 9000, 11000],
    "Unemployment_rate": [5.0, 5.2, 4.8, 5.1, 5.5, 4.7, 5.3, 4.9],
    "Inflation_rate": [2.0, 2.5, 1.8, 2.2, 2.7, 1.9, 2.4, 2.1],
    "Interest_rate": [1.0, 1.2, 0.8, 1.1, 1.3, 0.9, 1.2, 1.0],
    "GDP_growth": [2.5, 1.8, 3.0, 2.2, 1.5, 3.2, 1.7, 2.8]
}
df = pd.DataFrame(data)

# -----------------------------
# 2. Features & Target
# -----------------------------
X = df.drop("Default", axis=1)
y = df["Default"]

# -----------------------------
# 3. Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -----------------------------
# 4. Logistic Regression Model
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------
# 5. Predictions
# -----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# -----------------------------
# 6. Evaluation
# -----------------------------
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# -----------------------------
# 7. ROC Curve Visualization
# -----------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label="Logistic Regression")
plt.plot([0,1],[0,1],"--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Default Forecast")
plt.legend()
plt.show()
