# ==========================================================
# STUDENT DEPRESSION PREDICTION – FINAL CORRECT VERSION
# ==========================================================

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

print("\n===== STUDENT DEPRESSION ML PROJECT =====\n")

# ==========================================================
# 1. LOAD DATA
# ==========================================================

df = pd.read_csv("Student Depression Dataset.csv")

# Drop ID column if exists
df.drop(columns=["id"], errors="ignore", inplace=True)

# ==========================================================
# 2. HANDLE MISSING VALUES
# ==========================================================

# Fill numeric columns with median
num_cols = df.select_dtypes(include=["int64","float64"]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill categorical columns with mode
cat_cols = df.select_dtypes(include=["object", "string"]).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ==========================================================
# 3. ENCODE CATEGORICAL VARIABLES
# ==========================================================

df = pd.get_dummies(df, drop_first=True)

# Ensure target is integer
df["Depression"] = df["Depression"].astype(int)

# ==========================================================
# 4. SPLIT FEATURES & TARGET
# ==========================================================

X = df.drop("Depression", axis=1)
y = df["Depression"]

# ==========================================================
# 5. FEATURE SCALING
# ==========================================================

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ==========================================================
# 6. TRAIN TEST SPLIT (STRATIFIED)
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================================
# 7. DEFINE 5 MODELS
# ==========================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=3000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(kernel="rbf"),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB()
}

results = []

print("Training Models...\n")

# ==========================================================
# 8. TRAIN & EVALUATE
# ==========================================================

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    print(f"{name}")
    print(f"   Accuracy : {acc*100:.2f}%")
    print(f"   F1 Score : {f1:.3f}")
    print("----------------------------------")

    results.append((name, f1))

# ==========================================================
# 9. SELECT BEST MODEL
# ==========================================================

best_model_name = max(results, key=lambda x: x[1])[0]
best_model = models[best_model_name]

print(f"\nBest Model Selected: {best_model_name}")

# ==========================================================
# 10. FINAL REPORT
# ==========================================================

final_pred = best_model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, final_pred))

# ==========================================================
# 11. SAVE BEST MODEL
# ==========================================================

pickle.dump(best_model, open("best_depression_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("\nModel & Scaler Saved Successfully ✅")
print("\n===== PROJECT COMPLETED SUCCESSFULLY =====\n")

