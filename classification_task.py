
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

data = load_wine(as_frame=True)
X, y = data.data, data.target

print("Shape:", X.shape)
print("Feature columns:", list(X.columns))
print("Target classes:", np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = [
    ("Logistic Regression",
     Pipeline([
         ("scaler", StandardScaler()),
         ("clf", LogisticRegression(max_iter=1000, random_state=42))
     ])),
    ("SVM (RBF)",
     Pipeline([
         ("scaler", StandardScaler()),
         ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42))
     ])),
    ("KNN Classifier",
     Pipeline([
         ("scaler", StandardScaler()),
         ("clf", KNeighborsClassifier(n_neighbors=5, weights="uniform"))
     ])),
]

def evaluate_classification(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted")
    print(f"\n{name} Results")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score (weighted): {f1:.4f}")
    print("Classification report:\n", classification_report(y_true, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    return {"Model": name, "Accuracy": acc, "F1-Score": f1}

results = []
predictions = {}

for name, pipe in models:
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    predictions[name] = y_pred
    results.append(evaluate_classification(name, y_test, y_pred))

results_df = pd.DataFrame(results).sort_values(["Accuracy", "F1-Score"], ascending=[False, False])
print("\n\nComparison Table (sorted by Accuracy):")
print(results_df)

best = results_df.iloc[0]
print("\nBest Model Analysis:")
print(
    f"The best performing model is **{best['Model']}** "
    f"with Accuracy = {best['Accuracy']:.4f} and F1-Score = {best['F1-Score']:.4f}."
)

plt.figure(figsize=(6,4))
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.title("Wine Dataset — Accuracy by Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
plt.xticks(rotation=10)
plt.tight_layout()
plt.show()
