import pandas as pd
import numpy as np
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv("student_burnout_dropout_dataset_2.csv")

# Features used for student segmentation
features = [
    "Attendance_Percent",
    "Study_Hours_Per_Day",
    "Previous_GPA",
    "Backlogs",
    "Sleep_Hours",
    "Screen_Time_Hours",
    "Exercise_Freq_Per_Week",
    "Social_Activity_Score",
    "Financial_Stress_Score",
    "Family_Support_Score",
    "Stress_Level",
    "Anxiety_Score",
    "Motivation_Score",
    "Peer_Pressure_Score"
]

# Prepare data
X = df[features].copy()

# Handle missing values
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Select number of clusters
best_score = -1
best_k = 2
best_model = None

for k in range(2, 7):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    if score > best_score:
        best_score = score
        best_k = k
        best_model = model

# Assign student segments
df["Student_Segment"] = best_model.labels_

# Save results
df.to_csv("student_segments.csv", index=False)

joblib.dump(
    {
        "model": best_model,
        "scaler": scaler,
        "imputer": imputer,
        "features": features
    },
    "student_segmentation_model.pkl"
)

print("\nBurnoutIQ Student Segmentation")
print("--------------------------------")
print("Optimal Clusters:", best_k)
print("Silhouette Score:", round(best_score, 3))

print("\nStudents per Segment:")
print(df["Student_Segment"].value_counts().sort_index())

print("\nSegmentation completed successfully!")