import pandas as pd
import numpy as np
df = pd.read_csv("student_lifestyle_dataset.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isna().sum())
print(df.isna().mean())
print(df.duplicated().sum())
mapping={
    "Low":1,
    "Moderate":2,
    "High":3
}
df["Stress_Level"]=df["Stress_Level"].map(mapping)
print(df)

#Exploratory Data Analysis(EDA)
#First,I found all the histograms of varibles
import matplotlib.pyplot as plt
import seaborn as sns
cols = [
    "Study_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day",
    "GPA",
    "Stress_Level"
]
#
plt.figure(figsize=(10,8))
sns.heatmap(df[cols].corr(), annot=True, cmap="coolwarm")
#Then I draw a correlation heatmap
plt.title("Correlation Heatmap")
plt.show()

#standarization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[cols] = scaler.fit_transform(df[cols])
X_scaled=df[cols]
print(df.head())

#choosing k values
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

K_range = range(2, 8)  

inertias = []
sil_scores = []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X_scaled)

    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

# ---- Elbow Plot ----
plt.figure(figsize=(6,4))
plt.plot(list(K_range), inertias, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (SSE)")
plt.title("Elbow Method")
plt.show()

# ---- Silhouette Plot ----
plt.figure(figsize=(6,4))
plt.plot(list(K_range), sil_scores, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method")
plt.show()

print("k values:", list(K_range))
print("inertias:", inertias)
print("silhouette scores:", sil_scores)


from sklearn.cluster import KMeans
import pandas as pd

for k in [2,3,5]:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X_scaled)
    df[f"cluster_{k}"] = labels
    
    print("\n===== k =", k, "=====")
    print(df.groupby(f"cluster_{k}")[cols].mean())

#Final K-means with k=3
from sklearn.cluster import KMeans
k = 3
km = KMeans(n_clusters=k, random_state=42, n_init="auto")
labels = km.fit_predict(X_scaled)
df["cluster"] = labels
print(df.tail())
cluster_profile = df.groupby("cluster")[cols].mean()
print(cluster_profile)

#Draw PCA plot to show clusters
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

df["PC1"] = pca_result[:, 0]
df["PC2"] = pca_result[:, 1]

plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x="PC1", y="PC2", hue="cluster", palette="viridis")
plt.title("K-means Clusters Visualized with PCA")
plt.show()

cluster_profile = df.groupby("cluster")[cols].mean()
print(cluster_profile.to_string())


