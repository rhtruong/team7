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
    "GPA"
]

for c in cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[c], kde=True)
    plt.title(f"Distribution of {c}")
    plt.show()
#Thourgh observation, I found that the distribution of Pyhsical_Activity_Hours_Per_Day is right skewed, but it is reasonable because in reality,few studetents will do sports out of 8 hours in a day.
#As for other distributions, they are reasonable to the real life. Also, I don't find outlier inside all distributions. Those distributions are suitbale for standardization and clustering.
#Then, I drew a correlation heatmap
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
#From the correlation heatmap, most variables show weak to moderate correlations, indicating that they capture different aspects of students’ lifestyles. This diversity is beneficial for constructing a K-means model, as it allows the algorithm to identify distinct clusters.