import pandas as pd

df = pd.read_csv('student_lifestyle_dataset.csv')

print(df.head())

if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# Convert stress levels to numerical values
# Low -> 1, Moderate -> 2, High -> 3
stress_mapping = {'Low': 1, 'Moderate': 2, 'High': 3}
if 'Stress_Level' in df.columns:
    df['Stress_Level'] = df['Stress_Level'].map(stress_mapping)

df.to_csv('cleaned_data.csv', index=False)

print(df.head())