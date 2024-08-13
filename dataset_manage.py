import pandas as pd
from sklearn.model_selection import train_test_split

file_path = './mnt/data/dataset_B_05_2020.csv'
df = pd.read_csv(file_path)

print(df.head())

df_phishing = df[df['status'] == 'phishing']
df_legitimate = df[df['status'] == 'legitimate']

subset_phishing = df_phishing.sample(n=500, random_state=42)
subset_legitimate = df_legitimate.sample(n=500, random_state=42)
subset = pd.concat([subset_phishing, subset_legitimate])

remaining_phishing = df_phishing.drop(subset_phishing.index)
remaining_legitimate = df_legitimate.drop(subset_legitimate.index)
remaining = pd.concat([remaining_phishing, remaining_legitimate])

train, validation = train_test_split(remaining, test_size=0.1, random_state=42)

print(f"Subset size: {subset.shape[0]}")
print(f"Training set size: {train.shape[0]}")
print(f"Validation set size: {validation.shape[0]}")

# Save the subsets to CSV files if needed
subset.to_csv('./mnt/data/phishing_subset.csv', index=False)
train.to_csv('./mnt/data/phishing_train.csv', index=False)
validation.to_csv('./mnt/data/phishing_validation.csv', index=False)

print("Datasets have been saved successfully.")

