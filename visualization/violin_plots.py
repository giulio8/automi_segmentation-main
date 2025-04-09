import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the filenames and models
filenames = ['dataset_003.csv', 'dataset_005.csv', 'dataset_006.csv', 'dataset_007.csv', 'dataset_009.csv', 'dataset_010.csv']
models = ['Base', 'MI-Z', 'EQ-Z', 'Crop-Z', 'MI', 'MI-TS']

# Initialize lists to store data
data = []

# Read and concatenate data from CSV files
for i, filename in enumerate(filenames):
    df = pd.read_csv(filename)
    df['Model'] = models[i]
    data.append(df)

# Concatenate all dataframes
data = pd.concat(data, ignore_index=True)

# Plot settings
sns.set(style="whitegrid")

# Create a violin plot for Dice Score with individual points
plt.figure(figsize=(14, 7))
sns.violinplot(x="Model", y="dice", hue="gender", data=data, split=True, inner="quartile", palette={"M": "blue", "F": "magenta"})
sns.stripplot(x="Model", y="dice", hue="gender", data=data, dodge=True, palette={"M": "black", "F": "black"}, alpha=0.5, jitter=True)
plt.ylim(0.65, 1)
plt.title("Dice Score Distribution by Model and Gender")
plt.ylabel("Dice Score")
plt.legend(title="Gender", bbox_to_anchor=(1.05, 1), loc=2)
plt.show()

# Create a violin plot for Hausdorff Distance 95 (HD95) with individual points
plt.figure(figsize=(14, 7))
sns.violinplot(x="Model", y="hd95", hue="gender", data=data, split=True, inner="quartile", palette={"M": "blue", "F": "magenta"})
sns.stripplot(x="Model", y="hd95", hue="gender", data=data, dodge=True, palette={"M": "black", "F": "black"}, alpha=0.5, jitter=True)
plt.ylim(0, 13)
plt.title("Hausdorff Distance 95 Distribution by Model and Gender")
plt.ylabel("Hausdorff Distance 95 (mm)")
plt.legend(title="Gender", bbox_to_anchor=(1.05, 1), loc=2)
plt.show()