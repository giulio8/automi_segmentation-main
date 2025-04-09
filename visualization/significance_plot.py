import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ranksums
import numpy as np

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

# Function to perform Wilcoxon Rank-Sum Test and create significance matrix for male vs. female
def create_gender_significance_matrix(metric):
    p_values = []

    for model in models:
        data_model = data[data['Model'] == model]
        data_male = data_model[data_model['gender'] == 'M'][metric]
        data_female = data_model[data_model['gender'] == 'F'][metric]
        _, p_value = ranksums(data_male, data_female)
        p_values.append(p_value)

    return pd.DataFrame(p_values, index=models, columns=['p-value'])

# Create significance matrix for Dice Score
significance_matrix_dice = create_gender_significance_matrix('dice')

# Plot the significance matrix for Dice Score
plt.figure(figsize=(8, 6))
sns.heatmap(significance_matrix_dice, annot=True, cmap="coolwarm", cbar=True, center=0, vmin=0, vmax=0.05)
plt.title("Gender Significance Matrix for Dice Score (p-values)")
plt.show()

# Create significance matrix for HD95
significance_matrix_hd95 = create_gender_significance_matrix('hd95')

# Plot the significance matrix for HD95
plt.figure(figsize=(8, 6))
sns.heatmap(significance_matrix_hd95, annot=True, cmap="coolwarm", cbar=True, center=0, vmin=0, vmax=0.05)
plt.title("Gender Significance Matrix for HD95 (p-values)")
plt.show()
