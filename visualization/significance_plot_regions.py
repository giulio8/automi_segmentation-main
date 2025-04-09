import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ranksums
import numpy as np

# Define the filenames and models
filenames = ['dataset_003_regions.csv', 'dataset_005_regions.csv', 'dataset_006_regions.csv',
             'dataset_007_regions.csv', 'dataset_009_regions.csv', 'dataset_010_regions.csv']
models = ['Base', 'MI-Z', 'EQ-Z', 'Crop-Z', 'MI', 'MI-TS']
regions = ['head_neck_dsc', 'torso_dsc', 'abdomen_dsc', 'pelvic_dsc']

# Initialize a dictionary to store data
data = {model: {} for model in models}

# Read and organize data from CSV files
for i, filename in enumerate(filenames):
    df = pd.read_csv(filename)
    model = models[i]
    for region in regions:
        data[model][region] = df[['Sex', region]]


# Function to perform Wilcoxon Rank-Sum Test and create significance matrix
def create_gender_significance_matrix(data, models, regions):
    p_values = []

    for model in models:
        for region in regions:
            data_model_region = data[model][region]
            data_male = data_model_region[data_model_region['Sex'] == 'M'][region]
            data_female = data_model_region[data_model_region['Sex'] == 'F'][region]
            _, p_value = ranksums(data_male, data_female)
            p_values.append(p_value)

    # Reshape the p-values into a matrix
    p_values_matrix = pd.DataFrame(
        np.array(p_values).reshape(len(models), len(regions)),
        index=models,
        columns=regions
    )
    return p_values_matrix


# Create significance matrix
significance_matrix = create_gender_significance_matrix(data, models, regions)

# Plot the significance matrix heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(significance_matrix, annot=True, cmap="coolwarm", cbar=True, center=0, vmin=0, vmax=0.05)
plt.title("Gender Significance Matrix for Dice Score by Region and Model (p-values)")
plt.show()