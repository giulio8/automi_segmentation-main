import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Open .csv file with the reults to a dataframe
csvPath = 'ctv_data.csv'
df = pd.read_csv(csvPath)
dsc = df['dice'].dropna().values
hd95 = df['hd95'].dropna().values
print(dsc)


# Create boxplots for the dice and hd95 of the three  models in the same figure, model 1 in blue color and model 2 in magenta color
c1 = 'magenta'
fig, axes = plt.subplots(1, 2)
ax1, ax2 = axes.flatten()
fig.suptitle('')
ax1.boxplot(df['dice'], positions=[1], boxprops=dict(color=c1), whiskerprops=dict(color='black'), capprops=dict(color=c1), medianprops=dict(color='black'))
ax1.set_title('DSC')
ax1.set_ylim(0.60, 1.0)

ax2.boxplot(df['hd95'], positions=[1], boxprops=dict(color=c1), whiskerprops=dict(color='black'), capprops=dict(color=c1), medianprops=dict(color='black'))
ax2.set_title('HD95')
ax2.set_ylim(0, 15)
plt.show()