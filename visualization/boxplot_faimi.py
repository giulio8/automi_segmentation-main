import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Open .csv file with the reults to a dataframe
csvPath = '/home/ricardo/Desktop/region_metric_dice.csv'
df = pd.read_csv(csvPath)
male_patients = df.loc[df['sex'] == 'M']
female_patients = df.loc[df['sex'] == 'F']

# Get the dice score of all the 4 regions for male and female patients
female_dsc_head = female_patients['head_neck_dsc'].dropna().values
male_dsc_head = male_patients['head_neck_dsc'].dropna().values

female_dsc_thorax = female_patients['thorax_dsc'].dropna().values
male_dsc_thorax = male_patients['thorax_dsc'].dropna().values

female_dsc_abdomen = female_patients['abdomen_dsc'].dropna().values
male_dsc_abdomen = male_patients['abdomen_dsc'].dropna().values

female_dsc_pelvic = female_patients['pelvic_dsc'].dropna().values
male_dsc_pelvic = male_patients['pelvic_dsc'].dropna().values



c1 = 'magenta'
c2 = 'blue'
# fig, ax = plt.subplots(1, 1)
# fig.suptitle('')
# ax.boxplot(female_dsc_head, positions=[1], boxprops=dict(color=c1), whiskerprops=dict(color='black'), capprops=dict(color=c1), medianprops=dict(color='black'))
# ax.boxplot(male_dsc_head, positions=[2], boxprops=dict(color=c2), whiskerprops=dict(color='black'), capprops=dict(color=c2), medianprops=dict(color='black'))
#
# ax.boxplot(female_dsc_thorax, positions=[4], boxprops=dict(color=c1), whiskerprops=dict(color='black'), capprops=dict(color=c1), medianprops=dict(color='black'))
# ax.boxplot(male_dsc_thorax, positions=[5], boxprops=dict(color=c2), whiskerprops=dict(color='black'), capprops=dict(color=c2), medianprops=dict(color='black'))
#
# ax.boxplot(female_dsc_abdomen, positions=[7], boxprops=dict(color=c1), whiskerprops=dict(color='black'), capprops=dict(color=c1), medianprops=dict(color='black'))
# ax.boxplot(male_dsc_abdomen, positions=[8], boxprops=dict(color=c2), whiskerprops=dict(color='black'), capprops=dict(color=c2), medianprops=dict(color='black'))
#
# ax.boxplot(female_dsc_pelvic, positions=[10], boxprops=dict(color=c1), whiskerprops=dict(color='black'), capprops=dict(color=c1), medianprops=dict(color='black'))
# ax.boxplot(male_dsc_pelvic, positions=[11], boxprops=dict(color=c2), whiskerprops=dict(color='black'), capprops=dict(color=c2), medianprops=dict(color='black'))
#
# ax.set_title('Region Dice Score - Female vs Male')
# ax.set_ylim(0, 1.0)
#
# plt.show()

data = [female_dsc_pelvic, male_dsc_pelvic,
        female_dsc_abdomen, male_dsc_abdomen,
        female_dsc_thorax, male_dsc_thorax,
        female_dsc_head, male_dsc_head]

colors = [c1, c2] * 4  # Alternate colors for female and male

fig, ax = plt.subplots(1, 1, figsize=(12, 5))

# Plot horizontal boxplots
box = ax.boxplot(data,
                 positions=list(range(len(data))),
                 patch_artist=True,
                 boxprops=dict(color='black'),
                 whiskerprops=dict(color='black'),
                 capprops=dict(color='black'),
                 medianprops=dict(color='black'),
                 vert=False)

# Set colors for each boxplot
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# Set labels and titles
ax.set_yticks([1, 3, 5, 7])
ax.set_yticklabels(['Pelvic', 'Abdomen', 'Thorax', 'Head/Neck'])
ax.set_xlabel('Dice Score')
ax.set_xlim(0, 1.0)

# Show plot
plt.show()