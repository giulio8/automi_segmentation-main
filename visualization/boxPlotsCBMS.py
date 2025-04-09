import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Open .csv file with the reults to a dataframe
csvPath = 'cbmsBoxplot.csv'
df = pd.read_csv(csvPath)
dsc_bcel = df['DSC BCEL'].dropna().values
dsc_dl = df['DSC DL'].dropna().values
hd95_bcel = df['HD95 BCEL'].dropna().values
hd95_dl = df['HD95 DL'].dropna().values

plt.figure(figsize=(15,10))
data = plt.boxplot( positions=[0, 1], labels=['DSC BCEL', 'DSC DL'],
                   x=[dsc_bcel, dsc_dl])
#plt.title('DSC Comparison of BCEL and DL', fontsize=30)
plt.xticks(size = 17)
plt.yticks(size = 25)
plt.show()

plt.figure(figsize=(15,10))
data = plt.boxplot( positions=[0, 1], labels=['HD95 BCEL', 'HD95 DL'],
                   x=[hd95_bcel, hd95_dl])
#plt.title('HD95 Comparison of BCEL and DL', fontsize=30)
plt.xticks(size = 17)
plt.yticks(size = 25)
plt.show()