
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


# model1_data = pd.read_csv('unet.csv')
# model2_data = pd.read_csv('nnunet2d.csv')
# model3_data = pd.read_csv('nnunet3d.csv')
model1_data = pd.read_csv('unet_nobones.csv')
model2_data = pd.read_csv('nnunet2d_nobones.csv')
model3_data = pd.read_csv('nnunet3d_nobones.csv')

# dice_t_statistic, dice_p_value = stats.ttest_ind(model1_data['dice'], model2_data['dice'])
# hd_t_statistic, hd_p_value = stats.ttest_ind(model1_data['hd'], model2_data['hd'])
# hd95_t_statistic, hd95_p_value = stats.ttest_ind(model1_data['hd95'], model2_data['hd95'])

# Calculates and prints the mean and standard deviation of the dice and hd95 of both models
model1_mean_dice = model1_data['dice'].mean()
model1_std_dice = model1_data['dice'].std()
model1_mean_hd95 = model1_data['hd95'].mean()
model1_std_hd95 = model1_data['hd95'].std()

model2_mean_dice = model2_data['dice'].mean()
model2_std_dice = model2_data['dice'].std()
model2_mean_hd95 = model2_data['hd95'].mean()
model2_std_hd95 = model2_data['hd95'].std()

model1_min_dice = model1_data['dice'].min()
model1_max_dice = model1_data['dice'].max()
model1_min_hd95 = model1_data['hd95'].min()
model1_max_hd95 = model1_data['hd95'].max()

model2_min_dice = model2_data['dice'].min()
model2_max_dice = model2_data['dice'].max()
model2_min_hd95 = model2_data['hd95'].min()
model2_max_hd95 = model2_data['hd95'].max()

model3_mean_dice = model3_data['dice'].mean()
model3_std_dice = model3_data['dice'].std()
model3_mean_hd95 = model3_data['hd95'].mean()
model3_std_hd95 = model3_data['hd95'].std()

model3_min_dice = model3_data['dice'].min()
model3_max_dice = model3_data['dice'].max()
model3_min_hd95 = model3_data['hd95'].min()
model3_max_hd95 = model3_data['hd95'].max()

print(f'Model 1 DSC: {model1_mean_dice} +- {model1_std_dice}')
print(f'Model 1 HD95: {model1_mean_hd95} +- {model1_std_hd95}')

print(f'Model 2 DSC: {model2_mean_dice} +- {model2_std_dice}')
print(f'Model 2 HD95: {model2_mean_hd95} +- {model2_std_hd95}')

print(f'Model 3 DSC: {model3_mean_dice} +- {model3_std_dice}')
print(f'Model 3 HD95: {model3_mean_hd95} +- {model3_std_hd95}')

# Prints the minimum and maximum dice and hd95 of both models
print(f'Model 1 DSC min and max: {model1_min_dice, model1_max_dice}')
print(f'Model 1 HD95 min and max: {model1_min_hd95, model1_max_hd95}')

print(f'Model 2 DSC min and max: {model2_min_dice, model2_max_dice}')
print(f'Model 2 HD95 min and max: {model2_min_hd95, model2_max_hd95}')

print(f'Model 3 DSC min and max: {model3_min_dice, model3_max_dice}')
print(f'Model 3 HD95 min and max: {model3_min_hd95, model3_max_hd95}')



# print("DSC P-value: " + str(dice_p_value))
# print("HD95 P-value" + str(hd95_p_value))

# Save the results in a csv file
results = pd.DataFrame({'model1_dice': [model1_mean_dice], 'model1_dice_std': [model1_std_dice], 'model1_hd95': [model1_mean_hd95], 'model1_hd95_std': [model1_std_hd95],
                        'model1_min_dice': [model1_min_dice], 'model1_max_dice': [model1_max_dice], 'model1_min_hd95': [model1_min_hd95], 'model1_max_hd95': [model1_max_hd95],
                        'model2_dice': [model2_mean_dice], 'model2_dice_std': [model2_std_dice], 'model2_hd95': [model2_mean_hd95], 'model2_hd95_std': [model2_std_hd95],
                        'model2_min_dice': [model2_min_dice], 'model2_max_dice': [model2_max_dice], 'model2_min_hd95': [model2_min_hd95], 'model2_max_hd95': [model2_max_hd95],
                        'model3_dice': [model3_mean_dice], 'model3_dice_std': [model3_std_dice], 'model3_hd95': [model3_mean_hd95], 'model3_hd95_std': [model3_std_hd95],
                        'model3_min_dice': [model3_min_dice], 'model3_max_dice': [model3_max_dice], 'model3_min_hd95': [model3_min_hd95], 'model3_max_hd95': [model3_max_hd95]})
results.to_csv('unet_nnunet2d_nnunet3d_bones.csv', index=False)

# Create boxplots for the dice and hd95 of the three  models in the same figure, model 1 in blue color and model 2 in magenta color
c1 = 'blue'
c2 = 'darkcyan'
c3 = 'magenta'
fig, ax1, ax2 = plt.subplots(1, 2)
fig.suptitle('')
ax1.boxplot(model1_data['dice'], positions=[1], boxprops=dict(color=c1), whiskerprops=dict(color='black'), capprops=dict(color=c1), medianprops=dict(color='black'))
ax1.boxplot(model2_data['dice'], positions=[2], boxprops=dict(color=c2), whiskerprops=dict(color='black'), capprops=dict(color=c2), medianprops=dict(color='black'))
ax1.boxplot(model3_data['dice'], positions=[3], boxprops=dict(color=c3), whiskerprops=dict(color='black'), capprops=dict(color=c3), medianprops=dict(color='black'))
ax1.set_title('DSC')
ax1.set_ylim(0.60, 1.0)
ax1.set_xticklabels(['U-Net', 'nnU-Net 2D', 'nnU-Net 3D'], fontsize=12)
ax2.boxplot(model1_data['hd95'], positions=[1], boxprops=dict(color=c1), whiskerprops=dict(color='black'), capprops=dict(color=c1), medianprops=dict(color='black'))
ax2.boxplot(model2_data['hd95'], positions=[2], boxprops=dict(color=c2), whiskerprops=dict(color='black'), capprops=dict(color=c2), medianprops=dict(color='black'))
ax2.boxplot(model3_data['hd95'], positions=[3], boxprops=dict(color=c3), whiskerprops=dict(color='black'), capprops=dict(color=c3), medianprops=dict(color='black'))
ax2.set_title('HD95')
ax2.set_ylim(0, 40)
ax2.set_xticklabels(['U-Net', 'nnU-Net 2D', 'nnU-Net 3D'], fontsize=8)
plt.show()


