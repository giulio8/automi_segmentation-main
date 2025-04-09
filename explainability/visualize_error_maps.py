import numpy as np
import plotly.graph_objs as go
import nibabel as nib
import pandas as pd



def visualize_error_map(error_map_file_path, metrics_file=None):

    error_map = nib.load(error_map_file_path).get_fdata()
    error_indices = np.argwhere(error_map > 0)  # remove noise
    density = 30000

    # Randomly sample points based on the specified density
    error_indices = error_indices[np.random.choice(len(error_indices), density, replace=False)]


    if metrics_file is not None:
        # Load the .csv metrics file and get the DSC and HD95 values from the dataframe
        metrics_df = pd.read_csv(metrics_file)
        dsc_slices_values = metrics_df['DSC']
        hd95_slices_values = metrics_df['HD95']
        print(f"Metrics file loaded with shape {metrics_df.shape}")
        print(f"DSC values: {dsc_slices_values}")
        print(f"HD95 values: {hd95_slices_values}")
        print(f"Max DSC value: {np.max(dsc_slices_values)}")
        print(f"Min DSC value: {np.min(dsc_slices_values)}")

        # Round the DSC and HD95 values to 2 decimal places and multiply by 100 to get percentage values
        dsc_slices_values = np.round(dsc_slices_values, 2) * 100
        hd95_slices_values = np.round(hd95_slices_values, 2)

        print(f"DSC values: {dsc_slices_values}")
        print(f"HD95 values: {hd95_slices_values}")
        print(f"Max DSC value: {np.max(dsc_slices_values)}")
        print(f"Min DSC value: {np.min(dsc_slices_values)}")

    elif metrics_file is None:
        dsc_slices_values = np.zeros(density)
        hd95_slices_values = np.zeros(density)

    # # Get matrix values for color mapping
    # error_colors = error_map[error_indices[:, 0], error_indices[:, 1], error_indices[:, 2]]

    # Create 3D scatter plot for the DSC values for each slice
    dsc_scatter = go.Scatter3d(
        x=error_indices[:, 0],
        y=error_indices[:, 1],
        z=error_indices[:, 2]*5,
        mode='markers',
        marker=dict(
            size=5,
            opacity=0.6,
            color=dsc_slices_values[error_indices[:, 2]],
            colorscale='inferno',
            cmin=50,
            cmax=100,
            colorbar=dict(title='DSC (%)'),
        ),
        name='Prediction - GT Point Cloud'
    )

    # Create figure
    fig = go.Figure(data=[dsc_scatter])

    # Set layout properties
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        title='3D Error Map Visualization',
    )

    # Show figure
    fig.show()

    # Create 3D scatter plot for the HD95 values for each slice
    hd95_scatter = go.Scatter3d(
        x=error_indices[:, 0],
        y=error_indices[:, 1],
        z=error_indices[:, 2]*5,
        mode='markers',
        marker=dict(
            size=5,
            opacity=0.6,
            color=hd95_slices_values[error_indices[:, 2]],
            colorscale='inferno',
            cmin=0,
            cmax=20,
            colorbar=dict(title='HD95 (mm)'),
        ),
        name='Prediction - GT Point Cloud'
    )

    # Create figure
    fig = go.Figure(data=[hd95_scatter])

    # Set layout properties
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        title='3D Error Map Visualization',
    )

    # Show figure
    fig.show()


if __name__ == "__main__":
    error_map_file_path = "/home/ricardo/Desktop/CTV_LN_1_error_maps/AUTOMI_00039.nii.gz"
    visualize_error_map(error_map_file_path, metrics_file="/home/ricardo/Desktop/CTV_LN_1_error_maps/AUTOMI_00039.nii.gz_metrics.csv")