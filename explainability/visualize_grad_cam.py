import nibabel
import matplotlib.pyplot as plt

# Visualize the gradient class activation map (Grad-CAM) for a given slice

def visualize_grad_cam(grad_cam_path, slice):
    grad_cam = nibabel.load(grad_cam_path).get_fdata()
    print("max:", grad_cam.max())
    print("min:", grad_cam.min())
    print("mean:", grad_cam.mean())
    print("shape:", grad_cam.shape)

    grad_cam = grad_cam[:, :, slice]

    plt.imshow(grad_cam, cmap='hot', interpolation='nearest')
    plt.axis('off')
    plt.show()


# Example usage
grad_cam_path = '/home/ricardo/Desktop/af36802d.nii.gz'
slice =10
visualize_grad_cam(grad_cam_path, slice)
