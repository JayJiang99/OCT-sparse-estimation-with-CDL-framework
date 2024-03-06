import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sporco.admm import cbpdn
from skimage import filters
from skimage.morphology import disk
from misc import processing


file_name = ['finger']
print(file_name)
# Load the example dataset
s= processing.load_data(file_name[0], decimation_factor=1, data_only=True)
print(np.shape(s))
rvmin, vmax = 5, 55  # dB

original_array = 20 * np.log10(abs(s))
original_array = processing.imag2uint(original_array, rvmin, vmax)

reshaped_array = np.abs(original_array.reshape((330, 512,20)))
# reshaped_array = np.abs(original_array.reshape((20, 512,330)))

# Display an XY plane image
xy_plane_image = reshaped_array[:,:,2]  # Assuming you want to display the first XY plane

# Now you can use any image display library to visualize xy_plane_image
# For example, using Matplotlib:
import matplotlib.pyplot as plt
plt.imshow(xy_plane_image,cmap='gray', vmax=255, vmin=0)
plt.show()