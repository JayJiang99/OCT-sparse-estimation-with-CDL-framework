import numpy as np
import pickle
from scipy.io import loadmat

if __name__ == '__main__':
    # Read .mat file
    mat_data = loadmat('../Data/partial_LK2_50.mat')  # Replace 'your_file.mat' with your file name

    # Assuming the variable name inside the .mat file is 'variable_name'
    numpy_array = mat_data['partial_LK2']

    # # Save the NumPy array using numpy.save()
    # np.save('partial_LK2', numpy_array)
    # Save the NumPy array using pickle.dump()
    with open('../Data/partial_LK2_50', 'wb') as file:
        pickle.dump(numpy_array, file)
    
