import numpy as np
from skimage.transform import resize
import math
import cv2
from matplotlib import pyplot as plt

class PointSpreadFunction:
    def __init__(self, radius,z_0, coeff, scale_factor = 1,estimated_axial_psf = None):
        """
        Initialize the PointSpreadFunction class with specific parameters.

        :param radius: The radius of the PSF.
        :param z_0: Axial position parameter in index format.
        :param coeff: Coefficient affecting the PSF's shape or scale.
        :param estimated_axial_psf: A 1D array representing an estimated axial PSF.
        """
        self.radius = radius
        self.coeff = coeff
        self.scale_factor = scale_factor
        self.estimated_axial_psf = estimated_axial_psf if estimated_axial_psf is not None else None
        self.z_0 = np.argmax(self.estimated_axial_psf) if self.estimated_axial_psf is not None else z_0
        self.psf = None
    @staticmethod
    def W_z(w_0, z_0, z):
        return w_0 * np.sqrt(1 + (z / z_0)**2)
    
    def get_psf(self, z_depth):
        """
        Generate and return the PSF kernel based on the object's parameters.

        :return: A 2D numpy array representing the PSF kernel.
        """
        
        depth_z = abs(z_depth - self.z_0)
        w_z = self.radius * np.sqrt(1 + (depth_z / self.z_0)**2)
        # w_z = W_z(self.radius, self.z_0, depth_z)
        D = 2 * w_z
        r_coord = np.linspace(-self.coeff * D, self.coeff * D, 1 + self.scale_factor * 2 * np.round(int(self.coeff * D)))

        X, Y = np.meshgrid(r_coord, r_coord)
        r = np.sqrt(X**2 + Y**2)
        PSF_xy = np.exp(-2 * r / w_z)
        PSF_xy /= np.sum(PSF_xy)
        if self.estimated_axial_psf is not None:
            PSF_return = PSF_xy[len(r_coord) // 2,:]
            PSF_return = self.estimated_axial_psf[z_depth] * np.array(PSF_return / np.sum(PSF_return))
            PSF_return = np.reshape(PSF_return,(1,-1))
            self.psf = np.squeeze(PSF_return)
            return np.squeeze(PSF_return)
        else:
            self.psf = PSF_xy
            return PSF_xy
    
    def get_psf_mask(self,center_x, center_y, z_depth, input_image):
        """
        Generate and return the PSF kernel mask based on the object's parameters.

        :return: A 1D numpy array representing the PSF kernel mask.
        """
        
        PSF = self.get_psf(z_depth)
            
        if self.estimated_axial_psf is not None:
            delta_x = 0
            delta_y = PSF.shape[0] - math.ceil(PSF.shape[0] / 2)
    
            mask_psf_padded = np.zeros((input_image.shape[0] + 2 * delta_x, input_image.shape[1] + 2 * delta_y))
            scaled_xc = center_x * self.scale_factor 
            scaled_yc = center_y * self.scale_factor 
            mask_psf_padded[int(scaled_xc), 
                            int(scaled_yc + delta_y - delta_y):int(scaled_yc + delta_y + delta_y+1)] = PSF
                    
            mask_psf_i_2 = mask_psf_padded[delta_x:delta_x + input_image.shape[0], 
                                           delta_y:delta_y + input_image.shape[1]]
            # print("mask_psf_i_2", np.shape(mask_psf_i_2))
            sc = mask_psf_i_2.astype(float)
            # plt.imshow(sc, cmap='viridis', aspect='auto')
            # plt.colorbar()
            # plt.xlabel('Array Index')
            # plt.ylabel('Row Index')
            # plt.title('Concatenated Array')
            # plt.show()
            # cv2.imwrite('output_image.jpg', sc)
            # print(np.shape(PSF))
            # print(PSF)
            # print(np.shape(sc))
            sc = resize(sc, (sc.shape[0] // self.scale_factor, sc.shape[1] // self.scale_factor), anti_aliasing=True)
            
            

            return sc.flatten()
 
        else:
            delta_x = PSF.shape[0] - math.ceil(PSF.shape[0] / 2)
            delta_y = PSF.shape[1] - math.ceil(PSF.shape[1] / 2)
    
            mask_psf_padded = np.zeros((input_image.shape[0] + 2 * delta_x, input_image.shape[1] + 2 * delta_y))
            scaled_xc = center_x * self.scale_factor 
            scaled_yc = center_y * self.scale_factor 
            mask_psf_padded[int(scaled_xc + delta_x - delta_x):int(scaled_xc + delta_x + delta_x+1), 
                            int(scaled_yc + delta_y - delta_y):int(scaled_yc + delta_y + delta_y+1)] = PSF
                    
            mask_psf_i_2 = mask_psf_padded[delta_x:delta_x + input_image.shape[0], 
                                           delta_y:delta_y + input_image.shape[1]]
            # print("mask_psf_i_2", np.shape(mask_psf_i_2))
            sc = mask_psf_i_2.astype(float)
            sc = resize(sc, (sc.shape[0] // self.scale_factor, sc.shape[1] // self.scale_factor), anti_aliasing=True)
            return sc.flatten()
            
            