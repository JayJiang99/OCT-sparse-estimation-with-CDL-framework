import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftshift, ifftshift
from scipy.interpolate import interp2d
from skimage.transform import resize
from skimage import color, data, restoration
from scipy.fftpack import dct, idct
import math

import cvxpy as cp

class OCTReconstruction:
    def __init__(self, recon_size, radius, step_size, z0_plane):
        self.recon_size = recon_size
        self.radius = radius
        self.step = step_size
        self.z0 = z0_plane


    @staticmethod
    def W_z(w_0, z_0, z):
        return w_0 * np.sqrt(1 + (z / z_0)**2)

    def reconstruct_patch_psf(self, startPosX, startPosY, inputImg, z_depth):
        scale_factor = 10
        num_col_src = self.recon_size
        num_row_src = self.recon_size
        N_full_sample = num_col_src * num_row_src

        center_x, center_y = np.mgrid[1:num_col_src+1:self.step, 1:num_row_src+1:self.step]
        center_x = center_x.flatten()
        center_y = center_y.flatten()
        delete_index = []
        for i in range(len(center_y)):
            if center_y[i] < self.radius or center_x[i] < self.radius:
                delete_index.append(i)
            if (center_y[i] + self.radius > num_row_src) or (center_x[i] + self.radius > num_col_src):
                delete_index.append(i)

        center_x = np.delete(center_x, delete_index)
        center_y = np.delete(center_y, delete_index)
        M = len(center_x)

        num_col_sensing = int(np.sqrt(M))
        num_row_sensing = int(np.sqrt(M))

        partial_img = inputImg[startPosX:startPosX+num_col_sensing, startPosY:startPosY+num_row_sensing]
        print("num_row_sensing", num_row_sensing)
        print("partial image", np.shape(partial_img))
        I_init_estimate = resize(partial_img, (num_row_src, num_col_src))
        I_init_orig = partial_img
        
        PHI = np.zeros((M, N_full_sample))
        PHIPSI = np.zeros((M, N_full_sample))
        y = np.zeros(M)
        
        # PSF estimation
        w_z = self.W_z(self.radius, self.z0, z_depth)
        D = 2 * w_z
        coeff = 0.5
        x = np.linspace(-round(coeff * D), round(coeff * D),  1 + scale_factor * 2 * np.round(int(coeff * D)))
        X, Y = np.meshgrid(x, x)
        r = np.sqrt(X**2 + Y**2)
        PSF = np.exp(-2 * r / w_z)
        PSF /= np.sum(PSF)
        # Create a background image filled with zeros
        background_img = np.zeros((num_col_src, num_row_src))
        background_img = resize(background_img, (background_img.shape[0] * scale_factor, background_img.shape[1] * scale_factor), anti_aliasing=False)

        # Flip the partial image (equivalent to MATLAB's transpose)
        # TODO: if cannot reconstruct, delete .T
        partial_img_flip = partial_img.T
        # Flatten the flipped image to create a 1D array (equivalent to MATLAB's `(:)` operation)
        I_result = partial_img_flip.flatten()

        for i in range(M):
            y[i] = I_result[i] 
    
            delta_x = PSF.shape[0] - math.ceil(PSF.shape[0] / 2)
            delta_y = PSF.shape[1] - math.ceil(PSF.shape[1] / 2)
    
            mask_psf_padded = np.zeros((background_img.shape[0] + 2 * delta_x, background_img.shape[1] + 2 * delta_y))
            scaled_xc = center_x[i] * scale_factor
            scaled_yc = center_y[i] * scale_factor
            mask_psf_padded[int(scaled_xc + delta_x - delta_x):int(scaled_xc + delta_x + delta_x+1), 
                            int(scaled_yc + delta_y - delta_y):int(scaled_yc + delta_y + delta_y+1)] = PSF
                    
            mask_psf_i_2 = mask_psf_padded[delta_x:delta_x + background_img.shape[0], 
                                           delta_y:delta_y + background_img.shape[1]]
            print("mask_psf_i_2", np.shape(mask_psf_i_2))
            sc = mask_psf_i_2.astype(float)
            sc = resize(sc, (sc.shape[0] // scale_factor, sc.shape[1] // scale_factor), anti_aliasing=True)
    
            sc_flat = sc.flatten()
            PHI[i, :] = sc_flat
            # print("sc_flat", np.shape(sc_flat))
    
            sc_fwht = dct(sc_flat,norm='ortho')
            PHIPSI[i, :] = sc_fwht * N_full_sample

        # lambda_1 = 0.001
        # sol_c = cp.Variable(N_full_sample)
        # objective = cp.Minimize(lambda_1 *cp.norm(sol_c, 1) +  0.5*cp.norm(PHIPSI @ sol_c - y, 2))
        # prob = cp.Problem(objective)
        # # constraints = [PHIPSI @ sol_c == y]
        # # prob = cp.Problem(objective, constraints)
        # prob.solve()
        # Regularization parameter
        lambda_reg = 0.001
        miu = 0.1
        # Define the optimization variable
        sol_c = cp.Variable(N_full_sample)
        # Define the objective function
        objective = cp.Minimize(0.5 * cp.norm(PHIPSI @ sol_c - y, 2) + lambda_reg * cp.norm(sol_c, 1))
        # Define the constraints
        constraints = [cp.norm(sol_c - dct(I_init_estimate.flatten(), norm='ortho'), 2) <= miu]  # You need to define 'some_value' or adjust this constraint as needed
        # Define the problem and solve it
        prob = cp.Problem(objective)
        result = prob.solve()
        
        # Apply inverse sparse basis
        I_rec = idct(sol_c.value,norm='ortho').real
        I_rec_ini = np.zeros((num_col_src, num_row_src))
        I_rec_ini = np.reshape(I_rec,(num_col_src, num_row_src))
        I_rec_norm = np.clip(I_rec_ini, 0, None)
        I_rec_norm = (255 * (I_rec_norm - np.min(I_rec_norm)) / np.ptp(I_rec_norm)).astype(np.uint8)
    
        return I_rec_norm.T, I_init_orig  # peaksnr and ssimval calculation can be added here
