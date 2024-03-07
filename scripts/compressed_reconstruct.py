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


from psf_exp import PointSpreadFunction

class OCTReconstruction:
    def __init__(self, recon_size, radius, step_size, z_0, estimated_axial_psf = None):
        self.recon_size = recon_size
        self.radius = radius
        self.step = step_size
        
        self.scale_factor = 10
        self.estimated_axial_psf = estimated_axial_psf if estimated_axial_psf is not None else None
        self.z_0 = np.argmax(self.estimated_axial_psf) if self.estimated_axial_psf is not None else z_0
        self.psf = PointSpreadFunction(self.radius, self.z_0, 0.5, self.scale_factor, self.estimated_axial_psf)


    # @staticmethod
    # def W_z(w_0, z_0, z):
    #     return w_0 * np.sqrt(1 + (z / z_0)**2)

    def reconstruct_patch_psf(self, startPosX, startPosY, inputImg, z_depth):
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
        
        self.psf.get_psf(z_depth)
        
        # Create a background image filled with zeros
        background_img = np.zeros((num_col_src, num_row_src))
        background_img = resize(background_img, (background_img.shape[0] * self.scale_factor , background_img.shape[1] * self.scale_factor ), anti_aliasing=False)

        # Flip the partial image (equivalent to MATLAB's transpose)
        # TODO: if cannot reconstruct, delete .T
        partial_img_flip = partial_img.T
        # Flatten the flipped image to create a 1D array (equivalent to MATLAB's `(:)` operation)
        I_result = partial_img_flip.flatten()

        for i in range(M):
            y[i] = I_result[i] 
            sc_flat = self.psf.get_psf_mask(center_x[i], center_y[i], z_depth, background_img)
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
    
    def reconstruct_patch_psf_bscan(self, startPosX, startPosY, inputImg, w_weight = None):
        # Row number is fixed
        # The reconstructed target image is recon_size by recon_size
        # The sampled data is recon_size/2 by recon_size
        num_col_src = self.recon_size
        num_row_src = self.recon_size
        N_full_sample = num_col_src * num_row_src
        
        # center position should be recon_size/2 by recon_size
        center_x, center_y = np.mgrid[0:num_row_src:1, 0:num_col_src:self.step]
        center_x = center_x.flatten()
        center_y = center_y.flatten()
        delete_index = []
        # only cares about x because y axis does not have gaussian effect
        for i in range(len(center_y)):
            if center_y[i] < self.radius:
                delete_index.append(i)
            if (center_y[i] + self.radius > num_col_src):
                delete_index.append(i)
        center_x = np.delete(center_x, delete_index)
        center_y = np.delete(center_y, delete_index)
        
        
        num_row_sensing = num_row_src
        num_col_sensing = int(len(center_x)/num_row_sensing)
        
        M = num_row_sensing * num_col_sensing

        partial_img = inputImg[startPosX:startPosX+num_row_sensing, startPosY:startPosY+num_col_sensing]
        print("num_row_sensing", num_row_sensing)
        print("partial image", np.shape(partial_img))
        
        I_init_estimate = resize(partial_img, (num_row_src, num_col_src))
        
        I_init_orig = I_init_estimate
        
        PHI = np.zeros((M, N_full_sample))
        PHIPSI = np.zeros((M, N_full_sample))
        y = np.zeros(M)
        
        # Create a background image filled with zeros
        background_img = np.zeros((num_row_src, num_col_src))
        background_img = resize(background_img, (background_img.shape[0] * self.scale_factor , background_img.shape[1] * self.scale_factor ), anti_aliasing=False)

        # Flip the partial image (equivalent to MATLAB's transpose)
        # TODO: if cannot reconstruct, delete .T
        partial_img_flip = partial_img
        # Flatten the flipped image to create a 1D array (equivalent to MATLAB's `(:)` operation)
        I_result = partial_img_flip.flatten()

        for i in range(M):
            y[i] = I_result[i] 
            sc_flat = self.psf.get_psf_mask(center_x[i], center_y[i], center_x[i]+startPosX, background_img)
            PHI[i, :] = sc_flat
            # print("sc_flat", np.shape(sc_flat))
    
            sc_fwht = dct(sc_flat,norm='ortho')
            PHIPSI[i, :] = sc_fwht * N_full_sample
        if w_weight is not None:
            local_weight = w_weight[startPosX:startPosX+num_row_sensing, startPosY:startPosY+num_col_sensing]
            local_weight = resize(local_weight, (num_row_src, num_col_src))
            local_weight_coeff = local_weight.flatten()
            
            lambda_reg = 0.002
            miu = 0.1
            # Define the optimization variable
            sol_c = cp.Variable(N_full_sample)
            
            # Define the objective function
            objective = cp.Minimize(0.5 * cp.norm(PHIPSI @ sol_c - y, 2) + lambda_reg * cp.norm(cp.multiply(local_weight_coeff, sol_c), 1))
            # Define the constraints
            constraints = [cp.norm(sol_c - dct(I_init_estimate.flatten(), norm='ortho'), 2) <= miu]  # You need to define 'some_value' or adjust this constraint as needed
            # Define the problem and solve it
            prob = cp.Problem(objective)
            result = prob.solve()
   
        else:
            lambda_1 = 0.002
            sol_c = cp.Variable(N_full_sample)
            objective = cp.Minimize(lambda_1 *cp.norm(sol_c, 1) +  0.5*cp.norm(sol_c - dct(I_init_estimate.flatten(), norm='ortho'), 2))
            prob = cp.Problem(objective)
            constraints = [PHIPSI @ sol_c == y]
            prob = cp.Problem(objective, constraints)
            prob.solve()
            # # Regularization parameter
            # lambda_reg = 0.001
            # miu = 0.1
            # # Define the optimization variable
            # sol_c = cp.Variable(N_full_sample)
            # # Define the objective function
            # objective = cp.Minimize(0.5 * cp.norm(PHIPSI @ sol_c - y, 2) + lambda_reg * cp.norm(sol_c, 1))
            # # Define the constraints
            # constraints = [cp.norm(sol_c - dct(I_init_estimate.flatten(), norm='ortho'), 2) <= miu]  # You need to define 'some_value' or adjust this constraint as needed
            # # Define the problem and solve it
            # prob = cp.Problem(objective)
            # result = prob.solve()
            
        # Apply inverse sparse basis
        I_rec = idct(sol_c.value,norm='ortho').real
        I_rec_ini = np.zeros((num_row_src, num_col_src))
        I_rec_ini = np.reshape(I_rec,(num_row_src, num_col_src))
        I_rec_norm = np.clip(I_rec_ini, 0, None)
        I_rec_norm = (255 * (I_rec_norm - np.min(I_rec_norm)) / np.ptp(I_rec_norm)).astype(np.uint8)
    
        return I_rec_norm, I_init_orig  # peaksnr and ssimval calculation can be added here
