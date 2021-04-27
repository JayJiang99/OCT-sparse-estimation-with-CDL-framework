# -*- coding: utf-8 -*-
# @Time    : 2021-04-26 8:48 p.m.
# @Author  : young wang
# @FileName: omega_compare.py
# @Software: PyCharm


'''this script generates images for the figure 3.1 as seen in
the paper. Sparse reconstructions of the same OCT
middle ear image using the same learned dictionary for
various values of the weighting parameter omega'''
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sporco.admm import cbpdn
from skimage.morphology import disk
from skimage.morphology import dilation, erosion
from misc import processing,quality,annotation
from functools import partial

# Module level constants
eps = 1e-14

def getWeight(lmbda,speckle_weight):
    b = cbpdn.ConvBPDN(D, snorm, lmbda, opt=opt_par, dimK=1, dimN=1)
    # Calculate the sparse vector and an an epsilon to keep the log finite
    xnorm = b.solve().squeeze() + eps
    # Caclulate sparse reconstruction
    xnorm = np.roll(xnorm, np.argmax(D), axis=0)

    # Convert back from normalized
    x = processing.from_l2_normed(xnorm, l2f)

    x_log = 20 * np.log10(abs(x))
    x_log = processing.imag2uint(x_log,rvmin,vmax)

    #set thresdhold
    x_log = np.where(x_log <= rvmin,0,x_log)

    W = dilation(x_log,  disk(5))

    # W = erosion(W,  disk(5))
    
    W = np.where(W > 0, speckle_weight,1)
    W = np.reshape(W, (W.shape[0], 1, -1, 1))

    return W

if __name__ == '__main__':

    plt.close('all')
    # Customize matplotlib params
    matplotlib.rcParams.update(
        {
            'font.size': 18,
            'text.usetex': False,
            'font.family': 'stixgeneral',
            'mathtext.fontset': 'stix',
        }
    )
    file_name = ['ear']
    # Load the example dataset
    s, D = processing.load_data(file_name[0], decimation_factor=20)

    rvmin = 65  # dB
    vmax = 115  # dB

    s_log = 20 * np.log10(abs(s))
    s_log = processing.imag2uint(s_log, rvmin, vmax)

    # l2 norm data and save the scaling factor
    l2f, snorm = processing.to_l2_normed(s)

    opt_par = cbpdn.ConvBPDN.Options({'FastSolve': True, 'Verbose': False, 'StatusHeader': False,
                                      'MaxMainIter': 20, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                      'RelaxParam': 1.515, 'AutoRho': {'Enabled': True}})

    # Weigth factor to apply to the fidelity (l2) term in the cost function
    # in regions segmented as containing speckle
    speckle_weight = np.linspace(1e-1,1,5)
    lmbda = 0.05

    update_weight = partial(getWeight,0.1)

    index = 400 # index A-line
    s_line = abs(snorm[:,index])

    x_line = np.zeros((snorm.shape[0], len(speckle_weight)))
    sparse = np.zeros((snorm.shape[0], snorm.shape[1], len(speckle_weight)))
    sparsity = np.zeros(len(speckle_weight))

    #update opt to include W

    for i in range(len(speckle_weight)):

        W = np.roll(update_weight(speckle_weight[i]),  np.argmax(D), axis=0)
        opt_par = cbpdn.ConvBPDN.Options({'FastSolve': True, 'Verbose': False, 'StatusHeader': False,
                                          'MaxMainIter': 200, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                          'RelaxParam': 1.515, 'L1Weight': W, 'AutoRho': {'Enabled': True}})

        b = cbpdn.ConvBPDN(D, snorm, lmbda, opt=opt_par, dimK=1, dimN=1)
        xnorm = b.solve().squeeze()

        #calculate sparsity
        sparsity[i] = (1-np.count_nonzero(xnorm) / xnorm.size)
        xnorm += eps

        xnorm = np.roll(xnorm, np.argmax(D), axis=0)

        x_line[:,i] = abs(xnorm[:,index])
        ## Convert back from normalized
        x = processing.from_l2_normed(xnorm, l2f)

        x_log = 20 * np.log10(abs(x))
        x_log = processing.imag2uint(x_log, rvmin, vmax)
        sparse[:,:,i] = x_log

    width, height = (100, 80)
    homogeneous = [[125, 120, width, height]]
    artifact = [[75, 5, 10, 8]]
    background = [[425, 250, width - 25, height - 20]]

    vmax, vmin = 255,0
    fig = plt.figure(constrained_layout=True, figsize=(16, 9))
    gs = fig.add_gridspec(ncols=len(speckle_weight) + 1, nrows=3)

    aspect = s_log.shape[1] / s_log.shape[0]
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(s_log, 'gray', aspect=aspect, vmax=vmax, vmin=vmin)
    ax.set_axis_off()
    ax.set_title('reference', fontname='Arial')
    ax.axvline(x=index, ymin=0.6, ymax=1, linewidth=1, color='orange', linestyle='--')
    ax.axvline(x=index, ymin=0, ymax=0.6, linewidth=1, color='orange')
    for k in range(len(homogeneous)):
        for j in annotation.get_homogeneous(*homogeneous[k]):
            ax.add_patch(j)
    for k in range(len(background)):
        for j in annotation.get_background(*background[k]):
            ax.add_patch(j)

    ho_original = quality.ROI(*homogeneous[0], s_log)

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(ho_original, 'gray', aspect=ho_original.shape[1] / ho_original.shape[0], vmax=vmax, vmin=vmin)
    ax.set_axis_off()
    ax.annotate('', xy=(72.5, 10), xycoords='data',
                xytext=(60, 5), textcoords='data',
                arrowprops=dict(facecolor='white', shrink=0.05),
                horizontalalignment='right', verticalalignment='top',
                )
    ax.annotate('', xy=(87.5, 55), xycoords='data',
                xytext=(92.5, 70), textcoords='data',
                arrowprops=dict(facecolor='red', shrink=0.05),
                horizontalalignment='right', verticalalignment='top',
                )
    for k in range(len(artifact)):
        for j in annotation.get_artifact(*artifact[k]):
            ax.add_patch(j)

    ax = fig.add_subplot(gs[2, 0])
    ax.plot(s_line)
    ax.set_xlabel('axial depth [pixels]', fontname='Arial')
    ax.set_ylabel('normalized intensity [a.u.]', fontname='Arial')
    ax.set_ylim(0, np.max(s_line)*1.1)

    ho_s = quality.ROI(*homogeneous[0], s_log)
    ba_s = quality.ROI(*homogeneous[0], s_log)
    ar_s = quality.ROI(*homogeneous[0], s_log)

    textstr = '\n'.join((
        '$\mathregular{SF_{S}}$''\n'
        r'%.2f' % (quality.SF(s_log),),
        '$\mathregular{SF_{B}}$''\n'
        r'%.2f' % (quality.SF(ba_s),),
        '$\mathregular{SF_{A}}$''\n'
        r'%.2f' % (quality.SF(ar_s),),
    ))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
    ax.text(0.05, 0.92, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props, fontname='Arial')

    for i in range(len(speckle_weight)):
        temp = sparse[:, :, i]
        aspect = temp.shape[1]/temp.shape[0]
        ax = fig.add_subplot(gs[0, i + 1])
        ax.imshow(temp, 'gray', aspect=aspect, vmax=vmax, vmin=vmin)
        ax.axvline(x=index, ymin=0.6, ymax=1, linewidth=1, color='orange', linestyle='--')
        ax.axvline(x=index, ymin=0, ymax=0.6, linewidth=1, color='orange')
        for k in range(len(background)):
            for j in annotation.get_background(*background[k]):
                ax.add_patch(j)

        ax.set_title('𝜆 = %.2f \n $\omega$ = %.1f' % (lmbda, speckle_weight[i]))
        ax.set_axis_off()
        for k in range(len(homogeneous)):
            for j in annotation.get_homogeneous(*homogeneous[k]):
                ax.add_patch(j)

        ho_x = quality.ROI(*homogeneous[0], temp)
        ba_x = quality.ROI(*background[0], temp)
        ar_x = quality.ROI(*artifact[0], temp)

        aspect = width / height
        ax = fig.add_subplot(gs[1, i + 1])
        ax.imshow(ho_x, 'gray', aspect=aspect, vmax=vmax, vmin=vmin)
        ax.annotate('', xy=(72.5, 10), xycoords='data',
                    xytext=(60, 5), textcoords='data',
                    arrowprops=dict(facecolor='white', shrink=0.05),
                    horizontalalignment='right', verticalalignment='top',
                    )
        ax.annotate('', xy=(87.5, 55), xycoords='data',
                    xytext=(92.5, 70), textcoords='data',
                    arrowprops=dict(facecolor='red', shrink=0.05),
                    horizontalalignment='right', verticalalignment='top',
                    )
        for k in range(len(artifact)):
            for j in annotation.get_artifact(*artifact[k]):
                ax.add_patch(j)

        ax.set_axis_off()

        ax = fig.add_subplot(gs[2, i + 1])
        ax.plot(x_line[:, i])
        ax.set_yticks([])
        ax.set_ylim(0, np.max(s_line)*1.1)

        textstr = '\n'.join((
            '$\mathregular{SF_{S}}$''\n'
            r'%.2f' % (quality.SF(temp),),
            '$\mathregular{SF_{B}}$''\n'
            r'%.2f' % (quality.SF(ba_x),),
            '$\mathregular{SF_{A}}$''\n'
            r'%.2f' % (quality.SF(ar_x),),
        ))

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
        ax.text(0.05, 0.92, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props, fontname='Arial')

        ax.set_xlabel('axial depth [pixels]', fontname='Arial')
    plt.show()

