import tensorflow as tf
from tensorflow.keras import backend as K
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np



# --------------------------------------------------------------------------------
#                              define loss
# --------------------------------------------------------------------------------
def mse_ssim(y_true, y_pred):

    ssim_para = 1e-1  # 1e-2
    mse_para = 1

    # normalization
    x = y_true
    y = y_pred
    x = (x - K.min(x)) / (K.max(x) - K.min(x))
    y = (y - K.min(y)) / (K.max(y) - K.min(y))

    # calculate loss
    ssim = ssim_para * (1 - K.mean(tf.image.ssim(x, y, 1)))
    mse = mse_para * K.mean(K.square(y - x))
    

    return mse + ssim
def mse_ssim_psnr(y_true, y_pred, data_range=None):

    ssim_para = 0.3  # 1e-2
    mse_para = 1
    psnr_para = 0.1

    # normalization
    x = y_true
    y = y_pred
    # print(f'this is y_true: { x.shape} \nmin : {np.min( x)} \nmax : {np.max( x)}')
    # print(f'this is y_true: { x.shape} \nmin : {np.min(x.numpy())} \nmax : {np.max(x.numpy())} \ndtype : {x.dtype}')
    # print('\n\n')

    # # print(f'this is Y_pred: { y.shape} \nmin : {np.min( y)} \nmax : {np.max( y)}')
    # print(f'this is Y_pred: { y.shape} \nmin : {K.min( y)} \nmax : {K.max( y)} \ndtype : {y.dtype}')

    x = (x - K.min(x)) / (K.max(x) - K.min(x))
    y = (y - K.min(y)) / (K.max(y) - K.min(y))

    # calculate loss
    ssim = ssim_para * (1 - K.mean(tf.image.ssim(x, y, 1)))
    mse = mse_para * K.mean(K.square(y - x))
    psnr_ = psnr_para *  tf.image.psnr(x,y, max_val=1.0)
    

    return mse + ssim + psnr_


def mse_ssim_3d(y_true, y_pred):
    ssim_para = 1e-1  # 1e-2
    mse_para = 1

    # normalization
    x = K.permute_dimensions(y_true, (0, 4, 1, 2, 3))
    y = K.permute_dimensions(y_pred, (0, 4, 1, 2, 3))
    x = (x - K.min(x)) / (K.max(x) - K.min(x))
    y = (y - K.min(y)) / (K.max(y) - K.min(y))

    # calculate loss
    ssim_loss = ssim_para * (1 - K.mean(tf.image.ssim(x, y, 1)))
    mse_loss = mse_para * K.mean(K.square(y - x))

    return mse_loss + ssim_loss


def mae_ssim(y_true, y_pred):

    ssim_para = 1e-1 # 1e-2
    mse_para = 1

    # normalization
    x = y_true
    y = y_pred
    x = (x - K.min(x)) / (K.max(x) - K.min(x))
    y = (y - K.min(y)) / (K.max(y) - K.min(y))

    # calculate loss
    ssim = ssim_para * (1 - K.mean(tf.image.ssim(x, y, 1)))
    mae = mse_para * K.abs(y - x)

    return mae + ssim


def mse_gar(y_true, y_pred):
    gamma = 1

    # mse loss
    y_pred = y_pred[..., 0]
    y_gt = y_true[..., 0]
    mse_loss = K.mean(K.square(y_gt - y_pred))

    # gar regularization
    y_gar = y_true[..., 1]
    neighbor_loss = K.mean(K.square(y_gt - y_pred - y_gar))

    return mse_loss + gamma * neighbor_loss