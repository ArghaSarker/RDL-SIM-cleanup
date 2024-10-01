# %% [markdown]
# ## In this Notebook, we train the Super Resoulation Module. 
# - the training progress can be viewed W&B in tensorboard section. 

# %%




import datetime
import os
from csbdeep.io import load_training_data
from csbdeep.utils import axes_dict, plot_some,plot_history
import matplotlib.pyplot as plt
from models import DFCAN
from loss_functions import mse_ssim, mse_ssim_psnr
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
from tensorflow.keras.models import load_model
from wandb.integration.keras import WandbMetricsLogger, WandbEvalCallback
from tensorflow.keras.callbacks import TensorBoard
from skimage.metrics import peak_signal_noise_ratio as psnr, mean_squared_error as mse, structural_similarity as ssim
import numpy as np
import wandb
from tensorflow.image import psnr, ssim
from skimage.metrics import mean_squared_error
from csbdeep.data.generate import norm_percentiles
from csbdeep.utils import normalize
    
print(tf.__version__)


# %%
# set up the train locations and logs.
# wandb.login()
root_dir = '../Microtubules'
model_dir = Path(root_dir)/'Microtubules_SR_model'
Path(model_dir).mkdir(exist_ok=True)
train_data_file = f'{root_dir}/Train/SR/augmented_Microtubules_02_SR.npz'
val_data_file = f'{root_dir}/val/Train/SR/augmented_Microtubules_02_SR.npz'
log_dir = "logs/fitSR/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

############## for saving the results of the training ################
output_dir = Path.cwd() / 'Microtubules_02_SR_results'
Path(output_dir).mkdir( exist_ok=True)

# for saving the training progress and viewing it in tensorboard
# wandb.init(project="Microtubules_SR",name=f"Microtubules_02_SR",config= tf.compat.v1.flags.FLAGS, sync_tensorboard=True)

# tensorboard_callback = TensorBoard(log_dir=wandb.run.dir)
tf.function(jit_compile=True)


# %% [markdown]
# ### load the training data and define the model and parameters

# %%
strategy = tf.distribute.MirroredStrategy()

with strategy.scope(): 
    ################  define the train  parameters  ################

    init_lr = 1e-4
    print(f'strategy.num_replicas_in_sync : {strategy.num_replicas_in_sync}')
    epochs = 2
    # batch_size =16 * strategy.num_replicas_in_sync
    batch_size = 4
 
    beta_1=0.9
    beta_2=0.999
    scale_gt = 2.0
    lr_decay_factor = 0.67	# Learning rate decay factor	


    print(f'this are the training setup: \nlearning_rate : {init_lr} \nepochs : {epochs} \nbatch_size : {batch_size} \nscale_gt : {scale_gt} \nlr_decay_factor : {lr_decay_factor}')

    (X,Y), _ , axes = load_training_data(train_data_file, verbose=True)
    (X_val,Y_val), _ , axes = load_training_data(val_data_file, verbose=True)
    print()
    print()
    print('Information about SR training data')
    print(f"X_shape :  {X.shape} ,\nX_dtype : {X.dtype}   Y_shape: {Y.shape}\nY_dtype : {Y.dtype}   ,\nX_val : {X_val.shape} ,\nY_val : {Y_val.shape}")
    print()
    # ###########################################
    # X = X[:50]
    # Y = Y[:50]
    # X_val = X_val[:10]
    # Y_val = Y_val[:10]


    ############### preprocess the data to fit into the model ################
    def preprocess_data(X, Y):
        # Squeeze the unnecessary dimensions and transpose the axes
        X = tf.squeeze(X, axis=-1)
        Y = tf.squeeze(Y, axis=-1)
        X = tf.transpose(X, perm=[0, 2, 3, 1])
        Y = tf.transpose(Y, perm=[0, 2, 3, 1])
        return X, Y

    X, Y = preprocess_data(X, Y)
    X_val, Y_val = preprocess_data(X_val, Y_val)

    train_dataset = (X, Y)
    val_dataset = (X_val, Y_val)

    print(f'after preprocessing the data in batch chunks. X : {X.shape} Y : {Y.shape} X_val : {X_val.shape}  Y_val : {Y_val.shape}')

    ################  plot some of the training data  ################
    plt.figure(figsize=(25,15))
    # Randomly pick 6 images
    indices = np.random.choice(X.shape[0], 6, replace=False)
    fig, axes = plt.subplots(2, 6, figsize=(25, 15))
    for i, idx in enumerate(indices):
        axes[0, i].imshow(X[idx,...,1], cmap='viridis')
        # print(X[idx].shape)
        axes[0, i].axis('off')
        axes[0, i].set_title(f'IN : shape: {X[idx].shape} \ndtype: {X.dtype}\nmin: {np.min(X[idx, :, :, 0])}\nmax: {np.max(X[idx, :, :, 0])}', pad=10, fontsize=16)
        
        axes[1, i].imshow(Y[idx], cmap='viridis')
        # print(Y[idx].shape)
        axes[1, i].axis('off')
        axes[1, i].set_title(f'GT : shape: {Y[idx].shape} \ndtype: {Y.dtype}\nmin: {np.min(Y[idx, :, :, 0])}\nmax: {np.max(Y[idx, :, :, 0])}', pad=10, fontsize=16)
        
    plt.suptitle('6 example training patches \n', fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/SR_train_image_Mictotubules_02.tiff', bbox_inches='tight', format='tiff')
    plt.show()
    plt.close()

    # dp same fpr x_val and y_val
    plt.figure(figsize=(25,15))
    # Randomly pick 6 images
    indices = np.random.choice(X_val.shape[0], 6, replace=False)
    fig, axes = plt.subplots(2, 6, figsize=(25, 15))
    for i, idx in enumerate(indices):
        axes[0, i].imshow(X_val[idx,...,1], cmap='viridis')
        # print(X_val[idx].shape)
        axes[0, i].axis('off')
        axes[0, i].set_title(f'IN : shape: {X_val[idx].shape} \ndtype: {X_val.dtype}\nmin: {np.min(X_val[idx, :, :, 0])}\nmax: {np.max(X_val[idx, :, :, 0])}', pad=10, fontsize=16)
        
        axes[1, i].imshow(Y_val[idx], cmap='viridis')
        # print(Y_val[idx].shape)
        axes[1, i].axis('off')
        axes[1, i].set_title(f'GT : shape: {Y_val[idx].shape} \ndtype: {Y_val.dtype}\nmin: {np.min(Y_val[idx, :, :, 0])}\nmax: {np.max(Y_val[idx, :, :, 0])}', pad=10, fontsize=16)

    plt.suptitle('6 example validation patches \n', fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/SR_VAL_image_Mictotubules_02.tiff', bbox_inches='tight', format='tiff')
    plt.show()
    plt.close()










    total_data,  height, width, channels= X.shape
    print(f'total_data,  height, width, channels : {total_data,  height, width, channels}')
    valid_data = val_dataset

    Trainingmodel = DFCAN((height, width, channels), scale=scale_gt)
    optimizer = Adam(learning_rate=init_lr, beta_1=beta_1, beta_2=beta_2)
    Trainingmodel.compile(loss=mse_ssim, optimizer=optimizer)
    #Trainingmodel.summary()
    
    tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # lrate = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1) # monitor val_loss for faster training
    
    lrate= callbacks.ReduceLROnPlateau(monitor='loss', factor=lr_decay_factor, 
                            patience=15, mode='auto', min_delta=1e-4,
                            cooldown=0, min_lr=  init_lr *0.1, verbose=1)

    hrate = callbacks.History()
    
    srate = callbacks.ModelCheckpoint(
                str(model_dir),
                monitor="loss",
                save_best_only=True,
                save_weights_only=False,                
                mode="auto",
            )

    ################  load the model if it exists  ################
    if len(os.listdir(model_dir)) > 0:
    
      with tf.keras.utils.custom_object_scope({'mse_ssim': mse_ssim}):
        if len(os.listdir(model_dir)) > 0:
            print(f'Loading model from {model_dir}')
            Trainingmodel = load_model(model_dir)
    else:
        # If no pre-trained model is found, define a new model
        Trainingmodel = DFCAN((height, width, channels), scale=scale_gt)
        optimizer = Adam(learning_rate=init_lr, beta_1=beta_1, beta_2=beta_2)
        Trainingmodel.compile(loss=mse_ssim, optimizer=optimizer)

   
    history = Trainingmodel.fit(X,Y, batch_size=batch_size,
                                   epochs=epochs, validation_data=val_dataset, shuffle=True,
                                   callbacks=[lrate, hrate, srate, tensorboard_callback ])
    
    
    Trainingmodel.save(model_dir)
        
    
    print(f'hisitry :: {history}')
    print(sorted(list(history.history.keys())))
    plt.figure(figsize=(16,5))
    plot_history(history,['loss','val_loss'])
    plt.figure(figsize=(12,7))
    plt.savefig(f'{output_dir}/SR_train_image_F-actin_04_history.png', bbox_inches='tight')


# %% [markdown]
# ### Prediction from the model
# %%
# Randomly pick 5 images for prediction
# Convert tensors to numpy arrays
X_val_sample = X_val.numpy()
Y_val_sample = Y_val.numpy()

# Print the shape, dtype, and class of the arrays
print(f'X_val_sample : {X_val_sample.shape} dtype : {X_val_sample.dtype} class : {type(X_val_sample)}')
print(f'Y_val_sample : {Y_val_sample.shape} dtype : {Y_val_sample.dtype} class : {type(Y_val_sample)}')

# Randomly select 5 indices and convert to tf.int32
indices = np.random.choice(X_val.shape[0], 5, replace=False)
indices = tf.convert_to_tensor(indices, dtype=tf.int32)  # Convert to TensorFlow tensor

# Select the subset of 5 images using TensorFlow indexing
X_val_sample = tf.gather(X_val, indices)
Y_val_sample = tf.gather(Y_val, indices)

# Convert tensors to numpy arrays for compatibility with skimage metrics
X_val_sample = X_val_sample.numpy()
Y_val_sample = Y_val_sample.numpy()

# Print the shape of the new subset
print(f'Selected X_val_sample : {X_val_sample.shape} dtype : {X_val_sample.dtype} class : {type(X_val_sample)}')
print(f'Selected Y_val_sample : {Y_val_sample.shape} dtype : {Y_val_sample.dtype} class : {type(Y_val_sample)}')
_P = Trainingmodel.predict(X_val_sample)
print(f'prediction : {_P.shape} dtype : {_P.dtype} class : {type(_P)}')

# Normalize the predictions using norm_percentiles

_P_normalized = normalize(_P)
# Ensure the normalized prediction has the same shape as _P
# _P_normalized = _P_normalized[:, 0, ...]

print(f'normalized prediction : {_P_normalized.shape} dtype : {_P_normalized.dtype} class : {type(_P_normalized)} \nmin: {np.min(_P_normalized)} \nmax: {np.max(_P_normalized)}')
# make the _p and _p_normalized the same
_P = _P_normalized
# Calculate the difference between ground truth and predicted images
difference = np.abs(Y_val_sample - _P)
print(f' seeing if the shapes are okay: {Y_val_sample.shape} {_P.shape} difference : {difference.shape}')
print(f' seeing Y_val_sample : {Y_val_sample.dtype}   {type(Y_val_sample)}')
print(f' seeing _P : {_P.dtype}   {type(_P)}')
print(f' seeing difference : {difference.dtype}   {type(difference)}')

# Calculate PSNR, MSE, NRMSE, and SSIM for the predictions
psnr_values = [psnr(Y_val_sample[i], _P[i], max_val=1.0).numpy() for i in range(5)]
mse_values = [mse(Y_val_sample[i], _P[i]) for i in range(5)]
nrmse_values = [np.sqrt(mse(Y_val_sample[i], _P[i])) / (np.max(Y_val_sample[i]) - np.min(Y_val_sample[i])) for i in range(5)]
ssim_values = [ssim(Y_val_sample[i], _P[i], max_val=1.0).numpy() for i in range(5)]

# Plot the images
fig, axes = plt.subplots(5, 5, figsize=(30, 30))

for i in range(5):
    axes[0, i].imshow(X_val_sample[i, ..., 1], cmap='viridis')
    axes[0, i].axis('off')
    axes[0, i].set_title(f'Input: shape: {X_val_sample[i].shape} \ndtype : {X_val_sample.dtype} \nmin: {np.min(X_val_sample[i, :, :, 0])}\nmax: {np.max(X_val_sample[i, :, :, 0])}', pad=5, fontsize=14)

    axes[1, i].imshow(Y_val_sample[i], cmap='viridis')
    axes[1, i].axis('off')
    axes[1, i].set_title(f'GT: shape: {Y_val_sample[i].shape} \ndtype : {Y_val_sample.dtype} \nmin: {np.min(Y_val_sample[i, :, :, 0])}\nmax: {np.max(Y_val_sample[i, :, :, 0])}', pad=5, fontsize=14)

    axes[2, i].imshow(_P[i], cmap='viridis')
    axes[2, i].axis('off')
    axes[2, i].set_title(f'Prediction: shape: {_P[i].shape} \ndtype : {_P.dtype} \nmin: {np.min(_P[i, :, :, 0])}\nmax: {np.max(_P[i, :, :, 0])}', pad=5, fontsize=14)

    axes[3, i].imshow(difference[i], cmap='inferno')
    axes[3, i].axis('off')
    axes[3, i].set_title(f'Difference: shape: {difference[i].shape} \nmin: {np.min(difference[i, :, :, 0])}\nmax: {np.max(difference[i, :, :, 0])}', pad=5, fontsize=14)

    axes[4, i].text(0.5, 0.5, f'PSNR: {psnr_values[i]}\nMSE: {mse_values[i]}\nNRMSE: {nrmse_values[i]}\nSSIM: {ssim_values[i]}', 
                    horizontalalignment='center', verticalalignment='center', fontsize=25, transform=axes[4, i].transAxes)
    axes[4, i].axis('off')

plt.suptitle('Comparison of input, GT, prediction, difference images, and metrics', fontsize=12)
plt.tight_layout()
plt.savefig(f'{output_dir}/SR_train_image_Mictotubules_02_prediction.tiff', bbox_inches='tight', format='tiff')
plt.show()
plt.close()
