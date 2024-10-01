import tensorflow as tf
import numpy as np
import numpy.fft as F
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import constant as const
from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    Conv2D,
    Conv3D,
    Input,
    Lambda,
    Layer,
    LeakyReLU,
    UpSampling2D,
    add,
    concatenate,
    multiply,
)
from tensorflow.keras.models import Model
from sim_fitting import cal_modamp, create_psf, get_otf
import cv2 
import matplotlib.pyplot as plt
from tqdm import tqdm
from csbdeep.utils import normalize
from tensorflow.keras import callbacks
from dn_data_prep import PPE_branch, MPE_branch, GT_branch


class NoiseSuppressionModule(Layer):

    def __init__(self, init_cutoff_freq=4.1, dxy=0.0926, init_slop=100):
        super().__init__()
        self.cutoff_freq = self.add_weight(
            shape=(1,),
            initializer=const(init_cutoff_freq),
            trainable=True,
            name="cutoff_freq",
        )
        self.slop = self.add_weight(
            shape=(1,),
            initializer=const(init_slop),
            trainable=True,
            name="slop",
        )
        self.dxy = tf.Variable(initial_value=dxy, trainable=False, name="dxy")

    def call(self, inputs):
        bs, ny, nx, nz, ch = inputs.get_shape().as_list()
        ny = tf.cast(ny, tf.float32)
        nx = tf.cast(nx, tf.float32)
        dkx = tf.divide(1, tf.multiply(nx, self.dxy))
        dky = tf.divide(1, tf.multiply(ny, self.dxy))

        y = tf.multiply(tf.cast(tf.range(-ny // 2, ny // 2), tf.float32), dky)
        x = tf.multiply(tf.cast(tf.range(-nx // 2, nx // 2), tf.float32), dkx)
        [X, Y] = tf.meshgrid(x, y)
        rdist = tf.sqrt(tf.square(X) + tf.square(Y))

        otf_mask = tf.sigmoid(tf.multiply(self.cutoff_freq - rdist, self.slop))
        otf_mask = tf.expand_dims(
            tf.expand_dims(tf.expand_dims(otf_mask, 0), 0), 0
        )
        otf_mask = tf.tile(otf_mask, (1, nz, ch, 1, 1))
        otf_mask = tf.complex(otf_mask, tf.zeros_like(otf_mask))

        inputs = tf.complex(inputs, tf.zeros_like(inputs))
        inputs = tf.transpose(inputs, [0, 3, 4, 1, 2])
        fft_feature = tf.signal.fftshift(tf.signal.fft2d(inputs))
        output = tf.signal.ifft2d(
            tf.signal.fftshift(tf.multiply(otf_mask, fft_feature))
        )
        output = tf.transpose(output, [0, 3, 4, 1, 2])
        output = tf.math.real(output)

        return output


def GlobalAveragePooling(input):
    return tf.reduce_mean(input, axis=(1, 2, 3), keepdims=True)


def CALayer(input, channel, reduction=16):
    W = Lambda(GlobalAveragePooling)(input)
    W = Conv3D(
        channel // reduction, kernel_size=1, activation="relu", padding="same"
    )(W)
    W = Conv3D(channel, kernel_size=1, activation="sigmoid", padding="same")(W)
    mul = multiply([input, W])
    return mul


def RCAB3D(input, channel):
    conv = Conv3D(channel, kernel_size=3, padding="same")(input)
    conv = LeakyReLU(alpha=0.2)(conv)
    conv = Conv3D(channel, kernel_size=3, padding="same")(conv)
    conv = LeakyReLU(alpha=0.2)(conv)
    att = CALayer(conv, channel, reduction=16)
    output = add([att, input])
    return output


def ResidualGroup(input, channel, n_RCAB=5):
    conv = input
    for _ in range(n_RCAB):
        conv = RCAB3D(conv, channel)
    return conv


def RCAN3D(input_shape, n_ResGroup=4, n_RCAB=5):

    inputs = Input(input_shape)
    conv = Conv3D(64, kernel_size=3, padding="same")(inputs)
    for _ in range(n_ResGroup):
        conv = ResidualGroup(conv, 64, n_RCAB=n_RCAB)

    conv = Conv3D(256, kernel_size=3, padding="same")(conv)
    conv = LeakyReLU(alpha=0.2)(conv)
    conv = Conv3D(input_shape[3], kernel_size=3, padding="same")(conv)
    output = LeakyReLU(alpha=0.2)(conv)

    model = Model(inputs=inputs, outputs=output)

    return model


def RCANNSM3D(input_shape, n_ResGroup=4, n_RCAB=5):
    inputs = Input(input_shape)
    conv_input = Conv3D(64, kernel_size=3, padding="same")(inputs)
    NSM = NoiseSuppressionModule()
    inputs_ns = NSM(inputs)
    conv = Conv3D(64, kernel_size=3, padding="same")(inputs_ns)
    conv = conv + conv_input
    for _ in range(n_ResGroup):
        conv = ResidualGroup(conv, 64, n_RCAB=n_RCAB)

    conv = Conv3D(256, kernel_size=3, padding="same")(conv)
    conv = LeakyReLU(alpha=0.2)(conv)
    conv = Conv3D(input_shape[3], kernel_size=3, padding="same")(conv)
    output = LeakyReLU(alpha=0.2)(conv)

    model = Model(inputs=inputs, outputs=output)

    return model


def global_average_pooling(input):
    return tf.reduce_mean(input, axis=(1, 2), keepdims=True)


def FCALayer(input, channel, reduction=16):
    absfft1 = Lambda(fft2)(input)
    absfft1 = Lambda(fftshift)(absfft1)
    ###############   why this 2 lines are there ?? ################
    absfft1 = tf.abs(absfft1, name="absfft1") 
    absfft1 = tf.cast(absfft1, dtype=tf.float32)
    # the fftshit are are  not resized as the original code: it throughs lambda error. 
    # output = tf.image.resize(output, (128, 128)) # this line is not there into the ffshift output 

    ###################################################################
    absfft2 = Conv2D(
        channel, kernel_size=3, activation="relu", padding="same"
    )(absfft1)
    W = Lambda(global_average_pooling)(absfft2)
    W = Conv2D(
        channel // reduction, kernel_size=1, activation="relu", padding="same"
    )(W)
    W = Conv2D(channel, kernel_size=1, activation="sigmoid", padding="same")(W)
    mul = multiply([input, W])
    return mul


def FCAB(input, channel):
    conv = Conv2D(channel, kernel_size=3, padding="same")(input)
    conv = Lambda(gelu)(conv)
    conv = Conv2D(channel, kernel_size=3, padding="same")(conv)
    conv = Lambda(gelu)(conv)
    att = FCALayer(conv, channel, reduction=16)
    output = add([att, input])
    return output


def ResidualGroup(input, channel):
    conv = input
    n_RCAB = 4
    for _ in range(n_RCAB):
        conv = FCAB(conv, channel)
    conv = add([conv, input])
    return conv


def DFCAN(input_shape, scale=2):
    inputs = Input(input_shape)
    conv = Conv2D(64, kernel_size=3, padding="same")(inputs)
    conv = Lambda(gelu)(conv)
    n_ResGroup = 4
    for _ in range(n_ResGroup):
        conv = ResidualGroup(conv, channel=64)
    conv = Conv2D(64 * (scale**2), kernel_size=3, padding="same")(conv)
    conv = Lambda(gelu)(conv)

    upsampled = Lambda(pixelshuffle, arguments={"scale": scale})(conv)
    conv = Conv2D(1, kernel_size=3, padding="same")(upsampled)
    output = Activation("sigmoid")(conv)
    model = Model(inputs=inputs, outputs=output)
    return model


def gelu(x):
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))
    return x * cdf

def apodize2d(img, napodize=10):
    bs, ny, nx, ch = img.get_shape().as_list()
    img_apo = img[:, napodize:ny-napodize, :, :]

    imageUp = img[:, 0:napodize, :, :]
    imageDown = img[:, ny-napodize:, :, :]
    diff = (imageDown[:, -1::-1, :, :] - imageUp) / 2
    l = np.arange(napodize)
    fact_raw = 1 - np.sin((l + 0.5) / napodize * np.pi / 2)
    fact = fact_raw[np.newaxis, :, np.newaxis, np.newaxis]
    fact = tf.convert_to_tensor(fact, dtype=tf.float32)
    fact = tf.tile(fact, [tf.shape(img)[0], 1, nx, ch])
    factor = diff * fact
    imageUp = tf.add(imageUp, factor)
    imageDown = tf.subtract(imageDown, factor[:, -1::-1, :, :])
    img_apo = tf.concat([imageUp, img_apo, imageDown], axis=1)

    imageLeft = img_apo[:, :, 0:napodize, :]
    imageRight = img_apo[:, :, nx-napodize:, :]
    img_apo = img_apo[:, :, napodize:nx-napodize, :]
    diff = (imageRight[:, :, -1::-1, :] - imageLeft) / 2
    fact = fact_raw[np.newaxis, np.newaxis, :, np.newaxis]
    fact = tf.convert_to_tensor(fact, dtype=tf.float32)
    fact = tf.tile(fact, [tf.shape(img)[0], ny, 1, ch])
    factor = diff * fact
    imageLeft = tf.add(imageLeft, factor)
    imageRight = tf.subtract(imageRight, factor[:, :, -1::-1, :])
    img_apo = tf.concat([imageLeft, img_apo, imageRight], axis=2)

    return img_apo


def pixelshuffle(layer_in, scale):
    return tf.nn.depth_to_space(
        layer_in, block_size=scale
    )  # here I changes :  block_size=scale :  to :  block_size=2*scale  :


class NSM(Layer):

    def __init__(self, init_cutoff_freq, init_slop=100, dxy=0.0626, **kwargs):
        super().__init__(**kwargs)
        self.cutoff_freq = self.add_weight(
            shape=(1,),
            initializer=const(init_cutoff_freq),
            trainable=True,
            name="cutoff_freq",
        )
        self.slop = self.add_weight(
            shape=(1,),
            initializer=const(init_slop),
            trainable=True,
            name="slop",
        )
        self.dxy = tf.Variable(initial_value=dxy, trainable=False, name="dxy")

    def call(self, inputs, **kwargs):
        bs, ny, nx, ch = inputs.get_shape().as_list()
        nx = tf.cast(nx, tf.float32)
        ny = tf.cast(ny, tf.float32)
        dkx = tf.divide(1, (tf.multiply(nx, self.dxy)))
        dky = tf.divide(1, (tf.multiply(ny, self.dxy)))

        y = tf.multiply(
            tf.cast(tf.range(-ny // 2, ny // 2), dtype=tf.float32), dky
        )
        x = tf.multiply(
            tf.cast(tf.range(-nx // 2, nx // 2), dtype=tf.float32), dkx
        )
        [map_x, map_y] = tf.meshgrid(x, y)
        rdist = tf.sqrt(tf.square(map_x) + tf.square(map_y))

        otf_mask = tf.sigmoid(tf.multiply(self.cutoff_freq - rdist, self.slop))
        otf_mask = tf.expand_dims(tf.expand_dims(otf_mask, 0), -1)
        otf_mask = tf.tile(otf_mask, (1, 1, 1, ch))

        otf_mask = tf.complex(otf_mask, tf.zeros_like(otf_mask))
        fft_feature = fftshift(fft2(inputs))
        output = ifft2(fftshift(tf.multiply(otf_mask, fft_feature)))

        return tf.math.real(output)


# def ifft2(input):

#     temp = K.permute_dimensions(input, (0, 3, 1, 2))
#     ifft = tf.signal.ifft2d(temp)
#     output = K.permute_dimensions(ifft, (0, 2, 3, 1))
#     return output

def fft2(input):
    input = apodize2d(input, napodize=10) # the apodization has been added
    temp = K.permute_dimensions(input, (0, 3, 1, 2))
    fft = tf.signal.fft2d(tf.complex(temp, tf.zeros_like(temp)))
    absfft = tf.pow(tf.abs(fft)+1e-8, 0.1) # this line has been added
    output = K.permute_dimensions(absfft, (0, 2, 3, 1))
    return output


def fft2(input):
    temp = K.permute_dimensions(input, (0, 3, 1, 2))
    fft = tf.signal.fft2d(tf.complex(temp, tf.zeros_like(temp)))
    output = K.permute_dimensions(fft, (0, 2, 3, 1))
    return output


def fftshift(input):
    bs, h, w, ch = input.get_shape().as_list()
    fs11 = input[:, -h // 2 : h, -w // 2 : w, :]
    fs12 = input[:, -h // 2 : h, 0 : w // 2, :]
    fs21 = input[:, 0 : h // 2, -w // 2 : w, :]
    fs22 = input[:, 0 : h // 2, 0 : w // 2, :]
    output = tf.concat(
        [tf.concat([fs11, fs21], axis=1), tf.concat([fs12, fs22], axis=1)],
        axis=2,
    )
    # print(f'fftshift: {output.shape}')
#    output = tf.image.resize(output, (128, 128)) # added
    return output


def CALayer2D(input, input_height, input_width, channel, reduction=16):
    W = AveragePooling2D(pool_size=(input_height, input_width))(input)
    W = Conv2D(
        channel // reduction, kernel_size=1, activation="relu", padding="same"
    )(W)
    W = Conv2D(channel, kernel_size=1, activation="sigmoid", padding="same")(W)
    W = UpSampling2D(size=(input_height, input_width))(W)
    mul = multiply([input, W])
    return mul


def RCAB2D(input, input_height, input_width, channel):
    conv = Conv2D(channel, kernel_size=3, padding="same")(input)
    conv = LeakyReLU(alpha=0.2)(conv)
    conv = Conv2D(channel, kernel_size=3, padding="same")(conv)
    conv = LeakyReLU(alpha=0.2)(conv)
    att = CALayer2D(conv, input_height, input_width, channel, reduction=16)
    output = add([att, input])
    return output


def ResidualGroup2D(input, input_height, input_width, channel):
    conv = input
    n_RCAB = 5
    for _ in range(n_RCAB):
        conv = RCAB2D(conv, input_height, input_width, channel)
    output = add([conv, input])
    return output


def DenoiserNSM(
    input_shape, n_rg=(2, 5, 5), init_cutoff_freq=4.95, init_slop=100
):

    inputs1 = Input(input_shape)
    inputs2 = Input(input_shape)
    oa = NSM(
        init_cutoff_freq=init_cutoff_freq, init_slop=init_slop, dxy=0.0626
    )(inputs2)
    conv1 = Conv2D(32, kernel_size=3, padding="same")(oa)
    conv2 = Conv2D(32, kernel_size=3, padding="same")(inputs2)
    inputs2_oa = concatenate([conv1, conv2], axis=3)

    # --------------------------------------------------------------------------------
    #                      extract features of generated image
    # --------------------------------------------------------------------------------
    conv0 = Conv2D(64, kernel_size=3, padding="same")(inputs1)
    conv = LeakyReLU(alpha=0.2)(conv0)
    for _ in range(n_rg[0]):
        conv = ResidualGroup2D(conv, input_shape[0], input_shape[1], 64)
    conv = add([conv, conv0])
    conv = Conv2D(64, kernel_size=3, padding="same")(conv)
    conv1 = LeakyReLU(alpha=0.2)(conv)

    # --------------------------------------------------------------------------------
    #                      extract features of noisy image
    # --------------------------------------------------------------------------------
    conv0 = Conv2D(64, kernel_size=3, padding="same")(inputs2_oa)
    conv = LeakyReLU(alpha=0.2)(conv0)
    for _ in range(n_rg[1]):
        conv = ResidualGroup2D(conv, input_shape[0], input_shape[1], 64)
    conv = add([conv, conv0])
    conv = Conv2D(64, kernel_size=3, padding="same")(conv)
    conv2 = LeakyReLU(alpha=0.2)(conv)

    # --------------------------------------------------------------------------------
    #                              merge features
    # --------------------------------------------------------------------------------
    conct = add([conv1, conv2])
    conct = Conv2D(64, kernel_size=3, padding="same")(conct)
    conct = LeakyReLU(alpha=0.2)(conct)
    conv = conct

    for _ in range(n_rg[2]):
        conv = ResidualGroup2D(conv, input_shape[0], input_shape[1], 64)
    conv = add([conv, conct])

    conv = Conv2D(256, kernel_size=3, padding="same")(conv)
    conv = LeakyReLU(alpha=0.2)(conv)

    CA = CALayer2D(conv, input_shape[0], input_shape[1], 256, reduction=16)
    conv = Conv2D(input_shape[2], kernel_size=3, padding="same")(CA)

    output = LeakyReLU(alpha=0.2)(conv)

    model = Model(inputs=[inputs1, inputs2], outputs=output)
    return model


def Denoiser(input_shape, n_rg=(2, 5, 5)):

    inputs1 = Input(input_shape)
    inputs2 = Input(input_shape)
    # --------------------------------------------------------------------------------
    #                      extract features of generated image
    # --------------------------------------------------------------------------------
    conv0 = Conv2D(64, kernel_size=3, padding="same")(inputs1)
    conv = LeakyReLU(alpha=0.2)(conv0)
    for _ in range(n_rg[0]):
        conv = ResidualGroup2D(conv, input_shape[0], input_shape[1], 64)
    conv = add([conv, conv0])
    conv = Conv2D(64, kernel_size=3, padding="same")(conv)
    conv1 = LeakyReLU(alpha=0.2)(conv)

    # --------------------------------------------------------------------------------
    #                      extract features of noisy image
    # --------------------------------------------------------------------------------
    conv0 = Conv2D(64, kernel_size=3, padding="same")(inputs2)
    conv = LeakyReLU(alpha=0.2)(conv0)
    for _ in range(n_rg[1]):
        conv = ResidualGroup2D(conv, input_shape[0], input_shape[1], 64)
    conv = add([conv, conv0])
    conv = Conv2D(64, kernel_size=3, padding="same")(conv)
    conv2 = LeakyReLU(alpha=0.2)(conv)

    # --------------------------------------------------------------------------------
    #                              merge features
    # --------------------------------------------------------------------------------
    weight1 = Lambda(lambda x: x * 1)
    weight2 = Lambda(lambda x: x * 1)
    conv1 = weight1(conv1)
    conv2 = weight2(conv2)

    conct = add([conv1, conv2])
    conct = Conv2D(64, kernel_size=3, padding="same")(conct)
    conct = LeakyReLU(alpha=0.2)(conct)
    conv = conct

    for _ in range(n_rg[2]):
        conv = ResidualGroup2D(conv, input_shape[0], input_shape[1], 64)
    conv = add([conv, conct])

    conv = Conv2D(256, kernel_size=3, padding="same")(conv)
    conv = LeakyReLU(alpha=0.2)(conv)

    CA = CALayer2D(conv, input_shape[0], input_shape[1], 256, reduction=16)
    conv = Conv2D(input_shape[2], kernel_size=3, padding="same")(CA)

    output = LeakyReLU(alpha=0.2)(conv)

    model = Model(inputs=[inputs1, inputs2], outputs=output)
    return model


class Train_RDL_Denoising(tf.keras.Model):
    def __init__(self, srmodel, denmodel, loss_fn, optimizer,  parameters, PSF = 'given', verbose = True):
        super().__init__()
        self.srmodel = srmodel
        self.denmodel = denmodel
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.PSF = PSF
        self.verbose = verbose
        self.parameters = parameters 
        self.epochs = self.parameters['epochs']
        self.nphases = self.parameters['nphases']
        self.ndirs = self.parameters['ndirs']
        self.space = self.parameters['space']
        self.Ny = self.parameters['Ny']
        self.Nx = self.parameters['Nx']
        self.phase_space = 2 * np.pi / self.nphases
        self.scale = self.parameters['scale']
        self.dxy = self.parameters['dxy']
        self.sigma_x = self.parameters['sigma_x']
        self.sigma_y = self.parameters['sigma_y']
        self.dxy = self.parameters['dxy']
        self.sr_model_dir = self.parameters['sr_model_dir']
        self.den_model_dir = self.parameters['den_model_dir']
        self.log_dir = self.parameters['log_dir']
        self.batch_size = self.parameters['batch_size']
        [self.Nx_hr, self.Ny_hr] = [self.Nx* self.scale, self.Ny* self.scale] 
        [self.dx_hr, self.dy_hr] = [x / self.scale for x in [self.dxy, self.dxy]]

        xx = self.dx_hr * np.arange(-self.Nx_hr / 2, self.Nx_hr / 2, 1)
        yy = self.dy_hr * np.arange(-self.Ny_hr / 2, self.Ny_hr / 2, 1)
        [self.X, self.Y] = np.meshgrid(xx, yy)
        
        self.dkx = 1.0 / ( self.Nx *  self.dxy)
        self.dky = 1.0 / ( self.Ny * self.dxy)
        self.prol_OTF = None
        self.otf_path = self.parameters['otf_path']
        self.results_path = self.parameters['results_path']
        self.dkr = np.min([self.dkx, self.dky])
        
        if self.PSF is None:
            
            self.PSF, self.OTF = create_psf(self.sigma_x, 
                        self.sigma_x,
                        self.Nx_hr, 
                        self.Ny_hr, 
                        self.dkx, 
                        self.dky)
        elif self.PSF == 'given': # read out the PSF from the RDL_Sim pakage
            # print(f'in read otf being feed: Nx_hr : {self.Nx_hr} Ny_hr : {self.Ny_hr} dkx : {self.dkx} dky : {self.dky} dkr : {self.dkr}')
            # in read otf being feed: 
            # Nx_hr : 256.0 
            # Ny_hr : 256.0 
            # dkx : 0.12480031948881788 
            # dky : 0.12480031948881788 
            # dkr : 0.12480031948881788
            # extract the psf from the parameters
            self.OTF, self.prol_OTF, self.PSF = get_otf(self.otf_path, 
                                                        self.Nx_hr, 
                                                        self.Ny_hr, 
                                                        self.dkx, 
                                                        self.dky, 
                                                        self.dkr)
            print('Information OTF how its read')
            # (256, 256) float64 1.0 -0.0006461802372663593
            print(f'{self.OTF.shape} {self.OTF.dtype} {np.max(self.OTF)} {np.min(self.OTF)}')
            print()
            print('Information PSF how its read')
            # (256, 256) float64 0.011698316186498404 1.0611680661324662e-10
            print(f'{self.PSF.shape} {self.PSF.dtype} {np.max(self.PSF)} {np.min(self.PSF)}')
            print()
            print('Information prol_OTF how its read')
            # (366,) float64 1.0 -0.0006461802372663593
            print(f'{self.prol_OTF.shape} {self.prol_OTF.dtype} {np.max(self.prol_OTF)} {np.min(self.prol_OTF)}')


     
            
        else:
            self.PSF /= np.sum(self.PSF)  
            self.OTF = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.PSF)))
            self.OTF /= np.sum(self.OTF)
        
        if self.verbose:
            fig, axes = plt.subplots(1, 2, figsize=(15, 15))
            axes[0].imshow(self.PSF)
            axes[0].set_title('PSF')
            
            axes[1].imshow(abs(self.OTF))
            axes[1].set_title('OTF')

            plt.tight_layout()
            plt.show()
            plt.savefig(f'{self.results_path }/PSF_OTF.png', bbox_inches='tight')

        # print(f'PSF shape: {self.PSF} {self.PSF.shape}{type(self.PSF)}, OTF shape {self.OTF} {type(self.OTF)}')
        # print(f'nx: {self.Nx}{type(self.Nx)}  :: Ny: {self.Ny} {type(self.Ny)}')
        # print(f'Nx_hr: {self.Nx_hr} {type(self.Nx_hr)} :: Ny_hr: {self.Ny_hr} {type(self.Ny_hr)}')
        # print(f'dkx: {self.dkx} {type(self.dkx)} :: dky: {self.dky} {type(self.dky)}')
        # print(f'X: {self.X} {type(self.X)} :: Y: {self.Y} {type(self.Y)}')
    
    def _phase_computation(self, img_SR, modamp, cur_k0_angle, cur_k0):

            phase_list = -np.angle(modamp)
            img_gen = []
            for d in range(self.ndirs):
                alpha = cur_k0_angle[d]
                
                for i in range(self.nphases):
                    kxL = cur_k0[d] * np.pi * np.cos(alpha)
                    kyL = cur_k0[d] * np.pi * np.sin(alpha)
                    kxR = -cur_k0[d] * np.pi * np.cos(alpha)
                    kyR = -cur_k0[d] * np.pi * np.sin(alpha)
                    phOffset = phase_list[d] + i * self.phase_space
                    interBeam = np.exp(1j * (kxL * self.X + kyL * self.Y + phOffset)) + np.exp(1j * (kxR * self.X + kyR * self.Y))
                    pattern = normalize(np.square(np.abs(interBeam)))
                    
                    # print(f'Raw pattern information \n :  shape: {pattern.shape} dtype: {pattern.dtype} max: {np.max(pattern)} min: {np.min(pattern)} \n ')
                    # :  shape: (256, 256) dtype: float32 max: 1.0000102519989014 min: -0.0022249433677643538 
                    patterned_img_fft = F.fftshift(F.fft2(pattern * img_SR)) * self.OTF
                    # print(f'patterned_img_fft information \n :  shape: {patterned_img_fft.shape} dtype: {patterned_img_fft.dtype} max: {np.max(patterned_img_fft)} min: {np.min(patterned_img_fft)} \n ')
                    # :  shape: (256, 256) dtype: complex128 max: (2125.2898119242027+0j) min: (-158.72604990708248-125.02544266014829j) 
                    modulated_img = np.abs(F.ifft2(F.ifftshift(patterned_img_fft)))
                    # print(f'modulated_img information \n :  shape: {modulated_img.shape} dtype: {modulated_img.dtype} max: {np.max(modulated_img)} min: {np.min(modulated_img)} \n ')
                    # :  shape: (256, 256) dtype: float64 max: 0.10262195842657973 min: 0.015000125525139074 
                    modulated_img = normalize(cv2.resize(modulated_img, (self.Ny, self.Nx)))  
                    # print(f'modulated_img after resize information \n :  shape: {modulated_img.shape} dtype: {modulated_img.dtype} max: {np.max(modulated_img)} min: {np.min(modulated_img)} \n ')  
                    # :  shape: (128, 128) dtype: float32 max: 1.139858603477478 min: -0.04895542189478874 
                    img_gen.append(modulated_img)
          
           
            img_gen = np.asarray(img_gen)
            
            return img_gen
    
    
    def _get_cur_k(self, image_gt):
        
        cur_k0, modamp = cal_modamp(np.array(image_gt).astype(np.float32), self.prol_OTF, self.parameters)  #  prol_OTF in the main code base

        # print(f'from cal_modamp')
        # print(f'cur_k0: {cur_k0}   modamp: {modamp}')
        
        cur_k0_angle = np.array(np.arctan2(cur_k0[:, 1] , cur_k0[:, 0]))
        cur_k0_angle[1:self.parameters['ndirs']] = cur_k0_angle[1:self.parameters['ndirs']] + np.pi
        cur_k0_angle = -(cur_k0_angle - np.pi/2)
        
        for nd in range(self.parameters['ndirs']):
            if np.abs(cur_k0_angle[nd] - self.parameters['k0angle_g'][nd]) > 0.05:
                cur_k0_angle[nd] = self.parameters['k0angle_g'][nd]
        cur_k0 = np.sqrt(np.sum(np.square(cur_k0), 1))
        given_k0 = 1 / self.parameters['space']
        cur_k0[np.abs(cur_k0 - given_k0) > 0.1] = given_k0
        # print(" _get_cur_k")  
        # cur_k0: [5.53278689 5.5390269  5.53278689]  cur_k0_angle: [ 0.0908 -0.9564 -2.0036] 
        # modamp: [(0.11613527390398079-0.023063085715872362j), 
        # (0.0038545362463064513-0.0007995343811814879j), 
        # (0.15820753032403853+0.0027214278710692894j)]


        # print(f'cur_k0: {cur_k0}  cur_k0_angle: {cur_k0_angle} modamp: {modamp}')
        # cur_k0: [5.53278689 5.5390269  5.53278689]  
        # cur_k0_angle: [ 0.0908 -0.9564 -2.0036] 
        # modamp: [(0.11613527390398079-0.023063085715872362j), 
        #         (0.0038545362463064513-0.0007995343811814879j), 
        #         (0.15820753032403853+0.0027214278710692894j)]
  
        return cur_k0, cur_k0_angle, modamp
    

    def reshape_to_3_channels(self, batch):
        
        B, H, W, C = batch.shape
        #print(f'B: {B} , H: {H}, W: {W} , C: {C} ')
        assert C % self.ndirs == 0, "The last dimension must be divisible by 3"
        new_batch_size = B * (C // self.ndirs)
        return batch.reshape(new_batch_size, H, W, self.nphases)
    def reshape_to_9_channels(self, batch):
    
        B, H, W, C = batch.shape
        #print(f'B: {B} , H: {H}, W: {W} , C: {C} ')
        
        new_batch_size = int(B / (self.ndirs * self.nphases / C))
        #print(f' new_batch_size : { new_batch_size} B: {B} , H: {H}, W: {W} , C: {C} ')
        return batch.reshape(new_batch_size, H, W, self.ndirs * self.nphases)
    
    def fit(self, data, data_val):
        x, y = data
        x_val, y_val = data_val
        input_height = x.shape[1]
        input_width = x.shape[2]
        channels = x.shape[-1]
        tensorboard_callback = callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        # lrate = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=4, verbose=1)
        lrate= callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,  # here is teh change 
                            patience=15, mode='auto', min_delta=1e-4,
                            cooldown=0, min_lr = max(1e-4 * 0.1, 1e-5), verbose=1)
        hrate = callbacks.History()
        srate = callbacks.ModelCheckpoint(
                    str(self.den_model_dir),
                    monitor="loss",
                    save_best_only=True, # here is teh change
                    save_weights_only=False,
                    mode="auto",
                )
        sr_y_predict = self.srmodel.predict(x)
        sr_y_predict = tf.squeeze(sr_y_predict, axis=-1) # Batch, Ny, Nx, 1 

        ############################### cleaning up the model branches ###############################
        # to do later
        # input_MPE_batch = MPE_branch(sr_y_predict, gt_image)
        # input_PFE_batch = PPE_branch(x)
        # gt_batch = GT_branch(y)


        ##############################################################################################
        # Loop over each example in the batch
        
        list_image_gen = []
        list_image_in = []
        list_image_gt = []
        for i in range(x.shape[0]):
            img_in = x[i:i+1][0]  
            img_SR = sr_y_predict[i:i+1][0]   
            image_gt = y[i:i+1][0] 
            cur_k0, cur_k0_angle, modamp = self._get_cur_k(image_gt=image_gt)
           # print(f'img_SR before phase computation {img_SR.shape}')
            # print(f'img_SR before phase computation {img_SR.shape} dtype: {img_SR.dtype} max: {np.max(img_SR)} min: {np.min(img_SR)}')
            image_gen = self._phase_computation(img_SR, modamp, cur_k0_angle, cur_k0)
            # print('printing Image gen')
            # image_gen shape: (9, 128, 128) dtype: float32 max: 1.1431851387023926 min: -0.04943832382559776
            # print(f'image_gen shape: {image_gen.shape} dtype: {image_gen.dtype} max: {np.max(image_gen)} min: {np.min(image_gen)}')
            #print(f'image_gen from phase computation {image_gen.shape}')

            # print('Image In')
            # img_in shape: (128, 128, 9) dtype: <dtype: 'float32'> max: 1.9397590160369873 min: -0.33734938502311707
            # print(f'img_in shape: {img_in.shape} dtype: {img_in.dtype} max: {np.max(img_in)} min: {np.min(img_in)}')
            # print()
            # print('Image GT')
            # image_gt shape: (128, 128, 9) dtype: <dtype: 'float32'> max: 1.525308609008789 min: 0.008641975000500679
            # print(f'image_gt shape: {image_gt.shape} dtype: {image_gt.dtype} max: {np.max(image_gt)} min: {np.min(image_gt)}')

            image_gen = np.transpose(image_gen, (1, 2, 0))
            list_image_gen.append(image_gen)
            list_image_in.append(img_in)
            list_image_gt.append(image_gt)
        
        input_MPE_batch = np.asarray(list_image_gen)
        input_PFE_batch = np.asarray(list_image_in)
        gt_batch = np.asarray(list_image_gt) 
        
        input_MPE_batch = self.reshape_to_3_channels(input_MPE_batch)
        input_PFE_batch = self.reshape_to_3_channels(input_PFE_batch)
        gt_batch = self.reshape_to_3_channels(gt_batch)    
        

        # print("\n ########  This is being feed into the model ######## \n")
        # print(f'input MPE {input_MPE_batch.shape} dtype : {input_MPE_batch.dtype} max: {np.max(input_MPE_batch)} min: {np.min(input_MPE_batch)}')
        # print(f'input PFE {input_PFE_batch.shape} dtype : {input_PFE_batch.dtype} max: {np.max(input_PFE_batch)} min: {np.min(input_PFE_batch)}')
        # print(f'gt {gt_batch.shape} dtype : {gt_batch.dtype} max: {np.max(gt_batch)} min: {np.min(gt_batch)}')
        
#  ########  This is being feed into the model ######## 

# input MPE (150, 128, 128, 3) dtype : float32 max: 1.5757898092269897 min: -0.17251379787921906
# input PFE (150, 128, 128, 3) dtype : float32 max: 2.247126340866089 min: -0.3700000047683716
# gt (150, 128, 128, 3) dtype : float32 max: 2.2066338062286377 min: -0.01050929632037878


        if self.verbose:
            print(f'input MPE {input_MPE_batch.shape}, input PFE {input_PFE_batch.shape},gt {gt_batch.shape}')
            plot_batches(self, input_MPE_batch, input_PFE_batch, gt_batch) # have to pass epoch
        self.denmodel.fit([input_MPE_batch, input_PFE_batch], gt_batch, batch_size=self.batch_size,
                            epochs=self.epochs, shuffle=True,
                            callbacks=[lrate, hrate, srate, tensorboard_callback])
        print('saving the  trained DN model')
        self.denmodel.save(self.den_model_dir)
        print('model has been saved')


    def predict(self, data):
        
            x = data
            input_height = x.shape[1]
            input_width = x.shape[2]
            channels = x.shape[-1]
            sr_y_predict = self.srmodel.predict(x)
            sr_y_predict = tf.squeeze(sr_y_predict, axis=-1) # Batch, Ny, Nx, 1 

            ############################### cleaning up the model branches ###############################
            # input_MPE_batch = MPE_branch(sr_y_predict, gt_image)
            # input_PFE_batch = PPE_branch(x)
            # gt_batch = GT_branch(y)


            ##############################################################################################
            # Loop over each example in the batch
            
            list_image_gen = []
            list_image_in = []
            #list_image_gt = []
            for i in range(x.shape[0]):
                img_in = x[i:i+1][0]  
                img_SR = sr_y_predict[i:i+1][0]   
                #image_gt = y[i:i+1][0] 
                cur_k0, cur_k0_angle, modamp = self._get_cur_k(image_gt=img_in)
            # print(f'img_SR before phase computation {img_SR.shape}')
                
                image_gen = self._phase_computation(img_SR, modamp, cur_k0_angle, cur_k0)
                #print(f'image_gen from phase computation {image_gen.shape}')
                image_gen = np.transpose(image_gen, (1, 2, 0))
                list_image_gen.append(image_gen)
                list_image_in.append(img_in)
                #list_image_gt.append(image_gt)
            
            input_MPE_batch = np.asarray(list_image_gen)
            input_PFE_batch = np.asarray(list_image_in)
            #gt_batch = np.asarray(list_image_gt) 
            
            input_MPE_batch = self.reshape_to_3_channels(input_MPE_batch)
            input_PFE_batch = self.reshape_to_3_channels(input_PFE_batch)
            #gt_batch = self.reshape_to_3_channels(gt_batch)    
            predictions = self.denmodel.predict([input_MPE_batch, input_PFE_batch])
            return self.reshape_to_9_channels(predictions)

        
def plot_batches(self, input_MPE_batch, input_PFE_batch, gt_batch):
    num_batches = input_MPE_batch.shape[0]
    random_indices = np.random.choice(num_batches, 5, replace=False)
    
    fig, axes = plt.subplots(5, 3, figsize=(15, 15))
    
    for i, idx in enumerate(random_indices):
        axes[i, 0].imshow(input_MPE_batch[idx])
        axes[i, 0].set_title(f'input MPE batch {idx} shape: {input_MPE_batch[idx].shape} \ndtype: {input_MPE_batch[idx].dtype}\n max: {np.max(input_MPE_batch[idx])} \nmin: {np.min(input_MPE_batch[idx])}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(input_PFE_batch[idx])
        axes[i, 1].set_title(f'input PFE batch {idx} shape: {input_PFE_batch[idx].shape} \ndtype: {input_PFE_batch[idx].dtype}\n max: {np.max(input_PFE_batch[idx])} \nmin: {np.min(input_PFE_batch[idx])}')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(gt_batch[idx])
        axes[i, 2].set_title(f'gt batch {idx} shape: {gt_batch[idx].shape} \ndtype: {gt_batch[idx].dtype} \nmax: {np.max(gt_batch[idx])}\n min: {np.min(gt_batch[idx])}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{self.results_path }/DN_output_features_brances_.png', bbox_inches='tight') 
    plt.show()    
       