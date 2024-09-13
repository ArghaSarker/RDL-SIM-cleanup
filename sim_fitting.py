import numpy as np 
import numpy.fft as F
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt 
from read_otf import read_otf

def make_matrix(nphases, norders):
    sepMatrix = np.zeros((2*norders-1, nphases))
    phi = 2 * np.pi / nphases

    j = np.arange(0, nphases, 1)
    order = np.arange(1, norders, 1)

    sepMatrix[0, :] = 1.0
    sepMatrix[1, :] = np.cos(order * j * phi)
    sepMatrix[2, :] = np.sin(order * j * phi)

    return sepMatrix



def apodize(napodize, image):
    """
    Apply apodization to the input image to smooth its edges.
    
    Parameters:
    napodize (int): The number of pixels to apply the apodization.
    image (np.ndarray): The input 3D image to be apodized (height x width x channels).
    
    Returns:
    np.ndarray: The apodized image.
    """
    # Initialize apoimage as a copy of the input image
    apoimage = image.copy()
    
    if napodize > 0:
        # Apodization factor calculation
        l = np.arange(napodize)
        fact = 1 - np.sin((l + 0.5) / napodize * np.pi / 2)
        
        # Apply vertical apodization
        imageUp = image[:napodize, :, :]
        imageDown = image[-napodize:, :, :]
        diff_up = (imageUp - imageUp[::-1, :, :]) / 2
        diff_down = (imageDown[::-1, :, :] - imageDown) / 2
        factor_up = diff_up * fact[:, None, None]
        factor_down = diff_down * fact[:, None, None]
        apoimage[:napodize, :, :] = imageUp - factor_up
        apoimage[-napodize:, :, :] = imageDown - factor_down[::-1, :, :]

        # Apply horizontal apodization
        imageLeft = apoimage[:, :napodize, :]
        imageRight = apoimage[:, -napodize:, :]
        diff_left = (imageLeft - imageLeft[:, ::-1, :]) / 2
        diff_right = (imageRight[:, ::-1, :] - imageRight) / 2
        factor_left = diff_left * fact[None, :, None]
        factor_right = diff_right * fact[None, :, None]
        apoimage[:, :napodize, :] = imageLeft - factor_left
        apoimage[:, -napodize:, :] = imageRight - factor_right[:, ::-1, :]

    return apoimage




def makeoverlaps(bands, Nx, Ny, order1, order2, k0x, k0y, dxy, dz, OTF, lamda):
 
    kx = k0x * (order2 - order1)
    ky = k0y * (order2 - order1)

    dkx = 1 / (Nx * dxy)
    dky = 1 / (Ny * dxy)
    dkr = min(dkx, dky)
    rdistcutoff = min(1.35 * 2 / lamda, Nx / 2 * dkx, Ny / 2 * dky)

    x1 = np.linspace(-Nx / 2, Nx / 2 - 1, Nx) * dkx
    y1 = np.linspace(-Ny / 2, Ny / 2 - 1, Ny) * dky

    X1, Y1 = np.meshgrid(x1, y1)
    rdist1 = np.sqrt(X1**2 + Y1**2)

    X12, Y12 = np.meshgrid(x1 - kx, y1 - ky)
    rdist12 = np.sqrt(X12**2 + Y12**2)

    X21, Y21 = np.meshgrid(x1 + kx, y1 + ky)
    rdist21 = np.sqrt(X21**2 + Y21**2)

    mask1 = (rdist1 <= rdistcutoff) & (rdist12 <= rdistcutoff)
    mask2 = (rdist1 <= rdistcutoff) & (rdist21 <= rdistcutoff)

    if order1 == 0:
        band1re = bands[0]
        band1im = None
    else:
        band1re = bands[order1 * 2 - 1]
        band1im = bands[order1 * 2]

    band2re = bands[order2 * 2 - 1]
    band2im = bands[order2 * 2]

    # Ensure OTF interpolation is correctly shaped
    rdist_flat = rdist1.flatten()
    OTF_flat = OTF.flatten()
    interp = interp1d(np.arange(OTF.size), OTF_flat, kind='linear', bounds_error=False, fill_value=0)
    OTF1 = interp(rdist_flat).reshape((Ny, Nx))
    OTF2 = interp(rdist_flat).reshape((Ny, Nx))
    OTF1_sk0 = interp(rdist21.flatten()).reshape((Ny, Nx))
    OTF2_sk0 = interp(rdist12.flatten()).reshape((Ny, Nx))

    OTF1_mag = np.abs(OTF1)
    OTF2_mag = np.abs(OTF2)
    OTF1_sk0_mag = np.abs(OTF1_sk0)
    OTF2_sk0_mag = np.abs(OTF2_sk0)



    
    root1 = np.sqrt(OTF1_mag**2 + OTF2_sk0_mag**2)
    root2 = np.sqrt(OTF1_sk0_mag**2 + OTF2_mag**2)

    fact1 = np.zeros((Ny, Nx))
    fact1[mask1] = OTF2_sk0[mask1] / root1[mask1]
    val1re = band1re * fact1
    if order1 > 0:
        val1im = band1im * fact1
        overlap0 = val1re + 1j * val1im
    else:
        overlap0 = val1re

    fact2 = np.zeros((Ny, Nx))
    fact2[mask2] = OTF1_sk0[mask2] / root2[mask2]
    val2re = band2re * fact2
    val2im = band2im * fact2
    overlap1 = val2re + 1j * val2im

    overlap0 = F.ifft2(F.ifftshift(overlap0))
    overlap1 = F.ifft2(F.ifftshift(overlap1))

    return overlap0, overlap1

def findk0(bands, overlap0, overlap1, Nx, Ny, Nz, k0, dxy, dz, OTF, lamda):
    dkx = 1 / (Nx * dxy)
    dky = 1 / (Ny * dxy)
    fitorder1 = 0
    fitorder2 = 1

    overlap0, overlap1 = makeoverlaps(bands, Nx, Ny,fitorder1, fitorder2, k0[0], k0[1], dxy, dz, OTF, lamda)
    
    if Nz > 1:
        crosscorr_c = np.sum(overlap0.conjugate() * overlap1, axis=3)
    else:
        crosscorr_c = overlap0.conjugate() * overlap1

    crosscorr = F.fftshift(F.ifft2(F.ifftshift(crosscorr_c)))
    index_k0 = np.argmax(np.abs(crosscorr)**2)
    new_k0 = np.array([index_k0 % Nx, index_k0 // Nx])
    new_k0 = new_k0 - np.array([Nx / 2, Ny / 2]) - 1

    # Adjust k0 if it's out of the FFT frequency range
    new_k0[0] = np.where(k0[0] / dkx < new_k0[0] - Nx / 2, new_k0[0] - Nx, new_k0[0])
    new_k0[0] = np.where(k0[0] / dkx > new_k0[0] + Nx / 2, new_k0[0] + Nx, new_k0[0])
    new_k0[1] = np.where(k0[1] / dky < new_k0[1] - Ny / 2, new_k0[1] - Ny, new_k0[1])
    new_k0[1] = np.where(k0[1] / dky > new_k0[1] + Ny / 2, new_k0[1] + Ny, new_k0[1])

    new_k0 = new_k0 / fitorder2
    new_k0 = new_k0 * np.array([dkx, dky])

    return new_k0

def fitxyparabola(x1, y1, x2, y2, x3, y3):
    x1, y1, x2, y2, x3, y3 = map(np.asarray, [x1, y1, x2, y2, x3, y3])
    
    # Check for equal points
    mask_invalid = (x1 == x2) | (x2 == x3) | (x3 == x1)
    eps = 1.0e-10
    # Compute intermediate values
    xbar1 = 0.5 * (x1 + x2)
    xbar2 = 0.5 * (x2 + x3)
    slope1 = (y2 - y1) / (x2 - x1 + eps)
    slope2 = (y3 - y2) / (x3 - x2 + eps)
    curve = (slope2 - slope1) / (xbar2 - xbar1)
    
    # Compute peak
    with np.errstate(divide='ignore', invalid='ignore'):
        peak = np.where(curve != 0, xbar2 - slope2 / curve, 0)
    
    # Handle invalid cases
    peak[mask_invalid] = 0

    return peak

def fitk0andmodamps(bands, overlap0, overlap1, Nx, Ny, Nz, k0, dxy, dz, OTF, lamda, parameters):
    # Find optimal k0 and modammps
    deltaangle = 0.001
    
    if Nx >= Ny:
        dkx = 1 / (Nx * dxy)
        deltamag = 0.1 * dkx
    else:
        dky = 1 / (Ny * dxy)
        deltamag = 0.1 * dky

    fitorder1 = 0
    fitorder2 = 2 if Nz > 1 else 1

    k0mag = np.sqrt(k0[0]**2 + k0[1]**2)
    k0angle = np.arctan2(k0[1], k0[0])

    redoarrays = parameters["recalcarrays"] >= 1

    def get_modamp_wrapper(angle, mag):
        return np.abs(getmodamp(angle, mag, bands, overlap0, overlap1, Nx, Ny, fitorder1, fitorder2, dxy, dz, OTF, lamda, redoarrays, parameters))

    # Search for optimal k0 angle
    x1 = k0angle
    amp1 = get_modamp_wrapper(k0angle, k0mag)
    
    x2 = k0angle + deltaangle
    amp2 = get_modamp_wrapper(k0angle + deltaangle, k0mag)
    
    if amp2 > amp1:
        while amp2 > amp1:
            x1 = k0angle
            amp1 = amp2
            x2 += deltaangle
            amp2 = get_modamp_wrapper(x2, k0mag)
    else:
        angle = k0angle
        a = amp2
        amp2 = amp1
        amp1 = a
        a = x2
        x2 = x1
        x1 = a
        while amp2 > amp1:
            amp1 = amp2
            x2 = x1
            angle -= deltaangle
            x1 = angle
            amp2 = get_modamp_wrapper(angle, k0mag)
    
    angle = fitxyparabola(x1, amp1, k0angle, amp2, x2, amp2)

    # Search for optimal k0 magnitude
    x1 = k0mag
    amp1 = get_modamp_wrapper(angle, k0mag)
    
    x2 = k0mag + deltamag
    amp2 = get_modamp_wrapper(angle, k0mag + deltamag)
    
    if amp2 > amp1:
        while amp2 > amp1:
            x1 = k0mag
            amp1 = amp2
            x2 += deltamag
            amp2 = get_modamp_wrapper(angle, x2)
    else:
        mag = k0mag
        a = amp2
        amp2 = amp1
        amp1 = a
        a = x2
        x2 = x1
        x1 = a
        while amp2 > amp1:
            amp1 = amp2
            x2 = x1
            mag -= deltamag
            x1 = mag
            amp2 = get_modamp_wrapper(angle, mag)
    
    mag = fitxyparabola(x1, amp1, k0mag, amp2, x2, amp2)

    if parameters['ifshowmodamp'] == 1:
        print('Optimum modulation amplitude:\n')

    redoarrays = parameters['recalcarrays'] >= 2
    modamp = getmodamp(angle, mag, bands, overlap0, overlap1, Nx, Ny,  fitorder1, fitorder2, dxy, dz, OTF, lamda, redoarrays, parameters)

    if parameters['ifshowmodamp'] == 1:
        print('Optimum k0 angle=%f rad, length=%f 1/microns, spacing=%f microns\n'% (angle, mag, 1.0 / mag))

    new_k0 = [mag * np.cos(angle), mag * np.sin(angle)]
    amps = modamp
    return new_k0, amps


def getmodamp(k0angle, k0length, bands, overlap0, overlap1, Nx, Ny, order1, order2, dxy, dz, OTF, lamda, redoarrays, parameters):
    k1 = np.array([k0length * np.cos(k0angle), k0length * np.sin(k0angle)])

    if redoarrays > 0:
        overlap0, overlap1 = makeoverlaps(bands, Nx, Ny, order1, order2, k1[0], k1[1], dxy, dz, OTF, lamda)

    dkx = 1 / (Nx * dxy)
    dky = 1 / (Ny * dxy)

    xcent = Nx / 2
    ycent = Ny / 2
    kx = k1[0] * (order2 - order1)
    ky = k1[1] * (order2 - order1)

    jj, ii = np.meshgrid(np.arange(Nx), np.arange(Ny))
    angle = 2 * np.pi * ((jj - xcent) * (kx / dkx) / Nx + (ii - ycent) * (ky / dky) / Ny)
    expiphi = np.exp(1j * angle)

    overlap1_shift = overlap1 * expiphi
    sumXstarY = np.sum(overlap0.conjugate() * overlap1_shift)
    sumXmag = np.sum(np.abs(overlap0)**2) + np.sum(np.abs(overlap1)**2)

    modamp = sumXstarY / sumXmag
    
    if parameters['ifshowmodamp'] == 1:
        print('In getmodamp: angle=%f, mag=%f 1/micron, amp=%f, phase=%f' % (k0angle, k0length, np.abs(modamp), np.angle(modamp)))
    
    return modamp



def cal_modamp(image, OTF, parameters):
    # Parameters initialization
    Nx, Ny = parameters['Nx'], parameters['Ny']
    lamda = parameters['wavelength']
    dxy = parameters['dxy']
    ndirs, nphases = parameters['ndirs'], parameters['nphases']
    norders, napodize = parameters['norders'], parameters['napodize']
    k0mod, k0angle = parameters['k0mod'], parameters['k0angle_c']
    k0 = np.transpose([k0mod * np.cos(k0angle), k0mod * np.sin(k0angle)])
    # Calculate modamp
    image = np.reshape(image, [Ny, Nx, ndirs, nphases])
    modamp = []
    cur_k0 = np.copy(k0)
    for d in range(ndirs):
        sepMatrix = make_matrix(nphases, norders)
        imagePro = np.squeeze(image[:, :, d, :])
        imagePro = np.transpose(imagePro, (1, 0, 2))
        Fimagepro = np.fft.fftshift(np.fft.fft2(apodize(napodize, imagePro)))

        bandsDir = np.dot(sepMatrix, Fimagepro.reshape((nphases, Ny * Nx)))
        bandsDir = np.reshape(bandsDir, (nphases, Nx, Ny))

        fitorder0 = 0
        fitorder1 = 1
        overlap0, overlap1 = makeoverlaps(bandsDir, Ny, Nx, fitorder0, fitorder1, k0[d, 0], k0[d, 1], dxy, 0, OTF, lamda)
        new_k0, cur_modamp = fitk0andmodamps(bandsDir, overlap0, overlap1, Ny, Nx, 1, k0[d, :], dxy, 0, OTF, lamda, parameters)
        cur_k0[d, :] = new_k0

        modamp.append(cur_modamp)
    return cur_k0, modamp
    


def create_psf(sigma_x, sigma_y, Nx_hr, Ny_hr, dkx, dky):
    kxx = dkx * np.arange(-Nx_hr / 2, Nx_hr / 2, 1)
    kyy = dky * np.arange(-Ny_hr / 2, Ny_hr / 2, 1)
    [dX, dY] = np.meshgrid(kxx, kyy)
    PSF = np.exp(-0.5 * ((dX / sigma_x) ** 2 + (dY / sigma_y) ** 2))
    PSF /= np.sum(PSF)  
    OTF = abs(F.ifftshift(F.ifft2(PSF)))
    OTF /= np.sum(OTF) 
    return PSF, OTF

def get_otf(otf_path, Nx_hr, Ny_hr, dkx, dky, dkr):
    OTF, prol_OTF, PSF = read_otf(otf_path, Nx_hr, Ny_hr, dkx, dky, dkr)
    return OTF, prol_OTF, PSF