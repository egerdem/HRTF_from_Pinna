
import numpy as np
from scipy.special import sph_harm
import matplotlib
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import tqdm
from utils import SonicomDatabase
import pickle
matplotlib.use('Qt5Agg')  # For PyQt5 or PySide2import numpy as np
import sofar as sf
from scipy.io.wavfile import write
from metrics import MeanSpectralDistortion
import torch
import sofar
from os.path import abspath, dirname, join

def dump(file_name, variable):
    file = open(file_name, 'wb')
    pickle.dump(variable, file)
    file.close()
    print(f"{file_name=} is dumped")
    return

def load(file_name):
    file = open(file_name, 'rb')
    data = pickle.load(file)
    file.close()
    print("%s is loaded " % file_name)
    return data

def compute_HRTF_from_hrir(hrir):
    """
    This function compute the RFFT of the given HRIRs and return HRTFs.
    nfft = 256
    Args:
          HRIRs (time domain)
        Returns:
           HRTFs (Frequency domain)"""
    nfft = 256
    return np.fft.rfft(hrir, n=nfft)


def convert_to_HRIR_from_hrtf(hrtfs):
    return np.fft.irfft(hrtfs, axis=-1)

def sph_harm_real(m, n, phi, theta):

    if m > 0:
        return np.sqrt(2) * (-1) ** m * np.real(sph_harm(m, n, phi, theta))
    elif m < 0:
        return np.sqrt(2) * (-1) ** m * np.imag(sph_harm(-m, n, phi, theta))
    else:  # m == 0
        return sph_harm(0, n, phi, theta).real

def sph_harm_realw(m, n, phi, theta):
    if m > 0:
        # Combine positive and negative harmonics as per the formula
        return (1 / np.sqrt(2)) * (sph_harm(m, n, phi, theta) + (-1) ** m * sph_harm(-m, n, phi, theta)).real
    elif m < 0:
        # Combine positive and negative harmonics as per the formula
        return (1 / np.sqrt(2)) * (sph_harm(-m, n, phi, theta) - (-1) ** m * sph_harm(m, n, phi, theta)).imag
    else:  # m == 0
        return sph_harm(0, n, phi, theta).real


def compute_sh_coeff_pseudoinv(hrtf, theta, phi, Nmax):
    """
    Compute spherical harmonic coefficients using least squares.
    """
    S = len(theta)  # Number of discrete measurement points
    shd_len = (Nmax + 1) ** 2  # Total number of SH coefficients

    # Compute spherical harmonics matrix Y
    Y = np.zeros((S, shd_len))
    counter = 0
    for n in range(Nmax + 1):
        for m in range(-n, n + 1):
            # Y[:, counter] = sph_harm(m, n, phi, theta)
            Y[:, counter] = sph_harm_real(m, n, phi, theta)
            counter += 1

    # Solve for coefficients using least squares
    Y_T_Y = np.dot(Y.T, Y)
    Y_T_f = np.dot(Y.T, hrtf)
    coeffs = np.linalg.solve(Y_T_Y, Y_T_f)  # Solve the linear system

    return coeffs

def reconstruct_hrtf_from_coeffs(coeffs, theta, phi, Nmax):
    """
    Reconstruct HRTF values using spherical harmonics coefficients.
    """
    P = len(theta)  # Number of points to reconstruct
    shd_len = (Nmax + 1) ** 2  # Total number of SH coefficients

    # Compute spherical harmonics matrix Y
    Y = np.zeros((P, shd_len))
    counter = 0
    for n in range(Nmax + 1):
        for m in range(-n, n + 1):
            # Y[:, counter] = sph_harm(m, n, phi, theta)
            Y[:, counter] = sph_harm_real(m, n, phi, theta)

            counter += 1

    # Reconstruct HRTF
    hrtf_recon = np.dot(Y, coeffs)  # Multiply Y and c
    return hrtf_recon  # Take magnitude if needed

def coeff_allband(hrtf_magnitude, theta_rad, phi_rad, Nmax):
    coeffs_pseudoinv = []
    bin_range = hrtf_magnitude.shape[1]

    for Fbin in range(bin_range):  # Loop over frequency bins
        coeffs = compute_sh_coeff_pseudoinv(hrtf_magnitude[:, Fbin], theta_rad, phi_rad, Nmax)
        coeffs_pseudoinv.append(coeffs)
    coeffs_pseudoinv = np.array(coeffs_pseudoinv)
    return(coeffs_pseudoinv)

def reconstruction_allband(coeffs, theta, phi, Nmax):
    reconstructed_hrtf_all_bins = []
    for Fbin, coeffs in enumerate(sh):
        reconstructed_hrtf = reconstruct_hrtf_from_coeffs(coeffs, theta_rad, phi_rad, Nmax)
        reconstructed_hrtf_all_bins.append(reconstructed_hrtf)
    reconstructed_hrtf_all_bins = np.array(reconstructed_hrtf_all_bins).T  # Shape: (793, 129)
    return(reconstructed_hrtf_all_bins)

def get_spectral_distortion(hrtf_ground_truth: torch.Tensor, hrtf_predicted: torch.Tensor, sd, weights) -> torch.Tensor:
    elevation_index = sd.position

    azimuths = elevation_index[:, 0]
    elevations = elevation_index[:, 1]

    # Define the elevation range
    elevation_min = -30
    elevation_max = 30
    # Find the indices for the specific elevation range
    elevation_indic = np.where((elevations >= elevation_min) & (elevations <= elevation_max))[0]
    elevation_indices = np.array(elevation_indic, dtype=int)

    """
    Computes the spectral distortion between the inputs.

    Args:
        hrtf_ground_truth: torch.tensor
        hrtf_predicted: torch.tensor
    Returns:
        weighted_error: torch.tensor in dB

    """

    weighted_error = ((torch.from_numpy(weights) * (hrtf_ground_truth[elevation_indices].abs() - hrtf_predicted[elevation_indices].abs())) ** 2).mean()
    return weighted_error.log10() * 10


# Define dimensions
# M = 793  # Number of measurements (directions)
# R = 2    # Number of receivers (ears)
# N = 129  # Frequency bins or time samples
# sampling_rate = 48000  # Sampling rate

# Example reduced HRTF data (replace this with actual reduced HRTF data)
# hrtf_reduced = np.random.rand(M, R, N)  # Shape: (M, R, N)

# Elevation and azimuth angles (in radians)
# theta = np.random.uniform(0, np.pi, M)  # Elevation angles
# phi = np.random.uniform(0, 2 * np.pi, M)  # Azimuth angles

# Specify output path
# output_path = "./r_hrtf.sofa"

# sofa = sf.Sofa('SimpleFreeFieldHRIR')
# sofa.inspect()

# sofa.Data_IR = hrtf_reduced

sonicom_root = './data'
sd = SonicomDatabase(sonicom_root, training_data=True, folder_structure='v2')
phi_sonicom_deg, teta_sonicom_deg, r = sd.position[:, 0], sd.position[:, 1], sd.position[:, 2]
teta_scipy_deg = 90 - teta_sonicom_deg  # Adjust to scipy convention
phi_scipy_deg = phi_sonicom_deg

theta_rad = np.deg2rad(teta_scipy_deg)  # Elevation angles in radians
phi_rad = np.deg2rad(phi_scipy_deg)  # Azimuth angles in radians

train_dataloader = DataLoader(sd, batch_size=1, shuffle=False)

Nmax = 7
nfft=256
# Load data and extract HRTF
for i, (images, hrtf) in tqdm.tqdm(enumerate(train_dataloader)):
    if i == 1:
        break

    print(f'Image size: {images.shape} and HRTF size: {hrtf.shape}')

    hrir_original = np.zeros((1, 793, 2, nfft))
    hrir_recon = np.zeros_like(hrir_original)
    hrtf_recon = np.zeros_like(hrtf.numpy())

    for ear in range(2): # Loop over 2 ears

        # Convert HRTF to HRIR using nfft=256
        hrtf_personX = hrtf[0, :, ear, :].numpy()  # Shape: 793x129 for this ear
        hrir_original[0, :, ear, :] = convert_to_HRIR_from_hrtf(hrtf_personX)  # Shape: 1x793x2x256

        # Compute magnitude of HRTF
        hrtf_personX_magnitude = np.abs(hrtf_personX)

        # Reconstruct HRTF using SH coefficients
        sh = coeff_allband(hrtf_personX, theta_rad, phi_rad, Nmax)
        hrtf_recon_channel = reconstruction_allband(sh, theta_rad, phi_rad, Nmax)

        # Convert back to hrir
        hrtf_recon[0, :, ear, :] = hrtf_recon_channel
        hrir_recon[0, :, ear, :] = convert_to_HRIR_from_hrtf(hrtf_recon_channel)

# Select a single direction's HRIR for both left (channel 0) and right (channel 1)
hrir_stereo = hrir_recon[0, 0, :, :]  # Shape: (2, 256)
hrir_original_ = hrir_original[0, 0, :, :]
# Normalize the stereo HRIR
hrir_stereo_normalized = hrir_stereo / np.max(np.abs(hrir_original_))  # Normalize to [-1, 1]

# Convert to PCM format
hrir_stereo_pcm = (hrir_stereo_normalized * 32767).astype(np.int16).T  # Transpose to shape (256, 2)
hrir_original_pcm = (hrir_original_ * 32767).astype(np.int16).T
# Save to a stereo WAV file

# output_path = "hrir_rec_n7_abss.wav"
# write(output_path, 48000, hrir_stereo_pcm)
# print(f"Saved stereo HRIR to {output_path}")

# hrtf_rec_torch = torch.from_numpy(hrtf_recon)
hrtf_rec_torch = torch.as_tensor(hrtf_recon)

def get_weights():
    frequencies_Hz = np.linspace(0, 24000, 129)  # 129 points between 0 Hz and 24 kHz
    frequencies_kHz = frequencies_Hz / 1000
    inv_cb = 1 / (25 + 75 * (1 + 1.4 * frequencies_kHz**2) ** 0.69)  # inverse of delta (critical bandwidth)
    a0 = sum(inv_cb)
    weights = inv_cb / a0
    return(weights)
weights = get_weights()

print(get_spectral_distortion(hrtf[0], hrtf_rec_torch[0], sd, weights))

# metric = MeanSpectralDistortion()
# print(metric.get_spectral_distortion(hrtf, hrtf_rec_torch, sd))

for image_batch, hrtf_batch in tqdm.tqdm(train_dataloader):
    for _, ground_truth_hrtf in zip(image_batch, hrtf_batch):
        y = 45
        break
    break

AVG_HRTF_PATH = join(dirname(abspath(__file__)), 'data', 'Average_HRTFs.sofa')
pred_ave = sofar.read_sofa(AVG_HRTF_PATH, verbose=False)
predicted_hrtf = torch.as_tensor(sd._compute_HRTF(pred_ave.Data_IR))
print(get_spectral_distortion(predicted_hrtf, ground_truth_hrtf, sd, weights))

