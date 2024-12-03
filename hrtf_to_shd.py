import numpy as np
from scipy.special import sph_harm
import matplotlib
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import tqdm
from utils import SonicomDatabase
import pickle
matplotlib.use('Qt5Agg')  # For PyQt5 or PySide2

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
def sph_harm_real(m, n, phi, theta):

    if m > 0:
        return np.sqrt(2) * (-1) ** m * np.real(sph_harm(m, n, phi, theta))
    elif m < 0:
        return np.sqrt(2) * (-1) ** m * np.imag(sph_harm(-m, n, phi, theta))
    else:  # m == 0
        return sph_harm(0, n, phi, theta).real
# Helper Functions

def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


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

def compute_weighted_error(hrtf_true, hrtf_predicted):
    """
    Compute weighted error in dB between true and predicted HRTF magnitudes.
    """

    weighted_error = np.mean((np.abs(hrtf_true) - np.abs(hrtf_predicted)) ** 2)
    return 10. * np.log10(weighted_error)  # Return error in dB

# Main Code
sonicom_root = './data'
sd = SonicomDatabase(sonicom_root, training_data=True, folder_structure='v2')
train_dataloader = DataLoader(sd, batch_size=1, shuffle=False)

# Load data and extract HRTF
for i, (images, hrtf) in tqdm.tqdm(enumerate(train_dataloader)):
    print(f'Image size: {images.shape} and HRTF size: {hrtf.shape}')


    if i == 1:
        break

# Extract HRTF and positional data
Fbin_ = 10
hrtf_demo = hrtf[0, :, 0, :].numpy()  # 793, 129
hrtf_demo_magnitude = np.abs(hrtf_demo)

# hrtf_normalized = hrtf / np.max(np.abs(hrtf))
# Take the magnitude (abs) of the HRTF
# hrtf_demo_magnitude = np.abs(hrtf_demo)

# hrtf_magnitude = np.abs(hrtf_demo[:, Fbin_])  # Only use magnitude for decomposition -- wrong
# hrtf_magnitude_normalized = hrtf_magnitude / np.max(hrtf_magnitude)  # Normalize to [0, 1]

phi_sonicom_deg, teta_sonicom_deg, r = sd.position[:, 0], sd.position[:, 1], sd.position[:, 2]
teta_scipy_deg = 90 - teta_sonicom_deg  # Adjust to scipy convention
phi_scipy_deg = phi_sonicom_deg

# SHD decomposition with
Nmax = 7  # Maximum SH order
theta_rad = np.deg2rad(teta_scipy_deg)  # Elevation angles in radians
phi_rad = np.deg2rad(phi_scipy_deg)  # Azimuth angles in radians

# Method 1: SHD Decomposition with Direct Method
# coeffs_direct, Ynm_direct = shd_decomposition_direct(hrtf_magnitude_normalized, phi_scipy_deg, teta_scipy_deg, Nmax)
# Reconstruction using Method 1
# reconstructed_hrtf_direct = np.dot(Ynm_direct.T, coeffs_direct.T).T  # Shape: (F, P)
# hrtf_recon_direct_normalized = reconstructed_hrtf_direct / np.max(reconstructed_hrtf_direct)

# Method 2: SHD Decomposition with Pseudoinverse
coeffs_pseudoinv_ = compute_sh_coeff_pseudoinv(hrtf_demo[:, Fbin_], theta_rad, phi_rad, Nmax)

# Reconstruction using Method 2
reconstructed_hrtf_pseudoinv = reconstruct_hrtf_from_coeffs(coeffs_pseudoinv_, theta_rad, phi_rad, Nmax)

# Visualization of original and reconstructed HRTFs
x, y, z = spherical_to_cartesian(1.5, theta_rad, phi_rad)

# Original HRTF plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z, c=hrtf_demo_magnitude[:, Fbin_], cmap='viridis', s=20)
plt.colorbar(sc, ax=ax, label="HRTF Values")
ax.set_title("Original HRTF Magnitude")
# plt.show()

# Reconstructed HRTF plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z, c=np.abs(reconstructed_hrtf_pseudoinv), cmap='viridis', s=20)
plt.colorbar(sc, ax=ax, label="HRTF Values")
ax.set_title("Reconstructed HRTF Magnitude (SHD)")
# plt.show()

# Initialize parameters
bin_range = hrtf_demo.shape[1]
selected_bins = range(bin_range)  # Use all frequency bins for comparison
N_values = [7]  # Example values of Nmax for comparison
errors_by_N = {}

# Loop through different values of Nmax
for Nmax in N_values:
    print(f"Processing for Nmax = {Nmax}...")
    # Compute SH coefficients for all bins
    coeffs_pseudoinv = []
    for Fbin in range(bin_range):  # Loop over frequency bins
        coeffs = compute_sh_coeff_pseudoinv(hrtf_demo[:, Fbin], theta_rad, phi_rad, Nmax)
        coeffs_pseudoinv.append(coeffs)
    coeffs_pseudoinv = np.array(coeffs_pseudoinv)  # Shape: (129, (Nmax+1)^2)

    # Reconstruct HRTFs for all bins
    reconstructed_hrtf_all_bins = []
    for Fbin, coeffs in enumerate(coeffs_pseudoinv):
        reconstructed_hrtf = reconstruct_hrtf_from_coeffs(coeffs, theta_rad, phi_rad, Nmax)
        reconstructed_hrtf_all_bins.append(reconstructed_hrtf)
    reconstructed_hrtf_all_bins = np.array(reconstructed_hrtf_all_bins).T  # Shape: (793, 129)

    # Compute weighted errors for specific bins
    errors_db = []
    for Fbin in selected_bins:
        error = compute_weighted_error(hrtf_demo[:, Fbin], reconstructed_hrtf_all_bins[:, Fbin])
        errors_db.append(error)

    # Store errors for the current Nmax
    errors_by_N[Nmax] = errors_db

# Plot errors for all N values across frequency bins
plt.figure(figsize=(12, 8))
for Nmax, errors_db in errors_by_N.items():
    plt.plot(selected_bins, errors_db, label=f"Nmax = {Nmax}")

plt.title("Reconstruction Error Across Frequency Bins for Different Nmax")
plt.xlabel("Frequency Bin")
plt.ylabel("MSE Error (dB)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(np.abs(reconstructed_hrtf_all_bins[:400, 30]), label="Reconstructed HRTF")
plt.plot(hrtf_demo_magnitude[:400, 30], label="Original HRTF")
plt.legend()  # Add legend
plt.title("Comparison of Original and Reconstructed HRTF")  # Optional title
plt.xlabel("Frequency Bin")  # Optional x-axis label
plt.ylabel("Magnitude")  # Optional y-axis label
plt.show()

plt.figure()
plt.plot(np.angle(hrtf_demo_magnitude[:, 30]), label="Original HRTF Phase")
plt.plot(np.angle(reconstructed_hrtf_all_bins[:, 30]), label="Reconstructed HRTF Phase")
plt.legend()
plt.title("Phase Comparison")
plt.xlabel("Frequency Bin")
plt.ylabel("Phase (Radians)")
plt.grid()
plt.show()

