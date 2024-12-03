import numpy as np
from scipy.special import sph_harm
from torch.utils.data import DataLoader
import tqdm
from utils import SonicomDatabase
import os
import sofar
import pickle
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
    Y = np.zeros((S, shd_len), dtype=np.complex128)
    counter = 0
    for n in range(Nmax + 1):
        for m in range(-n, n + 1):
            Y[:, counter] = sph_harm_real(m, n, phi, theta)
            counter += 1

    # Solve for coefficients using least squares
    Y_T_Y = np.dot(Y.T, Y)
    Y_T_f = np.dot(Y.T, hrtf)
    coeffs = np.linalg.solve(Y_T_Y, Y_T_f)  # pseudoinv

    return coeffs

def construct_reduced_hrtf_with_pseudoinv(hrtf, theta_rad, phi_rad, Nmax):
    """
    Constructs a reduced HRTF tensor using spherical harmonic decomposition with pseudoinverse.

    Parameters:
        hrtf (numpy.ndarray): Input HRTF tensor of shape (1, 793, 2, 129).
                              - 1: Batch size (unused)
                              - 793: Number of measurement points (positions)
                              - 2: Left and right ears
                              - 129: Frequency bins
        theta_deg (numpy.ndarray): Elevation angles in degrees, shape (793,).
        phi_deg (numpy.ndarray): Azimuth angles in degrees, shape (793,).
        Nmax (int): Maximum spherical harmonics order.

    Returns:
        numpy.ndarray: Reduced HRTF tensor of shape ((Nmax+1)^2, 2, 129).
    """
    # Extract dimensions
    num_positions, num_ears, num_bins = hrtf.shape
    shd_len = (Nmax + 1) ** 2  # Number of spherical harmonic coefficients

    # Initialize the reduced HRTF tensor
    hrtf_reduced = np.zeros((shd_len, num_ears, num_bins))

    # Iterate over ears (2 for left and right)
    for ear in range(num_ears):
        # Extract HRTF for this ear
        hrtf_ear = hrtf[:, ear, :]

        # Compute coefficients for each frequency bin
        for Fbin in range(num_bins):
            hrtf_bin = hrtf_ear[:, Fbin]  # HRTF values at all positions for this bin

            # Use the previously defined pseudoinverse function to compute coefficients
            coeffs = compute_sh_coeff_pseudoinv(hrtf_bin, theta_rad, phi_rad, Nmax)

            # Populate the reduced HRTF tensor
            hrtf_reduced[:, ear, Fbin] = coeffs

    return hrtf_reduced

# Main Code
sonicom_root = './data'
sd = SonicomDatabase(sonicom_root, training_data=False, folder_structure='v2')

# Load data and extract HRTF

def process_hrtfs(sd, Nmax):

    output_dir = f"./reduced_hrtf_N{Nmax}"
    os.makedirs(output_dir, exist_ok=True)

    train_dataloader = DataLoader(sd, batch_size=1, shuffle=False)
    phi_sonicom_deg = sd.position[:, 0]  # Azimuth angles in degrees (793,)
    teta_sonicom_deg = sd.position[:, 1]  # Elevation angles in degrees (793,)
    teta_scipy_deg = 90 - teta_sonicom_deg  # Convert to colatitudinal angles
    phi_scipy_deg = phi_sonicom_deg

    # Convert angles to radians
    theta_scipy_rad = np.deg2rad(teta_scipy_deg)  # Elevation angles
    phi_scipy_rad = np.deg2rad(phi_scipy_deg)  # Azimuth angles

    for i, (images, hrtf) in tqdm.tqdm(enumerate(train_dataloader)):
        # Get the subject's ID
        subject_name = sd.all_subjects[i]  # Subject name from the dataset
        print(f"\nProcessing Subject: {subject_name}")
        # print(f'Image size: {images.shape} and HRTF size: {hrtf.shape}')
        hrtf_dem_rl = hrtf[0, :, :, :].numpy()

        # Construct the reduced HRTF tensor
        hrtf_reduced = construct_reduced_hrtf_with_pseudoinv(hrtf_dem_rl, theta_scipy_rad, phi_scipy_rad, Nmax)

        # Construct SOFA file path
        filename = f"{subject_name}_N{Nmax}_SH"
        output_path = os.path.join(output_dir, filename)

        # Save to SOFA file
        dump(output_path, hrtf_reduced)

        # if i== 1:
        #     break

    return (hrtf_reduced)

hrtf_reduced = process_hrtfs(sd, Nmax=7)


# Output shape
print(f"Reduced HRTF tensor shape: {hrtf_reduced.shape}")  # Should print: ((Nmax+1)^2, 2, 129)


