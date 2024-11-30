import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Convert spherical to Cartesian for plotting
def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def spherical_harmonic_decomposition(hrtf, theta, phi, freq_bins, max_order):
    """
    Decompose HRTF into spherical harmonics.

    Parameters:
    - hrtf: numpy array of shape (num_points, num_freq_bins)
    - theta: array of elevation angles (num_points)
    - phi: array of azimuth angles (num_points)
    - freq_bins: array of frequency bins
    - max_order: maximum order for spherical harmonics

    Returns:
    - coeffs: dictionary with keys (n, m) containing arrays of coefficients per frequency bin
    """
    num_points, num_freq_bins = hrtf.shape
    coeffs = {}

    for n in range(max_order + 1):
        for m in range(-n, n + 1):
            coeff = np.zeros(num_freq_bins, dtype=complex)
            for i in range(num_points):
                Y_nm = sph_harm(m, n, phi[i], theta[i])
                coeff += hrtf[i, :] * Y_nm * np.sin(theta[i])  # Weight by sin(theta)
            coeff *= 4 * np.pi / num_points  # Normalize by total sampling points
            coeffs[(n, m)] = coeff

    return coeffs

# Reconstruct the HRTF on the structured grid
def reconstruct_hrtf_grid(coeffs, theta, phi, max_order, num_freq_bins):
    """
    Reconstruct the HRTF at specific sampling points for all frequency bins.

    Parameters:
    - coeffs: dictionary of spherical harmonic coefficients
    - theta: 1D array of elevation angles (793 points)
    - phi: 1D array of azimuth angles (793 points)
    - max_order: maximum spherical harmonic order
    - num_freq_bins: number of frequency bins

    Returns:
    - reconstructed: 2D array of shape (793, num_freq_bins)
    """
    num_points = len(theta)
    reconstructed = np.zeros((num_points, num_freq_bins), dtype=complex)

    for n in range(max_order + 1):
        for m in range(-n, n + 1):
            Y_nm = sph_harm(m, n, phi, theta)  # Shape: (793,)
            # Broadcast and add: coeffs[(n, m)] has shape (num_freq_bins,)
            reconstructed += Y_nm[:, None] * coeffs[(n, m)][None, :]

    return np.real(reconstructed)  # Use real part for visualization

# Example parameters
num_points = 793
num_freq_bins = 129
max_order = 20  # Choose based on desired accuracy
freq_bin_idx = 20

theta = np.linspace(0, np.pi, num_points)  # Example elevation angles
phi = np.linspace(0, 2 * np.pi, num_points)  # Example azimuth angles
hrtf = np.random.rand(num_points, num_freq_bins)  # Example HRTF data
freq_bins = np.linspace(0, 8000, num_freq_bins)  # Example frequency bins (0-8 kHz)

# Perform decomposition
coeffs = spherical_harmonic_decomposition(hrtf, theta, phi, freq_bins, max_order)

# Access coefficients for specific (n, m)
n, m = 2, 1
print(f"Spherical harmonic coefficients for order {n}, degree {m}: {coeffs[(n, m)]}")


# Generate structured grid for the sphere
phi_grid, theta_grid = np.meshgrid(
    np.linspace(0, 2 * np.pi, num_points),  # Azimuth angles
    np.linspace(0, np.pi, num_points)        # Elevation angles
)




# Reconstruct on the grid
reconstructed_grid = reconstruct_hrtf_grid(coeffs, theta, phi, max_order, num_freq_bins)

# Convert spherical to Cartesian for grid
x_grid = np.sin(theta_grid) * np.cos(phi_grid)
y_grid = np.sin(theta_grid) * np.sin(phi_grid)
z_grid = np.cos(theta_grid)

# Plot HRTF on the sphere
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(
    x_grid, y_grid, z_grid, facecolors=plt.cm.viridis(reconstructed_grid),
    rstride=1, cstride=1, antialiased=False, shade=False
)
ax.set_title('HRTF on Sphere')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax, shrink=0.5, aspect=10)
plt.show()

# Plot HRTF on the sphere
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(
    x_grid, y_grid, z_grid, facecolors=plt.cm.viridis(hrtf),
    rstride=1, cstride=1, antialiased=False, shade=False
)
ax.set_title('HRTF on Sphere')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax, shrink=0.5, aspect=10)
plt.show()

