## CrystalGrower

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read a .xyz
def read_XYZ(filepath):
    filepath = Path(filepath)
    xyz = None
    if filepath.suffix == ".xyz" or filepath.suffix == ".XYZ":
        xyz = np.genfromtxt(filepath, skip_header=2, usecols=(2,3,4,5))
        mask = [row[0] in [1, 2, 3, 4] for row in xyz]
        xyz = [list(row) for row, keep in zip(xyz, mask) if keep]
        xyz.sort(key=lambda row: row[0])  # Sort based on the first column
        xyz = np.array(xyz)  # Convert to a NumPy array
    #return xyz[:,0], xyz[:,1], xyz[:,2], xyz[:,3]
    return xyz

# Cut out a segment of n x n centred on (x,y) and normalise
def cut_out(filepath, center_x, center_y, threshold):
    xyz = read_XYZ(filepath)

    # Calculate the bounds for the filtering box
    x_min = center_x - threshold / 2
    x_max = center_x + threshold / 2
    y_min = center_y - threshold / 2
    y_max = center_y + threshold / 2

    # Create masks to filter rows based on the box bounds
    x_mask = (x_min < xyz[:, 1]) & (xyz[:, 1] <= x_max)
    y_mask = (y_min < xyz[:, 2]) & (xyz[:, 2] <= y_max)

    # Apply the masks to filter rows
    xyz_filtered = xyz[x_mask & y_mask]

    # Normalise
    norm = np.linalg.norm(xyz_filtered)
    xyz_filtered = xyz_filtered/norm

    return xyz_filtered

# Plot the facet in 3D
def plot_facet_3d(filtered_xyz):
    
    colour, x, y, z = [filtered_xyz[:, i] for i in range(4)]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colormap = plt.get_cmap("viridis")

    plt.rcParams["figure.figsize"] = [3.0, 3.0]
    plt.rcParams["figure.autolayout"] = True
    
    # Normalize the 'colour' column to map it to the colormap
    norm = plt.Normalize(colour.min(), colour.max())

    scatter = ax.scatter(x, y, z, c=colour, cmap=colormap, norm=norm, s=6)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Color Label")
    plt.tight_layout()

    plt.show()

# Plot the facet in 2D
def plot_facet_2d(filtered_xyz):
    
    colour, x, y, z = [filtered_xyz[:, i] for i in range(4)]
    plt.figure()
    colormap = plt.get_cmap("viridis")  # You can choose other colormaps as well

    plt.rcParams["figure.figsize"] = [3.0, 3.0]
    plt.rcParams["figure.autolayout"] = True

    # Normalize the 'colour' column to map it to the colormap
    norm = plt.Normalize(colour.min(), colour.max())

    scatter = plt.scatter(x, y, c=colour, cmap=colormap, norm=norm, s=15)

    cbar = plt.colorbar(scatter)
    cbar.set_label("Color Label")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("2D Scatter Plot in Z Projection")
    plt.tight_layout()

    plt.show()


## AFM

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, MeanShift
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# Read AFM xyz
def read_AFM_XYZ(filepath):
    filepath = Path(filepath)
    xyz = None
    if filepath.suffix == ".xyz" or filepath.suffix == ".XYZ":
        xyz = np.genfromtxt(filepath, skip_header=0, usecols=(0, 1, 2))
        
        # Create a mask to filter rows based on the specified conditions
        mask = [(row[0] >= 0) and (row[1] >= 0) for row in xyz]
        xyz = [list(row) for row, keep in zip(xyz, mask) if keep]
        xyz = np.array(xyz)  # Convert to a NumPy array
        norm = np.linalg.norm(xyz)
        xyz = xyz/norm
        # Sort based on the first column (x)
        xyz = xyz[xyz[:, 0].argsort()]
    return xyz

# Mean shift of z-levels
def mean_shift(point_cloud, bandwidth):
    # Create a Mean Shift clustering model
    ms = MeanShift(bandwidth=bandwidth)
    
    # Fit the model to the z-values of the point cloud
    z_values = point_cloud[:, 2].reshape(-1, 1)
    ms.fit(z_values)
    # Determine the cluster centers (modal z-values)
    modal_values = ms.cluster_centers_.flatten()

    # Round the z-values based on the modal z-values
    rounded_points = []
    for point in point_cloud:
        closest_modal_z = min(modal_values, key=lambda z: abs(z - point[2]))
        rounded_point = [point[0], point[1], closest_modal_z]
        rounded_points.append(rounded_point)

    return np.array(rounded_points)

# Down shift of z-levels
def down_shift(point_cloud, bandwidth):
    # Create a Mean Shift clustering model
    ms = MeanShift(bandwidth=bandwidth)
    
    # Fit the model to the z-values of the point cloud
    z_values = point_cloud[:, 2].reshape(-1, 1)
    ms.fit(z_values)

    # Determine the cluster centers (modal z-values)
    modal_values = ms.cluster_centers_.flatten()

    # Initialize a list to store the down-shifted points
    down_shifted_points = []

    for point in point_cloud:
        closest_modal_z = min(modal_values, key=lambda z: abs(z - point[2]))
        down_shifted_point = [point[0], point[1], closest_modal_z]
        down_shifted_points.append(down_shifted_point)

    return np.array(down_shifted_points)

# Terrace smoothing
def smooth_terraces(point_cloud, smoothing_sigma):
    # Sort the point cloud based on the z-values (terrace levels)
    sorted_point_cloud = point_cloud[np.argsort(point_cloud[:, 2])]

    # Initialize an empty array to store smoothed points
    smoothed_points = []

    # Iterate through each unique terrace level
    unique_z_levels = np.unique(sorted_point_cloud[:, 2])
    for z_level in unique_z_levels:
        # Extract the points for the current terrace level
        terrace_points = sorted_point_cloud[sorted_point_cloud[:, 2] == z_level]

        # Apply Gaussian smoothing to the x and y coordinates of the terrace points
        smoothed_x = gaussian_filter(terrace_points[:, 0], smoothing_sigma)
        smoothed_y = gaussian_filter(terrace_points[:, 1], smoothing_sigma)

        # Create a new array with smoothed x, y, and original z coordinates
        smoothed_terrace = np.column_stack((smoothed_x, smoothed_y, np.full_like(smoothed_x, z_level)))

        # Append the smoothed terrace to the result
        smoothed_points.append(smoothed_terrace)

    # Concatenate all smoothed terraces into a single array
    smoothed_point_cloud = np.vstack(smoothed_points)

    return smoothed_point_cloud

# Plot AFM in 3D
def plot_afm_3d(filtered_xyz):
    
    x, y, z = [filtered_xyz[:, i] for i in range(3)]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colormap = plt.get_cmap("viridis")

    plt.rcParams["figure.figsize"] = [4.0, 4.0]
    plt.rcParams["figure.autolayout"] = True
    
    # Normalize the 'colour' column to map it to the colormap
    norm = plt.Normalize(z.min(), z.max())

    scatter = ax.scatter(x, y, z, c=z, norm=norm, cmap=colormap, marker=",", s=1.0)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Color Label")
    
    ax.view_init(45, 60)

    plt.tight_layout()

    plt.show()

# Plot AFM in 2d
def plot_afm_2d(filtered_xyz):
    x, y, z = [filtered_xyz[:, i] for i in range(3)]
    plt.figure()
    colormap = plt.get_cmap("viridis")  # You can choose other colormaps as well
    plt.rcParams["figure.figsize"] = [3.0, 3.0]
    plt.rcParams["figure.autolayout"] = True
    # Normalize the 'colour' column to map it to the colormap
    norm = plt.Normalize(z.min(), z.max())

    scatter = plt.scatter(x, y, c=z, cmap=colormap, norm=norm, s=15)

    cbar = plt.colorbar(scatter)
    cbar.set_label("Color Label")

    plt.tight_layout()

    plt.show()

# Unique Z-values
def unique_z(afm):
    unique_z_values = set(point[2] for point in afm)
    return len(unique_z_values), unique_z_values

# Order terraces
def order_z(point_cloud):
    # Extract the Z-values and their unique sorted order
    z_values = np.array(point_cloud)[:, 3]
    sorted_order = np.argsort(z_values)
    
    # Create a mapping from original Z-values to their order
    z_to_order_mapping = {z: i + 1 for i, z in enumerate(np.unique(z_values))}
    
    # Update the first column based on the order
    ordered_indices = [z_to_order_mapping[z] for z in z_values]
    ordered_point_cloud = np.array(point_cloud)
    ordered_point_cloud[:, 0] = ordered_indices

    # Sort the data first by the index (first column), and then by the Z-value (last column)
    sorted_point_cloud = ordered_point_cloud[ordered_point_cloud[:, 0].argsort(), :]
    return sorted_point_cloud

# Interpolate downsample
def interpolate_downsample(afm_data, num_points_xy, method):
    """
    Perform downsampling with interpolation in X and Y dimensions while preserving Z dimension.

    Parameters:
    - afm_data: The input AFM 3D point cloud data as a NumPy array (X, Y, Z).
    - num_points_x: The desired number of points in the X dimension for the structured grid.
    - num_points_y: The desired number of points in the Y dimension for the structured grid.

    Returns:
    - downsampled_afm_data: The downsampled AFM data on the structured grid.
    - interpolated_xyz_point_cloud: The interpolated XYZ point cloud.
    """

    # Separate X, Y, and Z components of the AFM data
    x_points, y_points, z_points = afm_data.T

    # Calculate the min and max values along each axis
    x_min, x_max = min(x_points), max(x_points)
    y_min, y_max = min(y_points), max(y_points)

    # Create a structured grid with downsampling in X and Y
    x_grid, y_grid = np.meshgrid(
        np.linspace(x_min, x_max, num_points_xy),
        np.linspace(y_min, y_max, num_points_xy),
        indexing='ij'
    )

    # Perform interpolation for X and Y dimensions while preserving Z values
    downsampled_afm_data = griddata(
        (x_points, y_points),
        z_points,
        (x_grid, y_grid),
        method=method,
        fill_value=np.nan
    )

    # Convert the structured grid to XYZ point cloud format
    x_interpolated, y_interpolated = x_grid.ravel(), y_grid.ravel()
    z_interpolated = downsampled_afm_data.ravel()
    interpolated_xyz_point_cloud = np.column_stack((x_interpolated, y_interpolated, z_interpolated))

    return interpolated_xyz_point_cloud

