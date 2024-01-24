import numpy as np
from pathlib import Path

#CG
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist

#Mat-Eig
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R

#AFM
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, MeanShift
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


## Mat-Eig methodology

##THIS IS NOT WORKING
def rotate_point_cloud(points, miller_indices):
    """
    Rotate the point cloud so that the plane defined by the Miller indices (hkl)
    becomes parallel to the (001) plane.

    Parameters:
    - points: numpy array of points where columns are [label, x, y, z]
    - miller_indices: tuple of (h, k, l) defining the plane

    Returns:
    - A numpy array of the rotated points.
    """
    # Define the normal vector from the Miller indices
    h, k, l = miller_indices
    normal_vector = np.array([h, k, l])
    target_vector = np.array([0, 0, 1])  # z-axis

    # Calculate the rotation axis (cross product of normal vector and target vector)
    rotation_axis = np.cross(normal_vector, target_vector)
    # Calculate the angle of rotation (arccosine of the dot product of normalized vectors)
    angle = np.arccos(np.dot(normal_vector, target_vector) / (np.linalg.norm(normal_vector) * np.linalg.norm(target_vector)))

    # Calculate the rotation matrix using the axis-angle representation
    rotation_matrix = R.from_rotvec(rotation_axis * angle).as_matrix()

    # Apply the rotation to the point cloud
    rotated_points = np.dot(points[:, 1:4], rotation_matrix)

    # Combine the labels with the rotated coordinates
    rotated_point_cloud = np.hstack((points[:, [0]], rotated_points))

    return rotated_point_cloud




## CrystalGrower

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

# Removes specified z-levels
def remove_z_levels(point_cloud, z_levels_to_remove):
    # Convert the point cloud to a NumPy array if it's not already
    if not isinstance(point_cloud, np.ndarray):
        point_cloud = np.array(point_cloud)

    # Create a mask to filter rows based on Z-levels to remove
    mask = np.isin(point_cloud[:, 0], z_levels_to_remove, invert=True)

    # Apply the mask to filter rows and return the filtered point cloud
    filtered_point_cloud = point_cloud[mask]

    return filtered_point_cloud

# Cut out a segment of n x n (distance) centred on (x,y) and normalise
"""def cut_out(filepath, center_x, center_y, threshold):
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

    return xyz_filtered"""

# Cut out a segment of n x n (distance) with lower bounds (x_min, y_min) and normalise
def cut_out(xyz, x_min, y_min, threshold):
    # Calculate the bounds for the filtering box
    x_max = x_min + threshold
    y_max = y_min + threshold

    # Create masks to filter rows based on the box bounds
    x_mask = (x_min <= xyz[:, 0]) & (xyz[:, 0] <= x_max)
    y_mask = (y_min <= xyz[:, 1]) & (xyz[:, 1] <= y_max)

    # Apply the masks to filter rows
    xyz_filtered = xyz[x_mask & y_mask]

    if len(xyz_filtered) == 0:
        raise ValueError("No points found within the specified bounds.")

    # Normalise
    #norm = np.linalg.norm(xyz_filtered)
    #xyz_filtered = xyz_filtered / norm

    return xyz_filtered

# Voxelise a point cloud to remove atomic/crystallographic arrangement
def voxelization(points, resolution):
    # Extract columns
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    
    # Calculate voxel sizes based on resolution and the range of points in x and y
    x_voxel_size = (xs.max() - xs.min()) / resolution
    y_voxel_size = (ys.max() - ys.min()) / resolution
    
    # For voxelization, we'll floor-divide by voxel size and multiply again.
    # This will ensure points falling in the same voxel get the same coordinates.
    voxelized_xs = np.floor(xs / x_voxel_size) * x_voxel_size + x_voxel_size / 2
    voxelized_ys = np.floor(ys / y_voxel_size) * y_voxel_size + y_voxel_size / 2
    
    # Group by voxelized (X, Y) and Z, then calculate the mean for each group
    voxel_points = {}
    for x, y, z in zip(voxelized_xs, voxelized_ys, zs):
        if (x, y, z) not in voxel_points:
            voxel_points[(x, y, z)] = []
        voxel_points[(x, y, z)].append((x, y, z))
    
    # Now, for each voxel, we'll average the points (should be near identical due to voxelization)
    centroids = []
    for k, pts in voxel_points.items():
        avg_x = np.mean([p[0] for p in pts])
        avg_y = np.mean([p[1] for p in pts])
        centroids.append([avg_x, avg_y, k[2]])  # Z is preserved
        
    return np.array(centroids)

# Plot the facet in 3D
def plot_CG_3d(filtered_xyz):
    
    colour, x, y, z = [filtered_xyz[:, i] for i in range(4)]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colormap = plt.get_cmap("viridis")

    plt.rcParams["figure.figsize"] = [4.0, 4.0]
    plt.rcParams["figure.autolayout"] = True
    
    # Normalize the 'colour' column to map it to the colormap
    norm = plt.Normalize(colour.min(), colour.max())

    scatter = ax.scatter(x, y, z, c=colour, cmap=colormap, norm=norm, s=6)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Color Label")
    plt.tight_layout()

    plt.show()

# Plot the facet in 2D
def plot_CG_2d(filtered_xyz):
    
    colour, x, y, z = [filtered_xyz[:, i] for i in range(4)]
    plt.figure()
    colormap = plt.get_cmap("viridis")  # You can choose other colormaps as well

    plt.rcParams["figure.figsize"] = [4.0, 4.0]
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

# Normalize a point cloud
def normalize_PC(point_cloud):
    """
    Normalize a point cloud containing only XYZ coordinates.

    Args:
    - point_cloud (numpy.ndarray): An Nx3 NumPy array where N is the number of points,
    and each row contains [x, y, z] coordinates.

    Returns:
    - normalized_point_cloud (numpy.ndarray): The normalized point cloud.
    """
    norm = np.linalg.norm(point_cloud, axis=1, keepdims=True)
    normalized_point_cloud = point_cloud / norm

    return normalized_point_cloud

# remove overlapping voxels in point cloud at z+1 level
def adjacent_z_wipe(voxelized_points):
    """
    For voxels with the same (x, y) but adjacent z-levels, remove the one at the lower z-level.
    """
    
    # Create a dictionary to track voxel positions
    voxel_dict = {(x, y): z for x, y, z in voxelized_points}
    
    # Iterate over the voxelized_points to identify conflicts
    to_remove = set()
    for x, y, z in voxelized_points:
        if (x, y, z + 1) in voxel_dict:  # Check if there's a voxel directly above the current voxel
            to_remove.add((x, y, z))
    
    # Remove the identified conflicting points
    cleaned_points = np.array([point for point in voxelized_points if (point[0], point[1], point[2]) not in to_remove])
    
    return cleaned_points

# normalise a point cloud in range of points e.g. 32 x 32 0 -> 32
def scale_point_cloud(points, x_range=32, y_range=32):
    """
    Scale the given point cloud's x and y to the specified range. 
    The z-coordinates will be remapped to integer values from 1 to number of unique z-values.
    
    Parameters:
    - points (list of tuples or np.array): The point cloud in format [(x1,y1,z1), (x2,y2,z2), ...]
    - x_range (int): The maximum value for x after scaling.
    - y_range (int): The maximum value for y after scaling.
    
    Returns:
    - Scaled point cloud as a numpy array with shape (n, 3), where n is the number of points.
    """
    
    points = np.array(points)  # Convert to numpy array if not already
    
    # Compute the scaling factors and shifts for x and y
    x_shift = np.min(points[:, 0])
    y_shift = np.min(points[:, 1])
    x_scale = x_range / (np.max(points[:, 0]) - x_shift)
    y_scale = y_range / (np.max(points[:, 1]) - y_shift)
    
    # Scale and shift the x and y coordinates
    points[:, 0] = (points[:, 0] - x_shift) * x_scale
    points[:, 1] = (points[:, 1] - y_shift) * y_scale
    
    # Create a mapping from each unique z-value to an integer starting from 1
    unique_z_vals = np.unique(points[:, 2])
    z_mapping = {val: idx+1 for idx, val in enumerate(sorted(unique_z_vals))}
    
    # Replace z-values with their corresponding integer value
    for idx, row in enumerate(points):
        points[idx, 2] = z_mapping[row[2]]
    
    return points


## AFM


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

# UNUSED - Mean shift of z-levels
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

# UNUSED - Down shift of z-levels
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

# Plot point cloud in 3D
def plot_PC_3d(filtered_xyz, view1=45, view2=60):
    
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
    
    ax.view_init(view1, view2)

    plt.tight_layout()

    plt.show()

# Plot point cloud in 2d
def plot_PC_2d(filtered_xyz, size=4.0):
    x, y, z = [filtered_xyz[:, i] for i in range(3)]
    plt.figure()
    colormap = plt.get_cmap("viridis")  # You can choose other colormaps as well
    plt.rcParams["figure.figsize"] = [size, size]
    plt.rcParams["figure.autolayout"] = True
    # Normalize the 'colour' column to map it to the colormap
    norm = plt.Normalize(z.min(), z.max())

    scatter = plt.scatter(x, y, c=z, cmap=colormap, norm=norm, s=15)

    cbar = plt.colorbar(scatter)
    cbar.set_label("Color Label")

    plt.tight_layout()

    plt.show()


def plot_PC_2d_vers(filtered_xyz, size=4.0):
    x, y, z = [filtered_xyz[:, i] for i in range(3)]
    
    # Set figure size here rather than using rcParams to avoid global changes
    fig, ax = plt.subplots(figsize=(size, size))
    
    colormap = plt.get_cmap("viridis")  # You can choose other colormaps as well
    
    # Normalize the 'colour' column to map it to the colormap
    norm = plt.Normalize(z.min(), z.max())

    scatter = ax.scatter(x, y, c=z, cmap=colormap, norm=norm, s=15)

    #cbar = plt.colorbar(scatter)
    #cbar.set_label("Color Label")

    plt.tight_layout()

    ax.autoscale_view()

    plt.show()

# Unique Z-values
def unique_z(afm):
    unique_z_values = set(point[2] for point in afm)
    return len(unique_z_values), unique_z_values

# Point cloud dimensions
def extract_dimensions(point_cloud):
    """
    Extract dimensions (length, width, height) from a point cloud.

    Args:
    - point_cloud (numpy.ndarray): An Nx3 NumPy array where N is the number of points,
    and each row contains [x, y, z] coordinates.

    Returns:
    - dimensions (dict): A dictionary containing 'length', 'width', and 'height' dimensions.
    """
    if len(point_cloud) == 0:
        raise ValueError("Point cloud is empty")

    min_x, min_y, min_z = np.min(point_cloud, axis=0)
    max_x, max_y, max_z = np.max(point_cloud, axis=0)

    length = max_x - min_x
    width = max_y - min_y
    height = max_z - min_z

    dimensions = {
        'length': length,
        'width': width,
        'height': height
    }

    return dimensions

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

# Smooth terraces
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
