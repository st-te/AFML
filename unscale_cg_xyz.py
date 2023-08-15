import numpy as np
from pathlib import Path

def read_XYZ(filepath):
    filepath = Path(filepath)
    xyz = None
    first_line = ""
    second_line = ""
    if filepath.suffix == ".xyz" or filepath.suffix == ".XYZ":
        with open(str(filepath)) as file:
            first_line = file.readline().strip()
            second_line = file.readline(2).strip()

        xyz = np.genfromtxt(filepath, skip_header=2, usecols=(0, 1, 2, 3, 4, 5))
        mask = [row[0] in [1, 2, 3, 4] for row in xyz]
        xyz = [list(row) for row, keep in zip(xyz, mask) if keep]
        xyz.sort(key=lambda row: row[0])  # Sort based on the first column
        xyz = np.array(xyz)  # Convert to a NumPy array
    return (
        xyz[:, 0], xyz[:, 1], xyz[:, 2], xyz[:, 3], xyz[:, 4], xyz[:, 5],
        first_line, second_line
    )

def unscale_XYZ(a, path='', write=True):
    path = Path(path)
    col1, col2, col3, x, y, z, first_line, _ = read_XYZ(path)  # Ignore the original second line
    x_scaled = x * a
    y_scaled = y * a
    z_scaled = z * a

    if write:
        filename = path.name  # Get the filename without the path
        final_path = path.parents[0] / ('{}_unscaled.xyz'.format(path.stem))
        with open(str(final_path), 'w') as f:
            f.write('{}\n'.format(first_line))  # Preserve 3 spaces at the start
            f.write('{}\n'.format(filename))  # Use the filename as the second line
            for i in range(len(x_scaled)):
                formatted_line = '{:.0f} {:.0f} {:.0f} {:10.3f} {:10.3f} {:10.3f}\n'.format(
                    col1[i], col2[i], col3[i], x_scaled[i], y_scaled[i], z_scaled[i]
                )
                f.write(formatted_line)

    return x_scaled, y_scaled, z_scaled, col1, col2, col3

def XYZ_out(a, path):
    path = Path(path)
    extensions = ['.xyz', '.XYZ']
    
    for extension in extensions:
        for xyz_file in path.rglob('*' + extension):
            unscale_XYZ(a, xyz_file)

XYZ_out(12.93, '/Users/user/Documents/PhD/AFML/PAR_ETH_100/20230809_154357/XYZ_files')

