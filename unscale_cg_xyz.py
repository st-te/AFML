import numpy as np
from pathlib import Path

#################
a = 0
#################

def read_XYZ(filepath):
    filepath = Path(filepath)
    xyz = None
    if filepath.suffix == ".xyz" or filepath.suffix == ".XYZ":
        first_line = open(filepath).readline().strip()
        second_line = open(filepath).readline(2).strip()
        xyz = np.genfromtxt(filepath, skip_header=2, usecols=(2,3,4,5))
        mask = [row[0] in [1, 2, 3, 4] for row in xyz]
        xyz = [list(row) for row, keep in zip(xyz, mask) if keep]
        xyz.sort(key=lambda row: row[0])  # Sort based on the first column
        xyz = np.array(xyz)  # Convert to a NumPy array
    return xyz[:,0], xyz[:,1], xyz[:,2], xyz[:,3], first_line, second_line

# need to rewrite this w/ new read format
def unscale_XYZ(path='', write=True):

    path = Path(path)
    xyz = read_XYZ(path)
    xyz['particle'] = xyz['particle'].apply(lambda i: particle_1 if i == 1 else particle_2)
    xyz[['x']] = xyz[['x']].apply(lambda i: round(i * a, 5))
    xyz[['y']] = xyz[['y']].apply(lambda i: round(i * a, 5))
    xyz[['z']] = xyz[['z']].apply(lambda i: round(i * a, 5))
    particle_tot = xyz.shape[0]
    print(xyz)
    print(particle_tot)

    min_x, min_y, min_z = xyz['x'].min(), xyz['y'].min(), xyz['z'].min()
    max_x, max_y, max_z = xyz['x'].max(), xyz['y'].max(), xyz['z'].max()
    dx, dy, dz = m.ceil(max_x - min_x), m.ceil(max_y - min_y), m.ceil(max_z - min_z)
    print('min:', min_x, min_y, min_z, 'max:', max_x, max_y, max_z, 'cell:', dx, dy, dz)

    #xyz[['x']] = xyz[['x']].apply(lambda i: round(i - min_x, 5))
    #xyz[['y']] = xyz[['y']].apply(lambda i: round(i - min_y, 5))
    #xyz[['z']] = xyz[['z']].apply(lambda i: round(i - min_z, 5))


    min_x2, min_y2, min_z2 = xyz['x'].min(), xyz['y'].min(), xyz['z'].min()
    max_x2, max_y2, max_z2 = xyz['x'].max(), xyz['y'].max(), xyz['z'].max()
    dx2, dy2, dz2 = m.ceil(max_x2 - min_x2), m.ceil(max_y2 - min_y2), m.ceil(max_z2 - min_z2)
    print('min:', min_x2, min_y2, min_z2, 'max:', max_x2, max_y2, max_z2, 'cell:', dx2, dy2, dz2)

    #min_x_set_idx = xyz['x'].idxmin()
    #min_y_set_idx = xyz['y'].idxmin()
    #min_z_set_idx = xyz['z'].idxmin()

    #max_x_set_idx = xyz['x'].idxmax()
    #max_y_set_idx = xyz['y'].idxmax()
    #max_z_set_idx = xyz['z'].idxmax()

    #min_x_set = xyz.iloc[[min_x_set_idx]]
    #min_y_set = xyz.iloc[[min_y_set_idx]]
    #min_z_set = xyz.iloc[[min_z_set_idx]]

    #max_x_set = xyz.iloc[[max_x_set_idx]]
    #max_y_set = xyz.iloc[[max_y_set_idx]]
    #max_z_set = xyz.iloc[[max_z_set_idx]]

    #print(min_x_set)
    #print(min_y_set)
    #print(min_z_set)

    #print(max_x_set)
    #print(max_y_set)
    #print(max_z_set)

    if write: 
        final_path = path.parents[0] / f'{path.stem}_unscaled.xyz'
        with open(final_path, 'a') as f:
            f.write(f'{particle_tot}\n')
            f.write(f'CELL: {dx2}, {dy2}, {dz2}\n')
            xyz.to_csv(f, header=False, sep=' ', index=False)
    return(xyz)

 

def XYZ_out(path):
    path = Path(path)
    for xyz_file in path.rglob("*.XYZ"):
        unscale_XYZ(xyz_file)
        
XYZ_out('/Volumes/S_Tendyra/lno/perfect_bulk/unscaled_ideals')

