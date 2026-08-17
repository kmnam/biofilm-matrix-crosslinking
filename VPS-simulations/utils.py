"""
Utility functions. 

Authors:
    Kee-Myoung Nam

Last updated:
    5/30/2026
"""
import numpy as np

########################################################################
def periodic_dist_vec(p, q, xmin, xmax, ymin, ymax, zmin, zmax):
    """
    Get the periodic distance vector between two points in a periodic 
    domain with the given fundamental unit cell. 

    Parameters
    ----------
    p : `numpy.ndarray`
        First input point. 
    q : `numpy.ndarray`
        Second input point. 
    xmin, xmax : float
        x-bounds of fundamental unit cell.
    ymin, ymax : float
        y-bounds of fundamental unit cell. 
    zmin, zmax : float
        z-bounds of fundamental unit cell.
       
    Returns
    -------
    Periodic distance vector between `p` and `q`.
    """
    delta = p - q
    dims = np.array([xmax - xmin, ymax - ymin, zmax - zmin])

    return delta - dims * np.round(delta / dims)

########################################################################
def wrap_coords(coords, xmin, xmax, ymin, ymax, zmin, zmax):
    """
    Wrap the coordinates into the fundamental unit cell.

    The coordinates may be of one or multiple polymer chains.

    Parameters
    ----------
    coords : `numpy.ndarray`
        Coordinate array. May be 2-D, in which case it should contain the
        coordinates of one chain and have shape (N, 3), where N is the
        chain length; or 3-D, in which case it should contain the coordinates
        of multiple same-length chains and have shape (K, N, 3), where K 
        is the number of chains and N is the chain length.
    xmin, xmax : float
        x-bounds of fundamental unit cell.
    ymin, ymax : float
        y-bounds of fundamental unit cell. 
    zmin, zmax : float
        z-bounds of fundamental unit cell.

    Returns
    -------
    Wrapped coordinate array. 
    """
    mins = np.array([xmin, ymin, zmin])
    dims = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
    flags = np.floor((coords - mins) / dims)

    return coords - flags * dims

########################################################################
def map_coords_to_cell(coords, xmin, xmax, ymin, ymax, zmin, zmax, image_flags):
    """
    Map the given polymer coordinates into the given cells, as indicated by
    the given image flags.

    The polymer coordinates are first wrapped into the fundamental unit cell, 
    then mapped into the desired cells. 

    Parameters
    ----------
    coords : `numpy.ndarray`
        Coordinate array. May be 2-D, in which case it should contain the
        coordinates of one chain and have shape (N, 3), where N is the
        chain length; or 3-D, in which case it should contain the coordinates
        of multiple same-length chains and have shape (K, N, 3), where K 
        is the number of chains and N is the chain length.
    xmin, xmax : float
        x-bounds of fundamental unit cell.
    ymin, ymax : float
        y-bounds of fundamental unit cell. 
    zmin, zmax : float
        z-bounds of fundamental unit cell.
    image_flags : `numpy.ndarray`
        Array of image flags. Should have the same shape as `coords`.

    Returns
    -------
    Array of mapped coordinates into the specified cells in the periodic
    domain.  
    """
    dims = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
    coords_wrapped = wrap_coords(coords, xmin, xmax, ymin, ymax, zmin, zmax)

    return coords_wrapped + image_flags * dims

########################################################################
def get_contiguous_image_flags_com(coords, xmin, xmax, ymin, ymax, zmin, zmax):
    """
    Obtain an array of image flags for a polymer chain such that:

    1) the center of mass is in the fundamental unit cell, and  
    2) the coordinates are contiguous.

    The coordinates should be of one polymer.

    Parameters
    ----------
    coords : `numpy.ndarray`
        Coordinate array. Should have shape (N, 3), where N is the chain 
        length. 
    xmin, xmax : float
        x-bounds of fundamental unit cell.
    ymin, ymax : float
        y-bounds of fundamental unit cell. 
    zmin, zmax : float
        z-bounds of fundamental unit cell.

    Returns
    -------
    Array of image flags for each atom in the polymer. 
    """
    mins = np.array([xmin, ymin, zmin])
    dims = np.array([xmax - xmin, ymax - ymin, zmax - zmin])

    # Calculate the center of mass and map all coordinates such that the 
    # center of mass lies in the fundamental unit cell 
    center_of_mass = coords.mean(axis=0)
    center_delta = np.floor((center_of_mass - mins) / dims)
    coords_wrapped = coords - dims * center_delta

    # Get the corresponding image flags
    return np.floor((coords_wrapped - mins) / dims)

########################################################################
def get_contiguous_image_flags_mindist(coords1, coords2, image_flags1, anchor1,
                                       anchor2, xmin, xmax, ymin, ymax, zmin,
                                       zmax):
    """
    Given the coordinates and image flags for one polymer chain, and the 
    coordinates for a second polymer chain, obtain an array of image flags
    for the second chain such that:

    1) the distance between the two anchor points is minimized in the plot,
       and 
    2) the coordinates of the chain are contiguous.

    The two coordinate arrays should each be of one polymer.

    Parameters
    ----------
    coords1 : `numpy.ndarray`
        Coordinate array for first polymer. Should have shape (N, 3), where
        N is the chain length.
    coords2 : `numpy.ndarray`
        Coordinate array for second polymer. Should have shape (N, 3), where
        N is the chain length. 
    image_flags1 : `numpy.ndarray`
        Image flag array for first polymer. Should have shape (N, 3), where 
        N is the chain length.
    anchor1 : int
        Index of anchor atom in first polymer. 
    anchor2 : int
        Index of anchor atom in second polymer. The image flag of this atom
        is determined such that its distance to the other anchor (given its
        image flag) is minimized. 
    xmin, xmax : float
        x-bounds of fundamental unit cell.
    ymin, ymax : float
        y-bounds of fundamental unit cell. 
    zmin, zmax : float
        z-bounds of fundamental unit cell.

    Returns
    -------
    Array of image flags for each atom in the polymer. 
    """
    coords1_mapped = map_coords_to_cell(
        coords1, xmin, xmax, ymin, ymax, zmin, zmax, image_flags1
    )
    
    # Get the image flag for the anchor in the second chain, which should 
    # be plotted such that the distance to the anchor in the first chain 
    # is minimized
    mins = np.array([xmin, ymin, zmin])
    dims = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
    relative_anchor2_flags = np.round(
        (coords2[anchor2, :] - coords1_mapped[anchor1, :]) / dims
    )
    coords2_wrapped = coords2 - dims * relative_anchor2_flags

    # Get the corresponding absolute image flags
    return np.floor((coords2_wrapped - mins) / dims)

########################################################################
def parse_configurations(filename):
    """
    Parse the polymer configurations in the given file.

    This file is assumed to contain configurations for a single polymer. 

    Parameters
    ----------
    filename : str
        Input filename.

    Returns
    -------
    3-D coordinate array containing each polymer configuration, together 
    with their total energies and radii of gyration. 
    """
    # First parse the first configuration, to get the length of the polymer
    with open(filename) as f:
        length = 0       # Get the length of the polymer  
        for line in f:
            if line.startswith('# CONFIG\t0'):
                break
            if not line.startswith('#'):   # Skip over additional annotations
                length += 1

    # Then parse the entire file and extract the coordinates
    coords = []
    energies = []
    radii = []
    with open(filename) as f:
        # Skip to the 0-th configuration
        line = f.readline()
        while not line.startswith('# CONFIG\t0'):
            line = f.readline()
        config_coords = np.zeros((length, 3), dtype=np.float64)
        i = 0
        for line in f:
            # If a new configuration is encountered, append the current 
            # configuration and reset 
            if line.startswith('# CONFIG'):
                coords.append(config_coords)
                config_coords = np.zeros((length, 3), dtype=np.float64)
                i = 0
            # Otherwise, keep track of the coordinates in the line 
            else:
                if line.startswith('# ENERGY_TOTAL'):
                    energies.append(float(line.split()[-1]))
                elif line.startswith('# RADIUS_OF_GYRATION'):
                    radii.append(float(line.split()[-1]))
                elif not line.startswith('#'):
                    config_coords[i, :] = [float(x) for x in line.strip().split()]
                    i += 1

    # Return the entire set of coordinates as an array
    return np.array(coords), np.array(energies), np.array(radii)

########################################################################
def parse_melt_configurations(filename):
    """
    Parse the polymer melt configurations in the given file.

    Parameters
    ----------
    filename : str
        Input filename.

    Returns
    -------
    4-D coordinate array containing each polymer melt configuration, together 
    with their total energies. 
    """
    # First parse the first configuration, to get the number of chains and 
    # the length of each chain
    n_chains = 0
    length = 0  
    with open(filename) as f:
        prev_line = f.readline()
        next_line = None
        while True:
            next_line = f.readline()
            if next_line.startswith('# CONFIG\t0'):
                break
            prev_line = next_line
        data = prev_line.strip().split('\t')
        n_chains = int(data[0]) + 1
        length = int(data[1]) + 1

    # Then parse the entire file and extract the coordinates
    coords = []
    energies = []
    with open(filename) as f:
        # Skip to the 0-th configuration
        line = f.readline()
        while not line.startswith('# CONFIG\t0'):
            line = f.readline()
        config_coords = np.zeros((n_chains, length, 3), dtype=np.float64)
        for line in f:
            # If a new configuration is encountered, append the current 
            # configuration and reset 
            if line.startswith('# CONFIG'):
                coords.append(config_coords)
                config_coords = np.zeros((n_chains, length, 3), dtype=np.float64)
            # Otherwise, keep track of the coordinates in the line 
            else:
                if line.startswith('# ENERGY_TOTAL'):
                    energies.append(float(line.split()[-1]))
                elif not line.startswith('#'):
                    data = line.strip().split()
                    i = int(data[0])
                    j = int(data[1])
                    rx = float(data[2])
                    ry = float(data[3])
                    rz = float(data[4])
                    config_coords[i, j, :] = [rx, ry, rz]

    # Return the entire set of coordinates as an array
    return np.array(coords), np.array(energies)

########################################################################
def parse_primitive_paths(filename):
    """
    Parse the primitive paths in the given file.

    The returned data include:
    - The coordinates of the nodes along each primitive path, each as a 
      list of arrays (since each path has a distinct number of nodes). 
    - The contour indices of the nodes along each primitive path, i.e., the
      approximate (floating-point) index of the bead corresponding to each 
      node.
    - The partner chain and partner node indices for the nodes along each 
      primitive path.

    The coordinates of the k-th node in the j-th path in the i-th
    configuration are in paths[i][j][k, :]. 

    The contour index of the k-th node in the j-th path in the i-th 
    configuration is contours[i][j][k].

    The partner chain and partner node index of the k-th node in the j-th 
    path in the i-th configuration are partners[i][j][k][0] and
    partners[i][j][k][1], respectively.

    All indices in the latter two lists are 0-indexed (unlike in the Z1+ 
    output, which is 1-indexed). 

    Parameters
    ----------
    filename : str
        Input filename.

    Returns
    -------
    Primitive path data, as described above.  
    """
    paths = []
    contours = []
    partners = []

    with open(filename) as f:
        # The first line contains the number of chains 
        line = f.readline()
        n_paths = int(line.strip())

        # The second line contains the box dimensions
        line = f.readline()
        xlen, ylen, zlen = [float(x) for x in line.strip().split()]

        # Set up a new list of primitive paths
        curr_paths = []
        curr_contours = []
        curr_partners = []
        curr_path = []
        curr_contour = []
        curr_partner = []

        # Run through the rest of the file ...
        for line in f:
            # If we have reached a new path or melt configuration ...
            data = line.strip().split()
            if len(data) == 1:
                # If we have reached a new path in the current configuration ...
                if len(curr_paths) < n_paths - 1 and len(curr_path) > 0:
                    curr_paths.append(np.array(curr_path))
                    curr_contours.append(curr_contour)
                    curr_partners.append(curr_partner)
                    curr_path = []
                    curr_contour = []
                    curr_partner = []
                # If we have reached a new melt configuration ...
                elif len(curr_paths) == n_paths - 1 and len(curr_path) > 0: 
                    # Collect the last path in the current melt configuration
                    curr_paths.append(np.array(curr_path))
                    curr_contours.append(curr_contour)
                    curr_partners.append(curr_partner)
                    curr_path = []
                    curr_contour = []
                    curr_partner = []
                    # Prepare to parse the next melt configuration
                    n_chains = int(data[0])
                    paths.append(curr_paths)
                    contours.append(curr_contours)
                    partners.append(curr_partners)
                    curr_paths = []
                    curr_contours = []
                    curr_partners = []
                    line = f.readline()     # Skip over the next line, which contains box dimensions
            # If we have reached a new node in the primitive path ...
            else:
                rx = float(data[0])
                ry = float(data[1])
                rz = float(data[2])
                contour_idx = float(data[3]) - 1
                partner_idx = (-1 if int(data[4]) == 0 else int(data[5]) - 1)
                partner_node = (-1 if int(data[4]) == 0 else int(data[6]) - 1)
                curr_path.append([rx, ry, rz])
                curr_contour.append(contour_idx)
                curr_partner.append([partner_idx, partner_node])

    return paths, contours, partners

