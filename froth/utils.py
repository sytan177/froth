import numpy as np
import logging

def setup_logging(verbose: bool):
    if verbose:
        logging.getLogger().setLevel(logging.INFO)
        if not logging.getLogger().handlers:
            logging.getLogger().addHandler(logging.StreamHandler())

def build_adj(pairs, size = None):
    """
    Build symmetric adjacency matrix from edge pairs.
    
    Parameters
    ----------
    pairs : array-like of shape (n_edges, 2)
        Edge pairs as (source, target) indices
    size : int, optional
        Number of nodes. If None, inferred as max(pairs) + 1
    
    Returns
    -------
    adj : scipy.sparse.csr_matrix
        Symmetric adjacency matrix in CSR format
    """
    
    from scipy.sparse import csr_matrix
    edges_arr = np.array(pairs)
    if size is not None:
        n_nodes = size
    else:
        n_nodes = edges_arr.max() + 1
    row = edges_arr[:, 0]
    col = edges_arr[:, 1]
    data = np.ones(len(row), dtype=bool)
    adj = csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
    adj = adj + adj.T
    return adj

def dict_indexing(dist_data, indices):
    """Extract subset of dictionary values by indices."""
    return {key: val[indices] for key, val in dist_data.items()}

def load_masked_snapshot(snap_path, temperature = False, time = False, z_limit = 0.3, R_min = 7.5, R_max = 9.5, pad = 1e-4):
    from scida import load
    from astropy import units as u, constants as c
    snap_ds = load(snap_path, units = False)
    box_centre = np.array([0.5, 0.5, 0.5]) * snap_ds.boxsize
    UnitDensity_in_gpercm3 = snap_ds.header["UnitMass_in_g"] / (snap_ds.header["UnitLength_in_cm"] ** 3)
    UnitVelocity_in_km_per_s = snap_ds.header["UnitVelocity_in_cm_per_s"] * u.cm.to(u.km)
    UnitMass_in_Msun = snap_ds.header["UnitMass_in_g"] * u.g.to(u.Msun)
    UnitLength_in_kpc = snap_ds.header["UnitLength_in_cm"] * u.cm.to(u.kpc)
    
    snap_ds['PartType0']['position'] = snap_ds['PartType0']['Coordinates'] - box_centre
    gas_pos_all = snap_ds['PartType0']['position'] / 10
    gas_den_all = snap_ds['PartType0']['Density']
    gas_ids_all = snap_ds['PartType0']['ParticleIDs']
    gas_ch_ab_all = snap_ds['PartType0']['ChemicalAbundances']
    gas_inE_all = snap_ds['PartType0']['InternalEnergy']
    gas_mass_all = snap_ds['PartType0']['Masses']
    gas_vel_all = snap_ds['PartType0']['Velocities']
    gas_vol_all = gas_mass_all / gas_den_all
    
    R = gas_pos_all[:, 0]**2 + gas_pos_all[:, 1]**2
    mask = (R > R_min**2 - pad) & (R < R_max**2 + pad) & (np.abs(gas_pos_all[:, 2]) < z_limit  + pad)

    gas_pos = gas_pos_all[mask].compute()
    gas_den = (gas_den_all[mask].compute() * UnitDensity_in_gpercm3)
    gas_ids = gas_ids_all[mask].compute()
    gas_cha = gas_ch_ab_all[mask].compute()
    gas_inE = (gas_inE_all[mask].compute() * UnitVelocity_in_km_per_s**2)
    gas_mass = (gas_mass_all[mask].compute() * UnitMass_in_Msun)
    gas_vel = (gas_vel_all[mask].compute() * UnitVelocity_in_km_per_s)
    gas_vol = (gas_vol_all[mask].compute() * UnitLength_in_kpc**3)
    
    snap_gas = {}
    snap_gas['position'] = gas_pos
    snap_gas['density'] = gas_den
    snap_gas['id'] = gas_ids
    snap_gas['chemical_abundances'] = gas_cha
    snap_gas['internal_energy'] = gas_inE
    snap_gas['mass'] = gas_mass
    snap_gas['velocity'] = gas_vel
    snap_gas['volume'] = gas_vol
    
    if temperature:
        if "ABHE" in snap_ds.config:
            ABHE = snap_ds.config["ABHE"]
        else:
            ABHE = 0.1
        rho =  gas_den * (u.g / u.cm**3)
        n_H = rho / ((1.0 + 4.0 * ABHE) * c.m_p)
        ntot = (1.0 + ABHE - gas_cha[:, 0] + gas_cha[:, 1]) * n_H.to(1 / u.cm**3)

        snap_gas["n_H"] = n_H.to(1 / u.cm**3).value
        snap_gas["number_density"] = ntot.value
        
        internal_energy = gas_inE * (u.km / u.s) ** 2
        energy = internal_energy * rho
        tgas = (2.0 * energy / (3.0 * ntot * c.k_B)).to(u.K).value
        snap_gas["temperature"] = tgas
    if time:
        unit_length = 0.1 * u.kpc
        unit_velocity = u.km/u.s
        unit_time = (unit_length/unit_velocity).to(u.Myr)
        snap_time = snap_ds.header['Time'] * unit_time
        snap_gas['time'] = snap_time
    return snap_gas

def cylindrical_boundary_mask(pos_R, pos_z, R_limits=[7.5, 9.5], z_lims=[-1., 1.], padding=0.005):
    """ Boundary mask for disk galaxies (cylindrical simulation box) """
    return ((pos_R > R_limits[1] - padding) | (pos_R < R_limits[0] + padding) |
            (pos_z > z_lims[1] - padding) | (pos_z < z_lims[0] + padding))

def box_boundary_mask(pos, x_lims, y_lims, z_lims, padding=0.005):
    """ Boundary mask for coordinate box, not limiting to Cartesian """
    return ((pos[:, 0] < x_lims[0] + padding) | (pos[:, 0] > x_lims[1] - padding) |
            (pos[:, 1] < y_lims[0] + padding) | (pos[:, 1] > y_lims[1] - padding) |
            (pos[:, 2] < z_lims[0] + padding) | (pos[:, 2] > z_lims[1] - padding))