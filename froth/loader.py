import h5py, time, os, sys
import numpy as np
from typing import Optional, Union, List, Tuple
from .utils import load_masked_snapshot, build_adj, setup_logging

from .file_manager import PathsConfig, FileManager
import time

import logging
logger = logging.getLogger(__name__)

class SnapData:
    def __init__(self, snap_num: int, config_file: Optional[str] = None, 
                 data: Optional[dict] = None,
                 load_options: Optional[dict] = None,
                 neighbor_pairs: Union[np.ndarray, List[Tuple[int, int]], None] = None,
                 voronoi_distance_threshold: float = 0.03,
                 voronoi_overlap_padding: float = 0.005, 
                 voronoi_n_jobs: int = 8,
                 rebuild_voronoi: bool = False,
                 verbose: bool = True,
                 **path_kwargs): 
        """
        Initialize snapshot data.

        Parameters
        ----------
        snap_num : int
            Snapshot number (e.g., for 'snap_539.hdf5' use 539), or any identifier if providing custom data
        config_file : str
            Path to YAML config file specifying snapshot folder, working directories, and output paths
        data : dict, optional
            Pre-loaded snapshot data. Must include keys: 'position', 'density', 'temperature', 'mass', 'id'
              - 'position'    : (N, 3) array, units: kpc
              - 'density'     : (N,)   array, units: g/cm^3
              - 'temperature' : (N,)   array, units: K
              - 'mass'        : (N,)   array, units: 1 Msun
              - 'id'          : (N,)   array, int
            If provided, skips automatic snapshot loading
        neighbor_pairs : array-like of shape (n_neighbors, 2), optional
            Pre-computed neighboring structure as pairs of indices into the input data.
            Each row contains indices [i, j] of neighboring elements.
            If None, neighbors are computed from Voronoi tessellation.
            Default: None
        load_options : dict, optional
            Options for loading Arepo snapshots (ignored if `data` is provided):
            - snap_path (str): Path to snapshot file
            - temperature (bool): Compute temperature (required for temperature-aware models)
            - time (bool): Load snapshot time
            - z_limit (float): Vertical range limit
            - R_min (float): Minimum radial distance
            - R_max (float): Maximum radial distance
            - pad (float): Boundary padding
        voronoi_distance_threshold : float, default=0.03
            Distance cutoff for Voronoi ridge connections
        voronoi_overlap_padding : float, default=0.005
            Padding size for chunk overlap regions in Voronoi construction
        voronoi_n_jobs : int, default=8
            Number of parallel jobs for Voronoi construction
        rebuild_voronoi : bool, default=False
            Force recomputation of Voronoi tessellation, overwriting existing files
        verbose : bool, default=True
            Show logging info
        path_kwargs : dict, optional
            Additional path configuration:
            - num_of_vor_file (int): Number of Voronoi chunks to load (default: all available)
            - inner_file_suffix (str): Suffix for inner bubble result files (default: 'inner')
        """
        self.verbose = verbose
        setup_logging(self.verbose)
        self.snap_num = snap_num
        self.rebuild_voronoi = rebuild_voronoi
        self.distance_threshold = voronoi_distance_threshold
        self.n_jobs = voronoi_n_jobs
        self.padding = voronoi_overlap_padding
        if config_file is not None:
            self.paths = PathsConfig.from_yaml(config_file)
            for key, value in path_kwargs.items():
                if hasattr(self.paths, key):
                    setattr(self.paths, key, value)
        else:
            raise ValueError("   ⚠   Must provide 'config_file'")
        
        self.paths.create_directories()
        self._file_config = FileManager(self.paths)
        
        default_load_options = {'temperature': True, 'time': True, 'z_limit': 1.0, 'R_min': 7.5, 'R_max': 9.5}
        self._load_kwargs = {**default_load_options, **(load_options or {})}
        
        if data is not None:
            required = {'position', 'density', 'temperature', 'mass', 'id'}
            missing = required - data.keys()
            if missing:
                raise ValueError(f"   ⚠   Missing required keys in data: {missing}")
            if (load_options or path_kwargs):
                logger.warning("   ⚠   Data provided — ignoring load_options and path_kwargs")
        
        self._snap_data = data
        
        self._pos_phi = None
        self._pos_R = None
        self._num_particles = None
        self._node_degrees = None
                    
        self._vor_files = self._file_config.vor_file(self.snap_num)
        
        self.no_vor_file = neighbor_pairs is not None
        
        if neighbor_pairs is not None:
            self._vor = np.array(neighbor_pairs).astype(np.int32)
        else:
            if not self._file_config._validate_vor_files(self._vor_files) or self.rebuild_voronoi :
                logger.info("Voronoi files missing or rebuild requested.")
                logger.info("░░░░░░░░░░░░░░ computing Voronoi... ░░░░░░░░░░░░░░")
                self.compute_voronoi()
                self._vor_files = self._file_config.vor_file(self.snap_num)
            self._vor = None
            
        self._adj = None
        self._confidence = None
        self._labels = None
        #self._vor = None

    @property
    def snap_data(self):
        if self._snap_data is None:
            snap_file = self._file_config.snapshot_file(self.snap_num)
            logger.info(f"-----> loading snapshot {self.snap_num}...")
            self._snap_data = load_masked_snapshot(snap_file, **self._load_kwargs)
            logger.info(f"   ✔   snapshot loaded")
        return self._snap_data
    
    @property
    def vor_files(self):
        if not self._file_config._validate_vor_files(self._vor_files):
            self._vor_files = self._file_config.vor_file(self.snap_num)
        return self._vor_files
    
    @property
    def pos(self):
        return self.snap_data['position']
    
    @property
    def den(self):
        return self.snap_data['density']
    
    @property
    def mass(self):
        return self.snap_data['mass']
    
    @property
    def tem(self):
        return self.snap_data['temperature']
    
    @property
    def id(self):
        return self.snap_data['id']
    
    @property
    def num_particles(self):
        if self._num_particles is None:
            self._num_particles = len(self.id)
        return self._num_particles
    
    @property
    def node_degrees(self):
        if self._node_degrees is None:
            self._node_degrees = self._adj.sum(axis=1).A1
        return self._node_degrees
    
    @property
    def adj(self):
        if self._adj is None:
            logger.info("-----> building adjacency matrix... ")
            self._adj = build_adj(self.vor, size = self.num_particles)
            logger.info("   ✔   adjacency matrix built.")
        return self._adj
    
    @property
    def pos_phi(self):
        if self._pos_phi is None:
            self._pos_phi = np.arctan2(self.pos[:, 1], self.pos[:, 0])
        return self._pos_phi
    
    @property
    def pos_R(self):
        if self._pos_R is None:
            self._pos_R = np.sqrt(self.pos[:, 0]**2 + self.pos[:, 1]**2)
        return self._pos_R
    
    def get_clip_overlaps(self):
        gas_phi = self.pos_phi
        phi_ranges = np.linspace(-np.pi, np.pi, 33)
        mask_overlap = np.zeros(self.num_particles, dtype=bool)
        for phi_i in phi_ranges:
            mask_pad = ((gas_phi - phi_i + self.padding) % (2*np.pi)) < ((self.padding * 2) % (2*np.pi))
            mask_overlap[mask_pad] = True
        return mask_overlap #np.flatnonzero(mask_overlap)
    
    def map_ids_to_indices(self):
        id_to_index = np.full(int(self.id.max()) + 1, -1, dtype=np.int64)
        id_to_index[self.id] = np.arange(self.num_particles)
        return id_to_index        
    
    def compute_voronoi(self):
        from .voronoi import BuildVoronoi
        vor_builder = BuildVoronoi(self, verbose = self.verbose, n_jobs = self.n_jobs, padding = self.padding, 
                                   replace = self.rebuild_voronoi)
        vor_builder.voronoi()
    
    @property
    def vor(self): 
        if self._vor is None:
            t0 = time.time()
            mask_overlap = self.get_clip_overlaps()
            vor_files = self.vor_files
            id_to_index = self.map_ids_to_indices()
            logger.info(f'-----> loading {len(vor_files)} Voronoi chunks ...')
            core_edges = []
            overlap_edges = []
            for vor_file_i in vor_files:
                vor_data = np.load(vor_file_i)
                gas_ids_sec = vor_data['gas_ids']
                pairs = vor_data['vor_ridge_points']
                pairs[:, 0] = id_to_index[gas_ids_sec[pairs[:, 0]]]
                pairs[:, 1] = id_to_index[gas_ids_sec[pairs[:, 1]]]
                delta = self.pos[pairs[:, 0]] - self.pos[pairs[:, 1]]
                sq_distances = np.sum(delta**2, axis=1)
                mask = sq_distances < self.distance_threshold**2
                
                mask_overlap_local = mask_overlap[pairs[mask, 0]] | mask_overlap[pairs[mask, 1]]
                overlap_edges.append(pairs[mask][mask_overlap_local])
                core_edges.append(pairs[mask][~mask_overlap_local])
            del id_to_index
            core_edges = np.concatenate(core_edges, axis=0)
            overlap_edges = np.concatenate(overlap_edges, axis=0)
            overlap_unique = np.unique(np.sort(overlap_edges, axis=1), axis=0)
            del overlap_edges
            self._vor = np.vstack([core_edges, overlap_unique]).astype(np.int32)
            del core_edges
            del overlap_unique
            logger.info(f'   ✔   Voronoi ridges loaded in {np.round(time.time() - t0)}s.')
        return self._vor
    
    def _set_inner_bubble(self, confidence, threshold):
        self._confidence = confidence
        self._labels = confidence > threshold
            
    @property
    def confidence(self):
        return self._confidence

    @property
    def labels(self):
        return self._labels

class ImageData:
    def __init__(self, data: np.ndarray, verbose: bool = True):
        setup_logging(verbose)
        
        self.raw = data.copy()
        self.data = data.copy()
        self._data_filtered = None
        
        self._valid_pixels = None
        self._neighbors = None
        self._neighbors_all = None
        self._adj = None
        self._node_degrees = None
        self._num_pixels = None
        self._sigmas = None
        self._n_sigmas = 1
    
    @classmethod
    def from_file(cls, path: str, region=None, verbose: bool = True):
        if region is not None and not isinstance(region, (tuple, slice)):
            raise TypeError("   ⚠   Region must be a numpy slice, e.g. np.s_[1120:1450, 620:950]")
        from astropy.io import fits
        data = fits.getdata(path)[region] if region is not None else fits.getdata(path)
        return cls(data, verbose=verbose)

    @classmethod
    def from_array(cls, data: np.ndarray, verbose: bool = True):
        if not isinstance(data, np.ndarray):
            raise TypeError("   ⚠   Image data must be a numpy array")
        return cls(data, verbose=verbose)
        
    @property
    def valid_pixels(self):
        if self._valid_pixels is None:
            iy, ix = np.nonzero(self.data > 0) 
            self._valid_pixels = (iy, ix)
        return self._valid_pixels
        
    @property
    def neighbors(self):
        if self._neighbors is None:
            from scipy.spatial import cKDTree
            logger.info(f"-----> loading valid pixel neighborhood information...")
            iy, ix = self.valid_pixels
            coords = np.column_stack([iy, ix])
            tree = cKDTree(coords)
            pairs = tree.query_pairs(r=np.sqrt(2) + 1e-10, output_type='ndarray')

            self._neighbors = pairs
            logger.info(f'   ✔    done')
        return self._neighbors
    
    @property
    def neighbors_all(self):
        if self._neighbors_all is None:
            from scipy.spatial import cKDTree
            logger.info(f"-----> loading all neighborhood information...")
            iy, ix = np.indices(self.data.shape)
            coords = np.column_stack([iy.ravel(), ix.ravel()])
            tree = cKDTree(coords)
            pairs = tree.query_pairs(r=np.sqrt(2) + 1e-10, output_type='ndarray')

            self._neighbors_all = pairs
            logger.info(f'   ✔    done')
        return self._neighbors_all
    
    @property
    def neighbors(self):
        if self._neighbors is None:
            from scipy.spatial import cKDTree
            logger.info(f"-----> loading all pixel neighborhood information...")
            iy, ix = self.valid_pixels
            coords = np.column_stack([iy, ix])
            tree = cKDTree(coords)
            pairs = tree.query_pairs(r=np.sqrt(2) + 1e-10, output_type='ndarray')

            self._neighbors = pairs
            logger.info(f'   ✔    done')
        return self._neighbors

    @property
    def num_pixels(self):
        if self._num_pixels is None:
            self._num_pixels = len(self.data.flatten())
        return self._num_pixels
    
    @property
    def node_degrees(self):
        if self._node_degrees is None:
            self._node_degrees = self._adj.sum(axis=1).A1
        return self._node_degrees
    
    @property
    def adj(self):
        if self._adj is None:
            logger.info("-----> building adjacency matrix... ")
            self._adj = build_adj(self.neighbors, size = self.num_pixels)
            logger.info("   ✔   adjacency matrix built.")
        return self._adj
    
    