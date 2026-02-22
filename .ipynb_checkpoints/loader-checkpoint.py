import h5py, time, os, sys
import numpy as np
from utils import load_masked_snapshot
from file_manager import PathsConfig, FileManager

class LoadSnapshot:
    def __init__(self, snap_num, file_manager: FileManager, **load_kwargs):
        self.path = file_manager.snapshot_file(snap_num)
        self.load_kwargs = load_kwargs
        self._data = None
        
    @property
    def data(self):
        if self._data is None:
            self._data = load_masked_snapshot(self.path, **self.load_kwargs)
        return self._data
    
class SnapData:
    def __init__(self, snap_num, snap_data, file_manager: FileManager): 
        self.snap_num = snap_num
        self.snap_data = snap_data
        self._num_particles = None
        self._node_degrees = None
        self._adj = None

        self.pos = self.snap_data['position']
        self.den = self.snap_data['density']
        self.tem = self.snap_data['temperature']
        self.id = self.snap_data['id']
        self._confidence = None
        self._pos_phi = None
        self._pos_R = None
        self._labels = None
        self._vor = None
        
        self.file_config = file_manager
        self.vor_files = file_manager.vor_file(snap_num)
        self.inner_file = file_manager.bubble_file(self.snap_num, 'inner')
        
    @property
    def num_particles(self):
        if self._num_particles is None:
            self._num_particles = len(self.labels)
        return self._num_particles
    
    @property
    def node_degrees(self):
        if self._node_degrees is None:
            self._node_degrees = self._adj.sum(axis=1).A1
        return self._node_degrees
    
    @property
    def adj(self, test_mode = False):
        if self._adj is None:
            print("Building adjacency matrix...")
            if test_mode:
                self._adj = build_adj(self.vor, size = self.num_particles)
            else:
                self._adj = build_adj(self.vor, size = self.num_particles)
            print("Adjacency matrix built.")
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
    
    def get_clip_overlaps(self, padding = 0.005):
        gas_phi = self.pos_phi
        phi_ranges = np.linspace(-np.pi, np.pi, 33)
        mask_overlap = np.zeros(self.num_particles, dtype=bool)
        for phi_i in phi_ranges:
            mask_pad = ((gas_phi - phi_i + padding) % (2*np.pi)) < ((padding * 2) % (2*np.pi))
            mask_overlap[mask_pad] = True
        return mask_overlap #np.flatnonzero(mask_overlap)
    
    def map_ids_to_indices(self):
        id_to_index = np.full(int(self.id.max()) + 1, -1, dtype=np.int64)
        id_to_index[self.id] = np.arange(self.num_particles)
        return id_to_index
    
    @property
    def vor(self, distance_threshold=0.03): 
        if self._vor is None:
            #t_start = time.time()
            mask_overlap = self.get_clip_overlaps()
            vor_files = self.vor_files
            id_to_index = self.map_ids_to_indices()
            print(f'Loading {len(vor_files)} Voronoi files')

            core_edges = []
            overlap_edges = []

            for vor_file_i in vor_files[:]:
                vor_data = np.load(vor_file_i)
                gas_ids_sec = vor_data['gas_ids']
                pairs = vor_data['vor_ridge_points']
                pairs[:, 0] = id_to_index[gas_ids_sec[pairs[:, 0]]]
                pairs[:, 1] = id_to_index[gas_ids_sec[pairs[:, 1]]]
                delta = self.pos[pairs[:, 0]] - self.pos[pairs[:, 1]]
                sq_distances = np.sum(delta**2, axis=1)
                mask = sq_distances < distance_threshold**2
                
                mask_overlap_local = mask_overlap[pairs[mask, 0]] | mask_overlap[pairs[mask, 1]]
                overlap_edges.append(pairs[mask][mask_overlap_local])
                core_edges.append(pairs[mask][~mask_overlap_local])
                #print(np.round(time.time() - t_start, 2))
            del id_to_index
            core_edges = np.concatenate(core_edges, axis=0)
            overlap_edges = np.concatenate(overlap_edges, axis=0)
            overlap_unique = np.unique(np.sort(overlap_edges, axis=1), axis=0)
            del overlap_edges
            self._vor = np.vstack([core_edges, overlap_unique]).astype(np.int32)
            del core_edges
            del overlap_unique
            print(f'Voronoi ridges loaded.')
        return self._vor
        
    def _load_inner_data(self):
        if self._confidence is None or self._labels is None:
            with h5py.File(self.inner_file, 'r') as f:
                self._confidence = f['confidence'][()]
            self._labels = self._confidence > 0.5
            
    @property
    def confidence(self):
        if self._confidence is None:
            self._load_inner_data()
        return self._confidence

    @property
    def labels(self):
        if self._labels is None:
            self._load_inner_data()
        return self._labels