import os, time
import numpy as np
from scipy.spatial import Delaunay
from joblib import Parallel, delayed

from .utils import setup_logging

import logging
logger = logging.getLogger(__name__)

class BuildVoronoi:
    def __init__(self, i_SnapData, verbose=True, n_jobs = 8):
        self.verbose = verbose
        setup_logging(self.verbose)
        self.snap = i_SnapData
        self.pos = self.snap.pos
        self.id = self.snap.id
        self.phi = self.snap.pos_phi
        self.vor_path = self.snap._file_config.paths.vor_path
        self.n_jobs = self._configure_parallel(n_jobs, i_SnapData.num_particles)
        
        self.num_of_chunks = max(1, 2 ** int(np.round(np.log2(self.snap.num_particles / 5_000_000))))
        self.phi_mod = (self.phi + np.pi) % (2 * np.pi)
        self.phi_ranges = np.linspace(0, 2*np.pi, self.num_of_chunks + 1) 
        self.file_phi_ranges = np.linspace(-np.pi, np.pi, self.num_of_chunks + 1) 
        
    def _configure_parallel(self, n_jobs, num_particles):
        import psutil
        available_ram_gb = psutil.virtual_memory().available / 1e9
        
        # Estimate: ~60 GB RAM per 1.8e8 particles per job (adjust from your tests)
        estimated_ram_per_job = (num_particles / 1.8e8) * 60
        safe_jobs = int(available_ram_gb / estimated_ram_per_job)
        safe_jobs = max(1, min(safe_jobs, psutil.cpu_count()))
        
        if n_jobs is None:
            n_jobs = safe_jobs
            
        if n_jobs > safe_jobs and self.verbose:
            logger.warning(f"   ⚠   n_jobs={n_jobs} may exceed available RAM ({available_ram_gb:.0f} GB available)")
            logger.warning(f"   Recommended: n_jobs={safe_jobs} for {num_particles:,} particles")
            logger.warning(f"   Required RAM: ~{n_jobs * estimated_ram_per_job:.0f} GB")

        if available_ram_gb < estimated_ram_per_job:
            raise MemoryError(
                f"Insufficient RAM for {num_particles:,} particles.\n"
                f"Required: ~{estimated_ram_per_job:.0f} GB minimum (for n_jobs=1)\n"
                f"Available: {available_ram_gb:.0f} GB\n"
                f"Recommendation: Use cluster/HPC environment or reduce dataset size.")

        if self.verbose and n_jobs != safe_jobs:
            logger.info(f"   Using n_jobs={n_jobs} (recommended: {safe_jobs})")
        
        return n_jobs
        
    def get_tasks(self, pad=0.005):
        tasks = []
        two_pi = 2 * np.pi
        for i in range(self.num_of_chunks):
            phi_start, phi_end = self.phi_ranges[i:i+2]            
            mask_use = (self.phi_mod >= phi_start) & (self.phi_mod < phi_end)
            mask_phi = (self.phi_mod >= phi_start - pad) & (self.phi_mod < phi_end + pad)
            
            if phi_start - pad < 0:
                mask_phi |= self.phi_mod >= (two_pi + phi_start - pad)
            if phi_end + pad > two_pi:
                mask_phi |= self.phi_mod < (phi_end + pad - two_pi)
            tasks.append([self.id[mask_use], self.id[mask_phi], self.pos[mask_phi]])        
        return tasks
    
    @staticmethod
    def extract_ridge_pairs(gas_pos_filtered):   
        tri = Delaunay(gas_pos_filtered)
        simplices = tri.simplices
        edges = np.vstack([simplices[:, [0, 1]],
                           simplices[:, [1, 2]],
                           simplices[:, [2, 0]],
                           simplices[:, [0, 3]],
                           simplices[:, [1, 3]],
                           simplices[:, [2, 3]]])
        edges.sort(axis=1)                
        edges_view = np.ascontiguousarray(edges).view(
            np.dtype((np.void, edges.dtype.itemsize * edges.shape[1])))
        _, unique_idx = np.unique(edges_view, return_index=True)
        ridge_pairs = edges[unique_idx]
        return ridge_pairs
    
    @staticmethod
    def process_one(i, gas_ids_to_use, gas_ids_filtered, gas_pos_filtered, file_phi_ranges, snap_num, 
                    vor_save_path, replace = False):
        phi_start_i = file_phi_ranges[i]
        phi_end_i = file_phi_ranges[i+1]
        vor_save_name = f"{snap_num}_{i}_{np.round(phi_start_i / np.pi, 2)}_{np.round(phi_end_i / np.pi, 2)}_sec_vor.npz"
        save_path = os.path.join(vor_save_path, vor_save_name)
        if not replace:
            if os.path.exists(save_path):
                return
        ridge_pairs = BuildVoronoi.extract_ridge_pairs(gas_pos_filtered)
        np.savez_compressed(save_path,
                            gas_ids=gas_ids_filtered, 
                            vor_ridge_points=ridge_pairs, 
                            gas_ids_use=gas_ids_to_use)
        return
    
    def voronoi(self):
        if self.verbose:
            
            print(f'Building Voronoi diagram: {self.num_of_chunks} chunks, ' f'{len(self.id):,} particles')
        t_start = time.time()
        gas_tasks = self.get_tasks()
        Parallel(n_jobs = self.n_jobs)(delayed(BuildVoronoi.process_one)(i, *gas_tasks[i], self.file_phi_ranges, self.snap.snap_num, 
                                                  self.vor_path) for i in range(len(gas_tasks)))
        if self.verbose:
            print(f"Voronoi complete: {time.time() - t_start:.1f}s")



