import h5py, time, os, sys, joblib, json, hashlib
import numpy as np
from typing import List, Tuple, Optional, Literal
from dataclasses import dataclass
from numba import njit, types, prange
from numba.typed import Dict, List
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from joblib import Parallel, delayed
from math import ceil
from pathlib import Path

from .utils import build_adj, setup_logging
from .loader import SnapData, ImageData

import logging
logger = logging.getLogger(__name__)    

class BubbleExtractor:
    def __init__(self, data, 
                 force_recompute: bool = False, 
                 skip_shell: bool = False, 
                 boundary_mask = None,
                 inner_params=None, 
                 shell_params=None,
                 save_results: bool = True, 
                 verbose: bool = True):
        """
        Initialize bubble extractor.

        Parameters
        ----------
        data : SnapData or ImageData
            Input data object containing gas particles or image data
        force_recompute : bool, default=False
            Overwrite existing results if True
        skip_shell : bool, default=False
            Skip shell assignment step if True
        boundary_mask : np.ndarray, optional
            Boolean mask for boundary particles to exclude from open fraction calculations
        inner_params : dict, optional
            Inner bubble segmentation parameters:
            - peel_iteration (int): K-core decomposition iterations (default: 10)
            - threshold (float): Overlap fraction for merging groups (default: 0.2)
            - max_group_size (int): Max cells per group for merging (default: 1e6)
            - min_component_size (int) : Minimum component size
            - component_iteration_size_factor : int, default=0 (default: 5)
              Minimum component size increases with iterations: size_limit = min_component_size + factor * iteration.
              Set > 0 to progressively filter out small components during peeling.
            - single_connect (int): Min neighbors required when all neighbors belong to the same group
            - connect (int): Min neighbors required when neighbors belong to multiple different groups
            - chunk_size (int): Batch size for parallelization (default: 10000000)
            - n_jobs (int): Number of parallel jobs, -1 for all cores (default: -1)
        shell_params : dict, optional
            Shell assignment parameters:
            - inner_connect (int): Min inner neighbors to assign (default: 2)
            - connect (int): Min shell neighbors to assign (default: 3)
            - max_layer (int): Layers to expand for shells (default: 6)
            - max_iter (int): Max assignment iterations (default: 3)
            - pair_order (str): 'density' or 'density_temperature' for SnapData only (default: 'density')
            - chunk_size (int): Batch size for parallelization (default: 200000)
            - n_jobs (int): Number of parallel jobs (default: -1)
        save_results : bool, default=True
            Save extracted bubble groups to disk; avoids recomputation on subsequent runs
        """
        setup_logging(verbose)
        self._data = data
        if not isinstance(data, (SnapData, ImageData)):
            raise TypeError("   ⚠   Data must be a SnapData or ImageData instance")
        self.force_recompute = force_recompute
        self.skip_shell = skip_shell
        self.inner_params = inner_params or {}
        self.shell_params = shell_params or {}
        self.groups = InnerBubbleGroups(self._data, boundary_mask, **self.inner_params)
        self._shells = None 
        self.save_results = save_results
    
    @property
    def shells(self):
        if self._shells is None:
            self._shells = BubbleShellAssignment(self._data, **self.shell_params)
        return self._shells
    
    
    def _config_hash(self):
        def _exclude_keys(d, keys):
            return {k: v for k, v in d.items() if k not in keys}
        
        exclude = {'chunk_size', 'n_jobs'}
        config_dict = {'inner': _exclude_keys(self.inner_params, exclude)}
        if not self.skip_shell:
            config_dict['shell'] = _exclude_keys(self.shell_params, exclude)
    
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def _get_save_path(self):
        shell_tag = "" if not self.skip_shell else "_unshelled"
        suffix = f"_knocked{shell_tag}_{self._config_hash()}"
        save_pt = self._data.inner_results_path.with_stem(self._data.inner_results_path.stem + suffix)
        return save_pt
        
    def knock_snapshot(self):
        self.save_pt = self._get_save_path()
        if self.save_pt.exists() and not self.force_recompute:
            logger.info(f"◉  ◉  ◉ knocked ◉  ◉  ◉ \n knocked bubbles already exist at:\n  {self.save_pt.as_posix()}\n skipping computation ...")
            with h5py.File(self.save_pt, 'r') as f:
                inner_shell_labelled = f['groups'][:]
                shell_assigned = f.attrs.get('shell_assigned', False)
            logger.info(f"-----> loaded bubbles [shells: {'✓' if shell_assigned else '✗'}]")
            self._data.bubble_groups = inner_shell_labelled
            return
        
        self.groups.extract()
        self._data.inner_assigned = self.groups.assigned
        if not self.skip_shell:
            self.shells.assign()
            if self.save_results:
                self.save(shell_assigned=True)
        else:
            logger.info("-----> skipping shell assignment ...")
            self._data.bubble_groups = self._data.inner_assigned
            if self.save_results:
                self.save(shell_assigned=False)
    
    def save(self, shell_assigned: bool):
        logger.info('◉  ◉  ◉ knocked ◉  ◉  ◉ \n-----> saving the bubbles ... ')
        with h5py.File(self.save_pt, 'w') as f:
            f.create_dataset('labels', data=self._data.labels, compression='gzip', chunks=True)
            f.create_dataset('groups', data=self._data.bubble_groups, compression='gzip', chunks=True)
            f.attrs['description'] = "Labels (0/1) and group IDs (integers)"
            f.attrs['num_entries'] = self._data.labels.size
            f.attrs['derived_from'] = self._data.inner_results_path.as_posix()
            f.attrs['shell_assigned'] = shell_assigned
            f.attrs['inner_params'] = json.dumps(self.inner_params)
            f.attrs['shell_params'] = json.dumps(self.shell_params) if shell_assigned else ""
        logger.info(f"   ✔   saved [shells: {'✓' if shell_assigned else '✗'}]")
        
    def knock_image(self):
        self.groups.extract()
        self._data.inner_assigned = self.groups.assigned
        if not self.skip_shell:
            self.shells.assign()
        else:
            logger.info("-----> skipping shell assignment ...")
            self._data.bubble_groups = self._data.inner_assigned
        self._data.bubble_groups = self._data.bubble_groups.reshape(self._data.data.shape)
        
    
@dataclass
class BubbleMergeConfig:
    peel_iteration: int = 10
    threshold: float = 0.2
    max_group_size: int = 1_000_000
    min_component_size: int = 5
    component_iteration_size_factor: int = 0
    enforce_terminal_bijection: bool = False
    single_connect: int = 3
    connect: int = 4
    chunk_size: int = 1_000_000
    n_jobs: int = -1

class InnerBubbleGroups:
    def __init__(self, data, boundary_mask=None, **kwargs):
        self._data = data
        self.labels = self._data.labels
        if isinstance(self._data, SnapData):
            self._get_vor = lambda: data.vor
            self._get_size = lambda: data.num_particles
            self._get_cfd = lambda: data.confidence
            self._get_lineage = self.get_component_lineage
        elif isinstance(self._data, ImageData):
            self._get_vor = lambda: data.neighbors_all
            self._get_size = lambda: data.num_pixels
            self._get_cfd = lambda: data.probs
            self._get_lineage = self.get_image_component_lineage
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
            
        self._vor = None
        self._cfd = None
        self._size = None
        self._adj = None
        self._node_degrees = None
        self.config = BubbleMergeConfig(**kwargs)
        self._inner_adj = None
        self._boundary_mask = boundary_mask
        self._earliest_components = None
        self._invalid_components = None
        self._new_assignments = List.empty_list(types.int64)
        self.iter_labels = None
        
    @property
    def vor(self):
        if self._vor is None:
            self._vor = self._get_vor()
        return self._vor
    
    @property
    def size(self):
        if self._size is None:
            self._size = self._get_size()
        return self._size
    
    @property
    def cfd(self):
        if self._cfd is None:
            self._cfd = self._get_cfd()
        return self._cfd
    
    @property
    def adj(self):
        if self._adj is None:
            self._adj = self._data.adj
        return self._adj
    
    @property
    def node_degrees(self):
        if self._node_degrees is None:
            self._node_degrees = self._data.node_degrees
        return self._node_degrees
    
    @property
    def inner_adj(self):
        if self._inner_adj is None:
            a = self.vor[:, 0]
            b = self.vor[:, 1]
            inner_mask = self.labels[a] & self.labels[b]    
            bubble_inner_pairs = self.vor[inner_mask]
            self._inner_adj = build_adj(bubble_inner_pairs, size = np.where(self.labels)[0][-1] + 1)
        return self._inner_adj
    
    @property
    def boundary_mask(self):
        if self._boundary_mask is None:
            # default: no boundaries
            self._boundary_mask = np.zeros(self.size, dtype=bool)
        return self._boundary_mask
    
    def extract(self):
        logger.info(f'(((●))) knocking (((●)))')
        if self.iter_labels is None:
            self._get_lineage()
        self.grow_components()
        self.merge_all()
    
    def _full_lineage(self, lineage, all_components):
        lineage_full = {node: [] for node in all_components.keys()}
        for node, children in lineage.items():
            lineage_full[node] = children 
        self.lineage_full = lineage_full
        
    def get_image_component_lineage(self):
        t0 = time.time()
        from scipy.ndimage import label, binary_erosion, generate_binary_structure
        structure = generate_binary_structure(2, 2)  
        structure_test = generate_binary_structure(2, 1) 
        
        working_image = self._data.inner
        n_iter = self.config.peel_iteration
        min_size = self.config.min_component_size
        size_factor = self.config.component_iteration_size_factor
        
        label_structure = generate_binary_structure(2, 1)  
        erosion_structure = generate_binary_structure(2, 2)  
        
        pixel_id = np.arange(working_image.size).reshape(working_image.shape)
        
        erosion_iter = np.zeros_like(self._data.data, dtype=int)
        logger.info(f'-----> extracting k-core components ({n_iter} peeling iterations) ...')
        lineage = {}  
        all_components = {} 
        prev_comps = []
        iter_idx = 0
        while iter_idx < n_iter:
            num_limit = min_size + size_factor * iter_idx
            labeled, num_features = label(working_image, structure = label_structure)
            comps = []
            for i in range(1, num_features + 1):
                ids = pixel_id[labeled == i]
                if len(ids) >= num_limit:
                    comps.append(ids)
            num = len(comps)
            if num < 2:
                erosion_iter[working_image] = max(0, iter_idx - 1)
                break
            logger.info(f"   ◈   iter {iter_idx} | min_size: {num_limit} | components: {num}")

            voxel_to_prev = {}
            if iter_idx > 0:
                for pcid, pcomp in enumerate(prev_comps):
                    for vid in pcomp:
                        voxel_to_prev[vid] = pcid

            for cid, comp in enumerate(comps):
                all_components[(iter_idx, cid)] = comp
                if iter_idx > 0:
                    parent_ids = set(voxel_to_prev.get(vid) for vid in comp 
                                     if vid in voxel_to_prev)
                    for pid in parent_ids:
                        lineage.setdefault((iter_idx - 1, pid), []).append((iter_idx, cid))
            eroded = binary_erosion(working_image, structure=structure, border_value = 1, iterations = 1)
            removed_voxels = working_image & (~eroded)
            erosion_iter[removed_voxels] = iter_idx + 1
            working_image = eroded
            prev_comps = comps
            iter_idx += 1
        logger.info(f"   ✔   done in {np.round(time.time() - t0):.1f}s")
        self._full_lineage(lineage, all_components)
        self.iter_labels = erosion_iter.flatten()
        self.all_components = all_components
        
        
    def get_component_lineage(self, mask_valid = None):
        n_iter = self.config.peel_iteration
        min_size = self.config.min_component_size
        size_factor = self.config.component_iteration_size_factor
        t0 = time.time()
        working_inner = np.flatnonzero(self.labels)
        working_adj = self.adj[working_inner[:, None], working_inner]
        iter_label = np.zeros(self.size, dtype = int)
        lineage = {}
        all_components = {} 
        prev_comps = []
        
        logger.info(f'-----> extracting k-core components ({n_iter} peeling iterations) ...')
        iter_idx = 0
        while iter_idx < n_iter:
            iter_label[working_inner] = iter_idx + 1
            _, labels = connected_components(working_adj, directed=False)            
            counts = np.bincount(labels) 
            size_limit = min_size + size_factor * iter_idx
            large_component_ids = np.flatnonzero(counts >= size_limit)
            comps = [working_inner[labels == cid] for cid in large_component_ids]
            indices_to_exclude = None
            if iter_idx == 0 and mask_valid is not None:
                comps_use = [comp for comp in comps if np.any(mask_valid[comp])]
                if len(comps_use) < 2:
                    break
                all_nodes = np.concatenate(comps)
                valid_nodes = np.concatenate(comps_use)
                indices_to_exclude = np.setdiff1d(all_nodes, valid_nodes, assume_unique=False)
                comps = comps_use
                working_inner = working_inner[~np.isin(working_inner, indices_to_exclude)]
                working_adj = self.adj[working_inner[:, None], working_inner]
            num = len(comps)
            if num < 2:
                break
            logger.info(f"   ◈   iter {iter_idx} | min_size: {size_limit} | scanned: {len(working_inner)} | components: {num}")

            voxel_to_prev = {}
            if iter_idx > 0:
                for pcid, pcomp in enumerate(prev_comps):
                    for vid in pcomp:
                        voxel_to_prev[vid] = pcid

            for cid, comp in enumerate(comps):
                all_components[(iter_idx, cid)] = comp
                if iter_idx > 0:
                    parent_ids = set(voxel_to_prev.get(vid) for vid in comp if vid in voxel_to_prev)
                    for pid in parent_ids:
                        lineage.setdefault((iter_idx - 1, pid), []).append((iter_idx, cid))

            working_degree_ori = self.node_degrees[working_inner]
            working_degree = working_adj.sum(axis=1).A1
            working_inner = working_inner[working_degree == working_degree_ori]

            if len(working_inner) == 0:
                break
                
            working_adj = self.adj[working_inner[:, None], working_inner]
            prev_comps = comps
            iter_idx += 1 
        logger.info(f"   ✔   done in {np.round(time.time() - t0):.1f}s")
        self._full_lineage(lineage, all_components)
        self.iter_labels = iter_label
        self.all_components = all_components

    def _compute_components(self):
        parent_map = {}
        for parent, children in self.lineage_full.items():
            for child in children:
                parent_map[child] = parent
        leaves = [node for node, children in self.lineage_full.items() if len(children) == 0]
        earliest_nodes = set()
        invalid_nodes = set()
        for leaf in leaves:
            node = leaf
            invalid = False
            count = 0
            while node in parent_map:
                parent = parent_map[node]
                if len(self.lineage_full[parent]) > 1:
                    if count == 0:
                        invalid = True
                    break
                node = parent
                count += 1
            earliest_nodes.add(node)
            if invalid:
                invalid_nodes.add(node)
        self._earliest_components = list(earliest_nodes)
        self._invalid_components = list(invalid_nodes)

    def earliest_components(self):
        if self._earliest_components is None:
            self._compute_components()
        return self._earliest_components

    def invalid_components(self):
        if self._invalid_components is None:
            self._compute_components()
        return self._invalid_components
    
    @staticmethod
    @njit(parallel=True)
    def _assign_batch(unassigned_nodes, indptr, indices, assigned, valid_mask, 
                      single_connect, connect):
        n = len(unassigned_nodes)
        new_groups = -np.ones(n, dtype=assigned.dtype)
        max_g = assigned.max()
        for i in prange(n):
            node = unassigned_nodes[i]
            neighbors = indices[indptr[node]:indptr[node + 1]]
            valid_neighbors = 0
            for nb in neighbors:
                if valid_mask[nb]:
                    valid_neighbors += 1
            if valid_neighbors < single_connect:
                continue
                
            counts = np.zeros(max_g + 1, dtype=np.int32)
            for nb in neighbors:
                if valid_mask[nb]:
                    counts[assigned[nb]] += 1
            max_count = 0
            num_groups = 0
            for c in counts:
                if c > 0:
                    num_groups += 1
                    if c > max_count:
                        max_count = c
            if (num_groups == 1 and max_count >= single_connect) or (max_count >= connect):
                new_groups[i] = np.argmax(counts)
        return new_groups
    
    def assign_nodes(self, unassigned, assigned, use_comp_inds):
        chunk_size = self.config.chunk_size
        inner_adj = self.inner_adj
    
        valid_mask = np.zeros(self.size, dtype=bool)
        mask_lookup = np.zeros(assigned.max() + 1, dtype=bool)
        mask_lookup[use_comp_inds] = True
        while len(unassigned) > 0:
            valid_mask[:] = False
            grouped = assigned >= 0
            valid_mask[grouped] = mask_lookup[assigned[grouped]]
            num_batches = ceil(len(unassigned) / chunk_size)
            uni_groups = [unassigned[i*chunk_size : (i+1)*chunk_size] for i in range(num_batches)]
            single_connect = self.config.single_connect
            connect = self.config.connect
            # Assign nodes to components in parallel batches
            results = Parallel(n_jobs=self.config.n_jobs, prefer="threads")(
                delayed(InnerBubbleGroups._assign_batch)(
                    batch, inner_adj.indptr, inner_adj.indices,
                    assigned, valid_mask, single_connect, connect) for batch in uni_groups)

            mask_total = 0
            for batch, new_groups in zip(uni_groups, results):
                mask = new_groups >= 0
                if mask.any():
                    mask_total += mask.sum()
                    assigned[batch[mask]] = new_groups[mask]
            if mask_total == 0:
                break
            unassigned = unassigned[assigned[unassigned] < 0]
        return assigned
    
    def grow_components(self):
        iter_label = self.iter_labels
        all_components = self.all_components
        use_nodes = self.earliest_components()
        invalid_nodes = self.invalid_components()
        del self.lineage_full
        use_nodes = sorted(use_nodes)
        logger.info(f'-----> re-growing {len(use_nodes)} components ...')
        use_node_iter = np.array([i[0] for i in use_nodes])
        max_iter = np.max(use_node_iter)

        assigned = np.full(self.size, -1, dtype = np.int32)
        invalid_indices = []
        for group_id, use_node_i in enumerate(use_nodes):
            if use_node_i in invalid_nodes:
                invalid_indices.append(group_id)
            comp = all_components[use_node_i]
            for node in comp:
                assigned[node] = group_id
        del all_components
        del self.all_components
        for iter_i in range(max_iter, -1, -1):
            t0 = time.time()
            layer_i = iter_i + 1
            mask_unassigned = (iter_label >= layer_i) & (assigned == -1) & self.labels
            unassigned_nodes = np.flatnonzero(mask_unassigned)
            use_comp_inds = np.flatnonzero(use_node_iter >= iter_i)
            assigned = self.assign_nodes(unassigned_nodes, assigned, use_comp_inds)
            logger.info(f"   ◈   iter {iter_i} | t: {time.time() - t0:.1f}s | assigned: {np.sum(assigned >= 0)}")
        logger.info(f'   ✔   done ')
        self.assigned = assigned
        self.grouped_mask = self.assigned != -1
        self.invalid_component_ids = invalid_indices
        self.isolated_groups = np.flatnonzero(use_node_iter == 0).tolist()
        
    def get_group_cfd(self, group_num: int) -> float:
        group_cfd = self.cfd[self.assigned == group_num]
        if len(group_cfd) == 0:
            return 0.0
        return np.median(group_cfd)
    
    @staticmethod
    @njit
    def _gather_neighbors(nodes, indices, indptr):
        total = 0
        for n in nodes:
            total += indptr[n+1] - indptr[n]
        
        if total == 0:
            return np.empty(0, dtype=indices.dtype), 0
            
        all_neighbors = np.empty(total, dtype=indices.dtype)
        pos = 0
        for n in nodes:
            start, end = indptr[n], indptr[n+1]
            size = end - start
            all_neighbors[pos: pos + size] = indices[start:end]
            pos += size

        unique_neighbors = np.unique(all_neighbors)    
        nodes_sorted = np.sort(nodes)
        result = np.empty(len(unique_neighbors), dtype=indices.dtype)
        result_pos = 0

        for val in unique_neighbors:
            idx = np.searchsorted(nodes_sorted, val)
            if idx >= len(nodes_sorted) or nodes_sorted[idx] != val:
                result[result_pos] = val
                result_pos += 1
        return result[:result_pos], result_pos
    
    @staticmethod
    def _should_merge_group(group_num, indices, indptr, assigned, grouped_mask, boundary_mask, 
                           invalid_component_ids, max_group_size, threshold):
        invalid_groups = invalid_component_ids
        group_nodes = np.flatnonzero(assigned == group_num)
        if len(group_nodes) == 0 or len(group_nodes) > max_group_size:
            return None
        num_boundary = np.sum(boundary_mask[group_nodes])
        neighbor_indices, num_neighbors = InnerBubbleGroups._gather_neighbors(group_nodes, indices, indptr) 
        touching_groups = assigned[neighbor_indices]
        valid_touching = touching_groups[grouped_mask[neighbor_indices]]
        num_valid = len(valid_touching)
        if num_valid == 0:
            return None
        num_valid = num_valid + num_boundary
        if num_valid / num_neighbors >= threshold or group_num in invalid_groups:
            counts = np.zeros(np.max(touching_groups) + 1, np.int64)
            for v in touching_groups:
                counts[v] += 1
            group_to_merge = np.argmax(counts)
            return group_to_merge
        else:
            return group_num
        
    def merge_round(self) -> Tuple[np.ndarray, List[int]]:
        group_skip = self.isolated_groups or []
        unique_groups, counts = np.unique(
            self.assigned[self.grouped_mask], 
            return_counts=True)
        
        order = np.argsort(counts)
        groups_to_process = unique_groups[order]
        
        if len(group_skip) > 0:
            groups_to_process = [g for g in groups_to_process if g not in group_skip]
        logger.info(f"   ·   {len(groups_to_process)} candidate groups ...")
        if self.config.enforce_terminal_bijection:
            invalid_component_ids = self.invalid_component_ids
        else:
            invalid_component_ids = []
        
        # Batch merge groups in parallel
        merge_decisions = Parallel(n_jobs=self.config.n_jobs, prefer="threads")(
            delayed(InnerBubbleGroups._should_merge_group)(group_num, self.adj.indices, 
                                                               self.adj.indptr, self.assigned,
                                                               self.grouped_mask, self.boundary_mask, 
                                                               invalid_component_ids, self.config.max_group_size, 
                                                               self.config.threshold)for group_num in groups_to_process)
        new_assigned = self.assigned.copy()
        new_skip = []
        for group_id, target_group in zip(groups_to_process, merge_decisions):
            if target_group is None:
                new_skip.append(group_id)
            elif target_group != group_id:
                new_assigned[new_assigned == group_id] = target_group
        self.assigned = new_assigned
        self.grouped_mask = self.assigned != -1
        self.isolated_groups.extend(new_skip)
        return
    
    def merge_all(self, max_iterations: int = 30) -> np.ndarray:        
        iteration = 0
        logger.info(f"-----> merging groups with total overlap greater than {self.config.threshold} ...")
        while iteration < max_iterations:
            num_groups_before = len(np.unique(self.assigned[self.grouped_mask]))
            t0 = time.time()
            self.merge_round()
            num_groups_after = len(np.unique(self.assigned[self.grouped_mask]))
            logger.info(f"   ◈   iter {iteration} | t: {time.time() - t0:.1f}s | groups: {num_groups_before} → {num_groups_after}")
            if num_groups_after >= num_groups_before:
                logger.info(f"   ✔   converged after {iteration + 1} iterations")
                break
            iteration += 1        
        if iteration >= max_iterations:
            logger.warning(f" max. iterations ({max_iterations}) reached without convergence")
            
@dataclass
class BubbleShellConfig:
    inner_connect: int = 2
    connect: int = 3
    max_layer: int = 6
    max_iter: int = 3
    pair_order: Literal['density', 'density_temperature'] = 'density'
    chunk_size: int = 200000
    n_jobs: int = -1 
    
    
class BubbleShellAssignment:
    def __init__(self, data, save_results: bool = True, **kwargs):
        self._data = data
        if isinstance(self._data, SnapData):
            self._get_vor = lambda: data.vor
            self._swap_pairs = self.swap_vor_pairs
            self._shell_mask = self.shell_mask
            self.size = self._data.num_particles
            self._get_den = lambda: data.den
        elif isinstance(self._data, ImageData):
            self._get_vor = lambda: data.neighbors_all
            self._swap_pairs = self.swap_neighbour_pairs
            self._shell_mask = self.dilated_shell_mask
            self.size = self._data.num_pixels
            self._get_den = lambda: data.data.flatten()
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        self._vor = None
        self.inner_assigned = self._data.inner_assigned
        self.mask_inner = self.inner_assigned != -1
        self.config = BubbleShellConfig(**kwargs)
        self.save_results = save_results
        self._shell_csr = None
        
    @property
    def vor(self):
        if self._vor is None:
            self._vor = self._get_vor()
        return self._vor
        
    def dilated_shell_mask(self):
        from scipy.ndimage import binary_dilation
        mask_use = self._data.inner
        logger.info(f"-----> expanding {self.config.max_layer} layers outward to find shells ...")
        dilated = binary_dilation(mask_use, iterations = self.config.max_layer)
        mask_outer = dilated & (~self._data.inner) & (self._data.data > 0)     
        self.mask_outer = mask_outer.flatten()
        logger.info(f"   ✔   {np.sum(self.mask_outer)} possible shell cells found")
        
    def shell_mask(self):
        count = 0
        mask_use = self.mask_inner
        logger.info(f"-----> expanding {self.config.max_layer} layers outward to find shells ...")
        for _ in range(self.config.max_layer):
            mask_use = (self._data.adj[mask_use].sum(axis=0) > 0).A1
        self.mask_outer = mask_use & (~self.mask_inner)
        logger.info(f"   ✔   {np.sum(self.mask_outer)} possible shell cells found")
        
    def swap_neighbour_pairs(self):
        src = self.vor[:, 0]
        dst = self.vor[:, 1]
        iten_src, iten_dst = self._get_den()[src], self._get_den()[dst]
        logger.info("-----> ordering pairs by intensity ↑ ...")
        forward = (iten_src >= iten_dst)
        backward = (iten_dst >= iten_src)
        valid = forward | backward
        rows = np.where(forward[valid], src[valid], dst[valid])
        cols = np.where(forward[valid], dst[valid], src[valid])
        return rows, cols
        
    def swap_vor_pairs(self):
        src = self.vor[:, 0]
        dst = self.vor[:, 1]
        den_src, den_dst = self._data.den[src], self._data.den[dst]
        if self.config.pair_order == 'density':
            logger.info("-----> ordering pairs by density ↑ ...")
            forward = (den_src >= den_dst)
            backward = (den_dst >= den_src)
        elif  self.config.pair_order == 'density_temperature':
            logger.info("-----> ordering pairs by density ↑, temperature ↓ ...")
            tem_src, tem_dst = self._data.tem[src], self._data.tem[dst]
            forward = (den_src >= den_dst) & (tem_src <= tem_dst)
            backward = (den_dst >= den_src) & (tem_dst <= tem_src)
        else:
            raise ValueError(f"Unknown pair_order '{self.config.pair_order}'. Expected 'density' or 'density_temperature'.")
            
        valid = forward | backward
        rows = np.where(forward[valid], src[valid], dst[valid])
        cols = np.where(forward[valid], dst[valid], src[valid])
        return rows, cols
    
    @property
    def shell_csr(self):
        if self._shell_csr is None:
            logger.info("-----> building shell adjacency matrix ... ")
            self._shell_mask()
            mask_outer_pairs = self.mask_outer[self.vor[:, 0]] | self.mask_outer[self.vor[:, 1]]
            outer_vor = self.vor[mask_outer_pairs] 
            rows, cols = self._swap_pairs()
            self._shell_csr =  csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), 
                                          shape=(self.size, self.size)) 
            logger.info("   ✔   shell CSR built")
        return self._shell_csr
       
    def sort_nodes_by_den(self, outer_indices, den_order = 'increasing'):
        if den_order == 'increasing':
            sorted_nodes = outer_indices[np.argsort(self._get_den()[outer_indices])]
        if den_order == 'decreasing':
            sorted_nodes = outer_indices[np.argsort(self._get_den()[outer_indices])][::-1]
        return sorted_nodes

    @staticmethod
    @njit(parallel=True)
    def _assign_shell(nodes, indptr, indices, inner_shell_assigned, mask_inner, inner_connect, connect):
        result = np.full(len(nodes), -1, dtype=np.int32)
        for i in prange(len(nodes)):
            node = nodes[i]
            neighbours = indices[indptr[node]:indptr[node+1]]
            if len(neighbours) < inner_connect:
                continue
            inner_neighs = mask_inner[neighbours]
            neighbour_groups = inner_shell_assigned[neighbours]
            neighbour_regions = neighbour_groups[neighbour_groups >= 0]
            num_neighbour = len(neighbour_regions)
            if inner_neighs.any():
                if num_neighbour >= inner_connect:
                    n = len(neighbour_regions)
                    counts = np.zeros(np.max(neighbour_regions) + 1, np.int64)
                    for v in neighbour_regions:
                        counts[v] += 1
                    most_common = np.argmax(counts)
                    count = counts[most_common]
                    if count >= inner_connect:
                        result[i] = most_common
            else:
                if num_neighbour >= connect:
                    n = len(neighbour_regions)
                    counts = np.zeros(np.max(neighbour_regions) + 1, np.int64)
                    for v in neighbour_regions:
                        counts[v] += 1
                    most_common = np.argmax(counts)
                    count = counts[most_common]
                    if count >= connect:
                        result[i] = most_common
        return result

    def assign(self): 
        t0 = time.time()
        max_iter = self.config.max_iter
        indptr = self.shell_csr.indptr
        indices = self.shell_csr.indices
        sorted_nodes = self.sort_nodes_by_den(np.flatnonzero(self.mask_outer))
        inner_shell_assigned = self.inner_assigned.copy()
        
        inner_connect = self.config.inner_connect
        connect = self.config.connect
        logger.info("-----> assigning shells to bubbles ... ")
        
        # Batch process shell assignment in parallel
        chunk_size = self.config.chunk_size
        for round_i in range(max_iter):
            num_batches = ceil(len(sorted_nodes) / chunk_size)
            uni_groups = [sorted_nodes[i*chunk_size : (i+1)*chunk_size] for i in range(num_batches)]
            results = Parallel(n_jobs=self.config.n_jobs, prefer="threads")(delayed(BubbleShellAssignment._assign_shell)(
                    batch, indptr, indices, inner_shell_assigned, self.mask_inner, 
                    inner_connect, connect) for batch in uni_groups)
            mask_total = 0
            for batch, new_groups in zip(uni_groups, results):
                mask = new_groups >= 0
                if mask.any():
                    mask_total += mask.sum()
                    inner_shell_assigned[batch[mask]] = new_groups[mask]
            logger.info(f"   ◈   iter {round_i} | assigned: {mask_total}")
            if mask_total == 0:
                break 
            sorted_nodes = sorted_nodes[inner_shell_assigned[sorted_nodes] < 0]
        logger.info(f"   ✔   done in {np.round(time.time() - t0):.1f}s")
        self.inner_shell_assigned = inner_shell_assigned
        self._data.bubble_groups = self.inner_shell_assigned
