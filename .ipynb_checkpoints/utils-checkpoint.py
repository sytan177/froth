import pickle, os, sys, joblib
import numpy as np
import h5py
from typing import List, Tuple, Optional
from dataclasses import dataclass
arepo_analysis_path ='/home/hpc/a104bc/a104bc20/ecogal/analysis'
sys.path.append(arepo_analysis_path)
analysis_path ='/home/hpc/a104bc/a104bc20/analysis/bubbles'
sys.path.append(analysis_path)
import argparse
from bubble_utils_gnn import dict_indexing
from collections import Counter, defaultdict
import igraph as ig
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from GCN_utils import load_masked_snapshot
from numba import njit, types
from numba.typed import Dict
from scipy.sparse import csr_matrix
from joblib import load
from joblib import Parallel, delayed

#### Some calculation helpers:
@njit
def gather_neighbors(nodes, edges_flat, offsets):
    total_len = 0
    for n in nodes:
        total_len += offsets[n+1] - offsets[n]
    result = np.empty(total_len, dtype=edges_flat.dtype)
    pos = 0
    for n in nodes:
        start, end = offsets[n], offsets[n+1]
        length = end - start
        result[pos:pos+length] = edges_flat[start:end]
        pos += length
    result = np.unique(result)
    total_len = len(result)
    return result, total_len

@njit
def mode_and_count(arr):
    n = len(arr)
    if n == 0:
        return -1, 0

    amin = np.min(arr)
    amax = np.max(arr)
    if amax - amin < 10000:
        counts = np.zeros(amax - amin + 1, np.int64)
        for v in arr:
            counts[v - amin] += 1
        idx = np.argmax(counts)
        return amin + idx, counts[idx]
    else:
        counts = Dict.empty(key_type=types.int64, value_type=types.int64)
        for v in arr:
            counts[v] = counts.get(v, 0) + 1
        best_label = -1
        best_count = -1
        for k, v in counts.items():
            if v > best_count:
                best_label = k
                best_count = v
        return best_label, best_count

def extract_snapshot_id(filename):
    import re
    match = re.search(r"bubbles_\d+_(\d+)_", filename)
    return int(match.group(1)) if match else float('inf')

def build_adj(pairs):
    edges_arr = np.array(pairs)
    n_nodes = edges_arr.max() + 1
    row = edges_arr[:, 0]
    col = edges_arr[:, 1]
    data = np.ones(len(row), dtype=bool)
    adj = csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
    adj = adj + adj.T
    return adj

class SnapData:
    def __init__(self, snap_num, snap_data): 
        self.snap_num = snap_num
        self.snap_data = snap_data
        self._num_particles = None
        for key in snap_data.keys():
            setattr(self, f'{key}', snap_data[key])
            
    def num_particles(self):
        if self._num_particles is None:
            self._num_particles = len(self.labels)
        return self._num_particles
        
    def load_vor_edges(self, num_files=32, distance_threshold=0.03): 
        gas_phi = np.arctan2(self.pos[:, 1], self.pos[:, 0])
        pad = 0.005
        phi_ranges = np.linspace(-np.pi, np.pi, 33)
        mask_overlap = np.zeros(len(self.ids), dtype=bool)
        for phi_i in phi_ranges:
            mask_pad = ((gas_phi - phi_i + pad) % (2*np.pi)) < ((pad * 2) % (2*np.pi))
            mask_overlap[mask_pad] = True
        overlap_indices = np.flatnonzero(mask_overlap)
        vor_save_path = '/home/atuin/a104bc/a104bc20/gcn_res/results/vor_ridges/'
        vor_files = sorted([i for i in os.listdir(vor_save_path) if str(self.snap_num) in i])
        print(f'Loading {len(vor_files)} Voronoi files')
        
        core_edges = []
        overlap_edges = []
        
        id_to_index = np.full(int(self.id.max()) + 1, -1, dtype=np.int64)
        id_to_index[self.id] = np.arange(self.num_particles())
        
        for vor_file_i in vor_files[:num_files]:
            vor_data = np.load(os.path.join(vor_save_path, vor_file_i))
            gas_ids_sec = vor_data['gas_ids']
            pairs = vor_data['vor_ridge_points']
            mapped_pairs = id_to_index[gas_ids_sec][pairs]
            delta = self.pos[mapped_pairs[:, 0]] - self.pos[mapped_pairs[:, 1]]
            distances = np.linalg.norm(delta, axis=1)
            filtered = mapped_pairs[distances < distance_threshold]
            
            mask_overlap_local = np.isin(filtered[:, 0], overlap_indices) | np.isin(filtered[:, 1], overlap_indices)
            overlap_edges.append(filtered[mask_overlap_local])
            core_edges.append(filtered[~mask_overlap_local])
        
        core_edges = np.concatenate(core_edges, axis=0)
        overlap_edges = np.concatenate(overlap_edges, axis=0)
        overlap_unique = np.unique(np.sort(overlap_edges, axis=1), axis=0)
        self.vor = np.vstack([core_edges, overlap_unique]).astype(np.int32)
        return self.vor
    
@dataclass
class BubbleMergeConfig:
    threshold: float = 0.2
    confidence_threshold: float = 0.9
    max_group_size: int = int(1e6)
    n_jobs: int = -1
    
class BubbleInnerExtraction():
    def __init__(self, i_SnapData, gas_labels, gas_cfd, config):
        self.snap = i_SnapData
        self.config = config
        
        self.labels = ~gas_labels.astype(bool) # 1 for inner region, 0 for the rest
        self.cfd = gas_cfd
        self._adj = None
        self._inner_adj = None
        self._node_degrees = None
        
        self._earliest_components = None
        self._invalid_components = None
        
        self._boundary_mask = None
                
    def inner_indices(self):
        return np.flatnonzero(self.labels)
    
    def node_degrees(self):
        if self._node_degrees is None:
            self._node_degrees = self.adj.sum(axis=1).A1
        return self._node_degrees
    
    def all_adj(self):
        if self._adj is None:
            self._adj = build_adj(self.snap.vor)
        return self._adj
    
    def inner_adj(self):
        if self._inner_adj is None:
            a = self.snap.vor[:, 0]
            b = self.snap.vor[:, 1]
            inner_mask = self.labels[a] & self.labels[b]    
            bubble_inner_pairs = self.snap.vor[inner_mask]
            self._inner_adj = build_adj(bubble_inner_pairs)
        return self._inner_adj
    
    def get_boundary_mask(self, padding = 0.005):
        if self._boundary_mask is None:
            gas_R = np.sqrt(self.snap.pos[:, 0]**2 + self.snap.pos[:, 1]**2)
            touching_R = (gas_R > 9.5 - padding) | (gas_R < 7.5 + padding)
            touching_z_up =  gas_pos[:, 2] > 1. - padding
            touching_z_down =  gas_pos[:, 2] < -1. + padding
            self._boundary_mask = touching_R | touching_z_up | touching_z_down
        return self._boundary_mask
    
    def get_component_lineage(self, mask_valid = None, size_factor = 0, n_iter = 10):
        t0 = time.time()
        working_inner = self.inner_indices()
        working_adj = self.adj[working_inner[:, None], working_inner] # built from the inner pairs
        iter_label = np.zeros(self.snap.num_particles(), dtype = int)
        lineage = {}
        all_components = {} 
        prev_comps = []
        iter_idx = 0
        while iter_idx < n_iter:
            iter_label[working_inner] = iter_idx + 1
            _, labels = connected_components(working_adj, directed=False)            
            counts = np.bincount(labels) 
            size_limit = 5 + size_factor * iter_idx
            large_component_ids = np.flatnonzero(counts > size_limit)
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
            print(len(working_inner), num) 

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

            working_degree_ori = self.node_degrees()[working_inner]
            working_degree = working_adj.sum(axis=1).A1
            use_mask = np.flatnonzero(working_degree == working_degree_ori)
            new_inner_indices = working_inner[use_mask]

            if len(new_inner_indices) == 0:
                break
            working_adj = self.adj[new_inner_indices[:, None], new_inner_indices]
            working_inner = new_inner_indices
            prev_comps = comps
            iter_idx += 1
            
        self.iter_label = iter_label
        self.lineage = lineage
        self.all_components = all_components
        print('Time taken: ', np.round(time.time() - t0))
        return self.iter_label, self.lineage, self.all_components

    def _compute_components(self):
        parent_map = {}
        for parent, children in self.lineage.items():
            for child in children:
                parent_map[child] = parent

        leaves = [node for node, children in self.lineage.items() if len(children) == 0]
        earliest_nodes = set()
        invalid_nodes = set()
        for leaf in leaves:
            node = leaf
            invalid = False
            count = 0
            while node in parent_map:
                parent = parent_map[node]
                if len(self.lineage[parent]) > 1:
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

    @njit
    def _assign_nodes(indptr, indices, assigned, valid_mask, unassigned_nodes):
        new_assignments = []
        for node in unassigned_nodes:
            start = indptr[node]
            end = indptr[node + 1]
            neighbors = indices[start:end]
            valid_neighbors = neighbors[valid_mask[neighbors]]
            if len(valid_neighbors) == 0:
                continue
            neighbor_groups = assigned[valid_neighbors]
            counts = np.bincount(neighbor_groups)
            groups = np.flatnonzero(counts)
            counts = counts[groups]
            max_count = counts.max()
            if (len(groups) == 1 and max_count >= 3) or (max_count >= 4):
                assigned[node] = groups[np.argmax(counts)]
                new_assignments.append(node)
        return assigned, np.array(new_assignments)

    def assign_nodes(self, assigned, valid_mask, unassigned_nodes):
    return _assign_nodes(self.inner_adj.indptr, self.inner_adj.indices,assigned, valid_mask, unassigned_nodes)
    
    def grow_components(self):
        lineage_full = {node: [] for node in self.all_components.keys()}
        for node, children in self.lineage.items():
            lineage_full[node] = children 
        use_nodes = self.earliest_components()
        invalid_nodes = self.invalid_components()
        use_nodes = sorted(use_nodes)
        print('Raw components = ', len(use_nodes), use_nodes[-1])
        use_node_iter = np.array([i[0] for i in use_nodes])
        max_iter = np.max(use_node_iter)
        all_iterations = list(range(max_iter, -1, -1))
        assigned = np.full(self.snap.num_particles(), -1, dtype = np.int32)
        invalid_indices = []
        for group_id, use_node_i in enumerate(use_nodes):
            if use_node_i in invalid_nodes:
                invalid_indices.append(group_id)
            comp = self.all_components[use_node_i]
            for node in comp:
                assigned[node] = group_id

        valid_mask = np.zeros(self.snap.num_particles(), dtype=bool)
        mask_lookup = np.zeros(assigned.max() + 1, dtype=bool)
        for iter_i in all_iterations[:]:
            t0 = time.time()
            layer_i = iter_i + 1
            mask_unassigned = (self.iter_label >= layer_i) & (assigned == -1) & self.labels
            unassigned_nodes = np.nonzero(mask_unassigned)[0]
            use_comp_inds = np.flatnonzero(use_node_iter >= iter_i)
            print('Iteration: ', iter_i, '# of unassigned nodes: ', len(unassigned_nodes), '# current components: ', len(use_comp_inds))
            while len(unassigned_nodes) > 0:
                valid_mask[:] = False
                mask_lookup[:] = False
                mask_lookup[use_comp_inds] = True
                pos = assigned >= 0
                valid_mask[pos] = mask_lookup[assigned[pos]]
                assigned, new_assignments = assign_nodes(self, assigned, valid_mask, unassigned_nodes)
                if len(new_assignments) == 0:
                    break
                assigned_mask = np.zeros(len(assigned), dtype=bool)
                assigned_mask[np.array(new_assignments)] = True
                unassigned_nodes = unassigned_nodes[~assigned_mask[unassigned_nodes]]
            print(np.round(time.time() - t0), len(unassigned_nodes))
        self.assigned = assigned
        self.grouped_mask = self.assigned != -1
        self.invalid_component_ids = invalid_indices
        self.isolated_groups = np.flatnonzero(use_node_iter == 0)
        
    def get_group_cfd(self, group_num: int) -> float:
        group_cfd = self.cfd[self.assigned == group_num]
        if len(group_cfd) == 0:
            return 0.0
        return np.median(group_cfd)
    
    def should_merge_group(self, group_num: int) -> Optional[int]:
        invalid_groups = self.invalid_component_ids or []
        group_nodes = np.flatnonzero(self.assigned == group_num)
        if len(group_nodes) == 0 or len(group_nodes) > self.config.max_group_size:
            return None
        #num_boundary = np.sum(self._boundary_mask[group_nodes])
        neighbor_indices, num_neighbors = gather_neighbors(group_nodes, self.adj.indices, self.adj.indptr) 
        touching_groups = self.assigned[neighbor_indices]
        valid_touching = touching_groups[self.grouped_mask[neighbor_indices]]
        num_valid = len(valid_touching)
        if num_valid == 0:
            return None
        #num_valid = num_valid + num_boundary
        cfd_i = self.get_group_cfd(group_num)
        if cfd_i < self.config.confidence_threshold or group_num in invalid_groups:
            group_to_merge, max_count = mode_and_count(valid_touching) 
            return group_to_merge
        else:
            return group_num
        
    def merge_round(self) -> Tuple[np.ndarray, List[int]]:
        t0 = time.time()
        group_skip = self.isolated_groups or []
        unique_groups, counts = np.unique(
            self.assigned[self.grouped_mask], 
            return_counts=True)
        
        order = np.argsort(counts)
        groups_to_process = unique_groups[order]
        
        if len(group_skip) > 0:
            groups_to_process = [g for g in groups_to_process if g not in group_skip]
        print(f'Processing {len(groups_to_process)} groups')
        merge_decisions = Parallel(
            n_jobs=self.config.n_jobs, 
            prefer="threads"
        )(
            delayed(self.should_merge_group)(group_num)for group_num in groups_to_process)
        
        new_assigned = self.assigned.copy()
        new_skip = []
        for group_id, target_group in zip(groups_to_process, merge_decisions):
            if target_group is None:
                new_skip.append(group_id)
            elif target_group != group_id:
                new_assigned[new_assigned == group_id] = target_group
        
        self.assigned = new_assigned
        self.grouped_mask = self.assigned != -1
        self.first_groups.extend(new_skip)
        print(f'Round completed in {np.round(time.time() - t0, 2)}s')
        return
    
    def merge_all(self, max_iterations: int = 30) -> np.ndarray:        
        iteration = 0
        while iteration < max_iterations:
            num_groups_before = len(np.unique(self.assigned[self.grouped_mask]))
            self.merge_round()
            num_groups_after = len(np.unique(self.assigned[self.grouped_mask]))
            print(f'Iteration {iteration}: {num_groups_before} -> {num_groups_after} groups')
            if num_groups_after >= num_groups_before:
                print(f'Converged after {iteration + 1} iterations')
                break
            iteration += 1        
        if iteration >= max_iterations:
            print(f'Warning: Reached max iterations ({max_iterations})')

class BubbleShellAssignment:
    def __init__(self, i_SnapData, gas_labels, gas_cfd, config):
        self.snap = i_SnapData
        self.config = config
        
        self.labels = ~gas_labels.astype(bool) # 1 for inner region, 0 for the rest
        self.cfd = gas_cfd
        self._adj = None
        self._boundary_adj = None
        
    
    def assign_shells(gas_den, mask_inner, outer_indices, shell_csr, inner_labelled, inner_shell_labelled, 
                      den_order = 'increasing', connect = 4):
        indptr = shell_csr.indptr
        indices = shell_csr.indices  
        if den_order == 'increasing':
            sorted_nodes = outer_indices[np.argsort(gas_den[outer_indices])]
        if den_order == 'decreasing':
            sorted_nodes = outer_indices[np.argsort(gas_den[outer_indices])][::-1]

        failed_shell = []
        for node in sorted_nodes:
            neighbours = indices[indptr[node]:indptr[node+1]]
            if len(neighbours) < 2:
                continue
            inner_neighs = mask_inner[neighbours]
            neighbour_groups = inner_shell_labelled[neighbours]
            neighbour_regions = neighbour_groups[neighbour_groups >= 0]
            num_neighbour = len(neighbour_regions)
            recorded = False
            if inner_neighs.any():
                if num_neighbour >= 2:
                    most_common, count = mode_and_count(neighbour_regions)
                    if count >= 2:
                        inner_shell_labelled[node] = most_common
                        recorded = True
            else:
                if num_neighbour >= connect:
                    most_common, count = mode_and_count(neighbour_regions)
                    if count >= connect:
                        inner_shell_labelled[node] = most_common
                        recorded = True
            if not recorded:
                failed_shell.append(node)  
        return inner_shell_labelled, failed_shell

    def swap_vor_ridges(pairs, gas_den, gas_tem):
        src = pairs[:, 0]
        dst = pairs[:, 1]

        den_src, den_dst = gas_den[src], gas_den[dst]
        tem_src, tem_dst = gas_tem[src], gas_tem[dst]

        forward = (den_src >= den_dst) & (tem_src <= tem_dst)
        backward = (den_dst >= den_src) & (tem_dst <= tem_src)
        valid = forward | backward

        rows = np.where(forward[valid], src[valid], dst[valid])
        cols = np.where(forward[valid], dst[valid], src[valid])

        return csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), 
                          shape=(len(gas_den), len(gas_den)))

def find_bubble_groups(i_SnapData, gas_labels, gas_cfd, confidence_threshold: float = 0.9, 
                       threshold: float = 0.2,) -> np.ndarray:
    config = BubbleMergeConfig(confidence_threshold = confidence_threshold, threshold=threshold)
    segment_n_merge = BubbleInnerExtraction(i_SnapData, gas_labels, gas_cfd, config)
    segment_n_merge.merge_all()
    return segment_n_merge.assigned

def extracting_one_snapshot(snap_fd, snap_num, bubble_res_fd, shell_layers = 6):
    bubble_file_nm = f'bubbles_{snap_num}_inner.h5'
    with h5py.File(os.path.join(bubble_res_fd, bubble_file_nm), 'r') as f:
        gas_cfd = f['confidence'][()]
        gas_ids = f['gas_ids'][()]
    gas_labels = (gas_cfd <= 0.5).astype(int)
    snap_gas = load_masked_snapshot(snap_num, temperature = True, z_limit = 1.0)
    gas_pos = snap_gas['position']
    gas_den = snap_gas['density']
    gas_tem = snap_gas['temperature']
    
    snap = SnapData(snap_num, snap_gas)
    uni_vor = snap.load_vor_edges(num_files=1, distance_threshold=0.03)
    mask_boundary = snap.get_boundary()

    all_adj = build_adj(uni_vor)
    #node_mask = ~gas_labels.astype(bool)
    #mask = node_mask[uni_vor[:, 0]] & node_mask[uni_vor[:, 1]]
    #bubble_inner_pairs = uni_vor[mask]
    #mask = node_mask[uni_vor[:, 0]] ^ node_mask[uni_vor[:, 1]]
    #bubble_edge_pairs = uni_vor[mask]
    #edge_indices = np.unique(bubble_edge_pairs)
    #edge_indices = edge_indices[node_mask[edge_indices]]
        
    inner_labelled = inner_components(uni_vor, all_adj, gas_cfd, gas_labels, mask_boundary = mask_boundary)

    bubble_labels, counts = np.unique(inner_labelled[inner_labelled != -1], return_counts=True)
    bubble_labels = bubble_labels[(bubble_labels >= 0) & (counts >= 5)]
    mask_inner = np.isin(inner_labelled, bubble_labels)
    inner_labelled[~mask_inner] = -1
    inner_indices = np.flatnonzero(mask_inner)
    
    count = 0
    mask_use = mask_inner
    while count < shell_layers:
        indices = np.flatnonzero(mask_use)
        outer_neighbours_mask = np.array(all_adj[indices].sum(axis=0)).ravel() > 0
        mask_use = outer_neighbours_mask
        count += 1
    outer_indices = np.flatnonzero(mask_use)
    outer_indices = outer_indices[~np.isin(outer_indices, inner_indices)]
    mask_outer = np.zeros(len(gas_labels), dtype = bool)
    mask_outer[outer_indices] = 1
    print(len(inner_indices), len(outer_indices), len(bubble_labels))
    
    bubble_outer_mask = mask_outer[uni_vor[:, 0]] | mask_outer[uni_vor[:, 1]]
    outer_vor = uni_vor[bubble_outer_mask]
    print(len(outer_vor))
    
    ### Reorder the ridge pairs: pointing from low to high density ###
    t_scan1 = time.time()
    shell_csr = swap_vor_ridges(outer_vor, gas_den, gas_tem)
    inner_shell_labelled = [-1] * len(gas_labels)
    inner_shell_labelled = np.array(inner_shell_labelled)
    inner_shell_labelled[mask_inner] = inner_labelled[mask_inner]
    inner_shell_labelled, failed_shell = assign_shells(gas_den, mask_inner, outer_indices, shell_csr, 
                                                       inner_labelled, inner_shell_labelled)
    print('Shell nodes left = ', len(failed_shell))  
    print('First scan in ', np.round(time.time() - t_scan1), ' s.')
    
    shell_labelled = [-1] * len(gas_labels)
    shell_labelled = np.array(shell_labelled)
    shell_labelled[mask_outer] = inner_shell_labelled[mask_outer]
    
    t_group = time.time()
    bubble_info = []
    for group_i in bubble_labels:
        inner = np.flatnonzero(inner_labelled == group_i)
        shell = np.flatnonzero(shell_labelled == group_i)
        #cfd_i = np.median(gas_cfd[inner])
        if len(inner) < 1e6 and len(shell) > 5:
            bubble_info.append([inner, shell])
        else:
            inner_shell_labelled[inner_shell_labelled == group_i] = -1
            print(len(inner), len(shell)) #, cfd_i
    print(len(bubble_info), ' groups valid.')
    bubble_data = []
    print('Loading the results...')
    for sublist in bubble_info:
        i, j = sublist
        res = [dict_indexing(snap_gas, i), dict_indexing(snap_gas, j)]
        bubble_data.append(res)
    return bubble_data, bubble_info, mask_inner, inner_shell_labelled
    
snap_fd = '/home/atuin/a104bc/a104bc20/Agama_HD_3e3_ObsBasedDensity_SneN/run_output_10pc_vol_diff_4'
bubble_res_fd = '/home/atuin/a104bc/a104bc20/gcn_res/results/'

parser = argparse.ArgumentParser(description='Process snapshot number.')
parser.add_argument('-snap_num', type=int, help='snapshot number')

args = parser.parse_args()
snap_num = args.snap_num

import time
t0 = time.time()
bubble_data, bubble_info, mask_inner, inner_shell_labelled = extracting_one_snapshot(snap_fd, snap_num, bubble_res_fd)


save_path = os.path.join(bubble_res_fd, str(snap_num)+'_results_csr0.pkl')
print('Saving the results...')
#with open(save_path, 'wb') as file:
 #   pickle.dump(bubble_data, file)
    
with h5py.File(os.path.join(bubble_res_fd, str(snap_num)+'_labelled_csr0.h5'), 'w') as f:
    f.create_dataset('labels', data=mask_inner, compression='gzip', chunks=True)
    f.create_dataset('groups', data=inner_shell_labelled, compression='gzip', chunks=True)

    f.attrs['description'] = "Labels (0/1) and group IDs (integers)"
    f.attrs['num_entries'] = mask_inner.size
    
print('Total time taken = ', np.round(time.time() - t0, 2), ' s')
