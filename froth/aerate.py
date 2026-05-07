import time, torch, pickle, os, h5py
from math import ceil
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from typing import Union, Optional, Dict, Any
from scipy.interpolate import RegularGridInterpolator
from sklearn.preprocessing import RobustScaler
from scipy.ndimage import rotate, binary_fill_holes, binary_opening, label, median_filter, gaussian_filter
from skimage.morphology import remove_small_holes

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.loader import NeighborLoader

from .utils import load_masked_snapshot, setup_logging
from .config import ModelConfig, SnapConfig, ImageConfig

import logging
logger = logging.getLogger(__name__)

class ProcessingSnapshot():
    def __init__(self, i_SnapData, config=None):
        self.snap = i_SnapData
        self.config = config or SnapConfig()
    
    def get_Rz_bins(self):
        z_min, z_max = self.snap.pos[:, 2].min(), self.snap.pos[:, 2].max()
        R_min, R_max = self.snap.pos_R.min(), self.snap.pos_R.max()
        num_z_bins = ceil((z_max - z_min) / self.config.z_bin_size)
        num_R_bins = ceil((R_max - R_min) / self.config.R_bin_size)
        
        z_bins = np.linspace(z_min, z_max, num_z_bins + 1)
        R_bins = np.linspace(R_min, R_max, num_R_bins + 1)
        return z_bins, R_bins 
    
    def get_vertical_profile(self, quantity):
        z_bins, R_bins = self.get_Rz_bins()            
        z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
        R_centers = 0.5 * (R_bins[:-1] + R_bins[1:])

        if quantity == 'density':
            data = self.snap.den
        elif quantity == 'temperature':
            data = self.snap.tem
        else:
            raise ValueError(f"Unknown quantity: {quantity}")
            
        path = self.snap._file_config.profile_file(quantity)
        if os.path.exists(path) and not self.config.force_recompute:
            logger.info(f"-----> loading vertical {quantity} profile ...")
            with open(path, 'rb') as f:
                interp_func = pickle.load(f)
        else:
            logger.info(f"-----> creating vertical {quantity} profile ...")
            mass_per_bin, _, _ = np.histogram2d(self.snap.pos_R, self.snap.pos[:, 2],
                                                bins=[R_bins, z_bins], weights=self.snap.mass)
            weights = data * self.snap.mass
            weighted_sum, _, _ = np.histogram2d(self.snap.pos_R, self.snap.pos[:, 2], 
                                                bins=[R_bins, z_bins], weights=weights)
            mean_per_bin = weighted_sum / mass_per_bin
            interp_func = RegularGridInterpolator((R_centers, z_centers), mean_per_bin, method='nearest', bounds_error=False, fill_value=None)
            if self.config.save_scaler:
                logger.info(f"-----> saving vertical {quantity} profile ...")
                with open(path, 'wb') as f:
                    pickle.dump(interp_func, f)
                logger.info(f"   ✔   saved to {path}.")
        return data, interp_func
    
    def scale_quantity(self, quantity: str):
        path = self.snap._file_config.scaler_file(quantity)

        if self.config.normalize:
            logger.info(f"-----> normalizing {quantity} with vertical profile ...")
            data, profile_func = self.get_vertical_profile(quantity)
            positions = np.c_[self.snap.pos_R, self.snap.pos[:, 2]]
            corrected = data / profile_func(positions)
            log_corrected = np.log(corrected)
        else:
            logger.info(f"-----> skipping {quantity} vertical profile normalization ...")
            if quantity == 'density':
                data = self.snap.den
            elif quantity == 'temperature':
                data = self.snap.tem            
            log_corrected = np.log(data)
        if os.path.exists(path) and not self.config.force_recompute:
            with open(path, 'rb') as f:
                logger.info(f"-----> loading {quantity} scaler ...")
                scaler = pickle.load(f)
            scaled_data = scaler.transform(log_corrected.reshape(-1, 1)).flatten()
            logger.info(f"   ✔   {quantity} scaled.")
        else:
            logger.info(f"-----> creating {quantity} scaler ...")
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(log_corrected.reshape(-1, 1)).flatten()
            logger.info(f"   ✔   {quantity} scaled.")
            if self.config.save_scaler:
                logger.info(f"-----> saving vertical {quantity} profile ...")
                with open(path, 'wb') as f:
                    pickle.dump(scaler, f)
                logger.info(f"   ✔   saved to {path}.")
        return scaled_data
    
    def prepare_features(self, feature_names):
        scaled_features = []
        for feature in feature_names:
            scaled_features.append(self.scale_quantity(feature))
        return np.vstack(scaled_features)
    
    @staticmethod
    def prepare_chunk(gas_ids, phy_cons, voronoi_file):
        vor_data = np.load(voronoi_file)
        gas_ids_chunk = vor_data['gas_ids']
        ridge_points = vor_data['vor_ridge_points']
        
        mask = pd.Series(gas_ids).isin(gas_ids_chunk).values
        
        edge_index = torch.tensor(ridge_points, dtype=torch.long)
        edge_index = to_undirected(edge_index.t().contiguous())
        
        phy_cons_chunk = phy_cons[:, mask]
        node_features = torch.tensor(phy_cons_chunk.T, dtype=torch.float32)
        graph_data = Data(x=node_features, edge_index=edge_index)
        return graph_data, mask
    
    @staticmethod
    def prepare(phy_cons, ridge_points):        
        edge_index = torch.tensor(ridge_points, dtype=torch.long)
        edge_index = to_undirected(edge_index.t().contiguous())
        node_features = torch.tensor(phy_cons.T, dtype=torch.float32)
        graph_data = Data(x=node_features, edge_index=edge_index)
        return graph_data
        
        
class ProcessingImage():
    def __init__(self, i_ImageData, image_config):
        self.image = i_ImageData
        self.config = image_config
        if self.config.detection_mode not in ["max", "composed"]:
            raise ValueError(f"   ⚠   detection_mode must be 'composed' or 'max', got '{self.config.detection_mode}'")
        self.mode = self.config.detection_mode
        
    def _validate(self):
        if np.isnan(self.config.background_fill):
            raise ValueError("   ⚠   background_fill must be specified to replace empty image margins")        
    def process(self):
        self.image.data = self.image.raw.copy()
        self._validate()
        if self.config.align:
            self._align()
        if self.config.filter_sizes is not None:
            self._smooth_and_normalize()
                
    def _align(self):
        from sklearn.decomposition import PCA
        self.image.data[np.isnan(self.image.data) | np.isinf(self.image.data) | (self.image.data < 0)] = 0
        valid_mask = binary_fill_holes(self.image.data != 0).astype(int)
        coords = np.argwhere(valid_mask > 0)
        pca = PCA(n_components=2).fit(coords)
        angle = np.arctan2(*pca.components_[0])
        
        rotated = rotate(self.image.data, angle/np.pi*180-90, order = 1)
        rows = np.any(rotated != 0, axis=1)
        cols = np.any(rotated != 0, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        self.image.data = rotated[rmin:rmax+1, cmin:cmax+1]
    
    @staticmethod
    def multiscale_gaussian_normalized(image, sigmas):
        result = np.zeros_like(image, dtype=float)
        for sigma in sigmas:
            result += image / (gaussian_filter(image, sigma=sigma) + 1e-10)
        return result / len(sigmas)
    
    @staticmethod
    def gaussian_normalized(image, sigmas):
        results = []
        for sigma in sigmas:
            smoothed = image / (gaussian_filter(image, sigma=sigma) + 1e-10)
            results.append(smoothed)
        return  results
    
    def _smooth_and_normalize(self):
        logger.info(f"-----> pre-processing image ...")        
        self.image.data[np.isnan(self.image.data) | np.isinf(self.image.data) | (self.image.data < 0)] = 0
        '''bg_global = gaussian_filter(self.image.data, sigma = self.config.background_box)
        mask_invalid_all = (self.image.data <= 0)
        mask_invalid = binary_opening(mask_invalid_all, iterations = self.config.opening_iteration)
        mask_invalid_inside = mask_invalid_all ^ mask_invalid
        mask_outliers = binary_opening(self.image.data > bg_global, iterations = 1)
        mask_outliers = mask_outliers ^ (self.image.data > bg_global)
        self.image.data[mask_invalid_inside| mask_outliers] = bg_global[mask_invalid_inside|mask_outliers]
        self.image.data[mask_invalid] = self.config.background_fill
        
        bg_global = gaussian_filter(self.image.data, sigma = self.config.background_box)
        self.image.data = self.image.data / bg_global
        
        if self.config.local_box is not None:        
            bg_local = gaussian_filter(self.image.data, sigma = self.config.local_box)
            self.image.data = self.image.data / bg_local'''
        
        if not isinstance(self.config.filter_sizes, list) or not all(isinstance(s, int) for s in self.config.filter_sizes):
            logger.error(f"   ⚠   filter_sizes must be a list of integers, got {self.config.filter_sizes}")
            raise TypeError(f"filter_sizes must be a list of integers")
        if self.mode == 'max':
            self.image._data_filtered = ProcessingImage.gaussian_normalized(self.image.data, self.config.filter_sizes)
            self.image._n_sigmas = len(self.config.filter_sizes)
        if self.mode == 'composed':
            self.image.data = ProcessingImage.multiscale_gaussian_normalized(self.image.data, self.config.filter_sizes)
        logger.info(f"   ✔   preprocessing done.")
    
    def prepare(self):
        iy, ix = self.image.valid_pixels
        edge_index = torch.tensor(self.image.neighbors.T, dtype=torch.long)
        edge_index = to_undirected(edge_index)
        graph_data = []
        if self.image._data_filtered is not None:
            for ind in range(self.image._n_sigmas):
                intensities = self.image._data_filtered[ind][iy, ix]
                scaler = RobustScaler()
                features = scaler.fit_transform(np.log(intensities).reshape(-1, 1))
                node_features = torch.tensor(features, dtype=torch.float32)     
                graph_data.append(Data(x=node_features, edge_index=edge_index))
        else:
            intensities = self.image.data[iy, ix] 
            scaler = RobustScaler()
            features = scaler.fit_transform(np.log(intensities).reshape(-1, 1))
            node_features = torch.tensor(features, dtype=torch.float32)    
            graph_data.append(Data(x=node_features, edge_index=edge_index))
        return graph_data
        
class BubbleClassifier:
    def __init__(self, model_name: str, 
                 processing_config: Optional[Union[SnapConfig, ImageConfig]] = None,
                 device: Optional[str] = None, verbose: bool = True):
        """
        Classify bubbles using a trained GNN model.

        Parameters
        ----------
        model_name : str
            Name of trained model. Use `list_models()` to see available options.
        processing_config : SnapConfig or ImageConfig
            Configuration for preprocessing and classification:

            **SnapConfig** (for 3D simulation snapshots):
                - z_bin_size (float, default=0.05): Vertical bin size for profile normalization
                - R_bin_size (float, default=2): Radial bin size for profile normalization
                - normalize (bool, default=True): Apply vertical profile normalization (disable for non-disk geometries)
                - save_scaler (bool, default=True): Save fitted scaler for reuse
                - n_jobs (int, default=-1): Parallel jobs for batch classification
                - force_recompute (bool, default=False): Recompute even if results exist

            **ImageConfig** (for 2D observational images):
                - align (bool): Align image to minimize blank regions
                - background_fill (float, default=1.0): Background value for image margins
                - stretch (callable, optional): Non-linear stretch function applied to the image (e.g. np.arcsinh); 
                  set to None for linear scale
                - background_box (int, default=30): Box size for global background, set to None to skip global background normalisation
                - filter_size (int, default=3): Median filter size for smoothing the background grid
                - local_box (int, default=20): Box size for local background, set to None to skip global background normalisation
                - opening_iteration (int, default=10): Number of morphological opening iterations for cleaning invalid pixel masks
                - small_hole_size (int, default=1800): Minimum hole size to retain in the valid pixel mask
                - verbose (bool, default=True): Show logging info

        device : str, optional
            PyTorch device ('cuda' or 'cpu'). Auto-detected if not specified.
        verbose : bool, default=True
            Show logging info
        """
        setup_logging(verbose)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = ModelConfig(model_name=model_name,
                                  device=self.device,
                                  processing_config=processing_config)
        self.detection_threshold = processing_config.detection_threshold
        self.required_features = self.config.get_required_features()
        self._model = None
        
    @classmethod
    def list_models(cls, **kwargs):
        from .registry import ModelRegistry
        return ModelRegistry().list_models(**kwargs)
    
    @property
    def model(self):
        if self._model is None:
            logger.info(f"-----> loading model {self.config.model_name} ...")
            self._model = self.config.load_model()
            logger.info(f"   ✔   model loaded")
            logger.info(f"   ·   features: {self.required_features}")
            logger.info(f"   ·   device: {self.device}")
        return self._model
    
    def predict(self, graph_data):
        graph_data = graph_data.to(self.device)
        with torch.no_grad():
            output = self.model(graph_data)
            probs = torch.softmax(output, dim=1)
        return probs[:, 0].cpu().numpy()
    
    def predict_batch(self, graph_data):
        loader = NeighborLoader(graph_data, num_neighbors=[8, 8], batch_size=self.config.processing_config.batch_size)
        all_probs = np.zeros(graph_data.num_nodes)
        for batch in loader:
            batch = batch.to(self.device)
            with torch.no_grad():
                output = self.model(batch)
                probs = torch.sigmoid(output)
            all_probs[batch.n_id[:batch.batch_size]] = probs[:batch.batch_size].cpu().numpy().squeeze()
        return all_probs
    
    def predict_image(self, graph_data):
        prob_list = []
        for data_i in graph_data:
            probs = self.predict_batch(data_i)
            prob_list.append(probs)
        return np.max(np.stack(prob_list), axis=0)
    
    @staticmethod
    def save_results(gas_ids, gas_cfd, save_pt, snap_num, model_name):
        logger.info('●  ●  ● aerated ●  ●  ●')
        logger.info('-----> saving the bubbles ... ')
        with h5py.File(save_pt, 'w') as f:
            f.create_dataset('gas_ids', data=gas_ids, dtype='u4', 
                             compression='gzip', compression_opts = 4, chunks=True)
            f.create_dataset('confidence', data=gas_cfd, dtype='f4',
                             compression='gzip', compression_opts = 4, chunks=True)
            f.attrs['snapshot'] = snap_num
            f.attrs['model_name'] = model_name
        logger.info(f'   ✔   saved to {save_pt}')
    
    def aerate_snapshot(self, i_SnapData, save_results):
        save_pt = i_SnapData._file_config.bubble_file(i_SnapData.snap_num, self.config.model_name)
        i_SnapData.inner_results_path = Path(save_pt)
        if os.path.exists(save_pt) and not self.config.processing_config.force_recompute:
            logger.info(f"●  ●  ● aerated ●  ●  ●\n bubbles for snapshot {i_SnapData.snap_num} "
                        f" with model '{self.config.model_name}' already exist at:\n  {save_pt}\n skipping computation ...")
            with h5py.File(save_pt, 'r') as f:
                confidence = f['confidence'][()]
            i_SnapData._set_inner_bubble(confidence, self.detection_threshold)
            return 
        
        preparator = ProcessingSnapshot(i_SnapData, self.config.processing_config)
        phy_cons = preparator.prepare_features(self.required_features)        
        confidence = np.zeros(i_SnapData.num_particles)

        if not i_SnapData.no_vor_file:
            voronoi_files = i_SnapData.vor_files
            n_jobs = self.config.processing_config.n_jobs
            logger.info(f"◦ ○ ◎ aerating: {len(voronoi_files)} Voronoi chunks with {n_jobs} parallel jobs ... ◎ ○ ◦")
            
            def process_chunk(vor_file):
                graph_data, mask = ProcessingSnapshot.prepare_chunk(i_SnapData.id, phy_cons, vor_file)
                probs = self.predict(graph_data)
                return probs, mask
        
            # Batch classify bubbles for each Voronoi chunk
            results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=10)(delayed(process_chunk)(vf) for vf in voronoi_files)
            for probs, mask in results:
                confidence[mask] = probs
        else:
            logger.info(f"◦ ○ ◎ aerating ◎ ○ ◦")
            graph_data = ProcessingSnapshot.prepare(phy_cons, i_SnapData.vor)
            confidence = self.predict(graph_data)
            
        i_SnapData._set_inner_bubble(confidence, self.detection_threshold)
        if save_results:
            BubbleClassifier.save_results(i_SnapData.id, confidence, save_pt, i_SnapData.snap_num, self.config.model_name)
        return
    
    def aerate_image(self, image_data, preprocess_only = False):
        preparator = ProcessingImage(image_data, self.config.processing_config)
        preparator.process()
        if preprocess_only:
            return
        logger.info(f"◦ ○ ◎ aerating with a batch size of {self.config.processing_config.batch_size} ... ◎ ○ ◦")
        
        graph_data = preparator.prepare()
        probs = self.predict_image(graph_data)
        inner_prob_map = np.zeros(image_data.data.shape, dtype = float)
        iy, ix = image_data.valid_pixels
        inner_prob_map[iy, ix] = probs
        inner_map = inner_prob_map > self.detection_threshold
        inner_map = remove_small_holes(inner_map, area_threshold = self.config.processing_config.small_hole_size)
        image_data.inner = inner_map
        image_data.labels = inner_map.flatten()
        image_data.probs = inner_prob_map
        logger.info('●  ●  ● aerated ●  ●  ● ')
        return