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
        
class ProcessingImage():
    def __init__(self, i_ImageData, image_config):
        self.image = i_ImageData
        self.config = image_config
        
    def process(self):
        self.image.data = self.image.raw.copy()
        if self.config.rotate:
            self._align()
        self._smooth_and_normalize()
    
    def _align(self):
        from sklearn.decomposition import PCA
        self.image.data[np.isnan(self.image.data)] = 0
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
        
    def _smooth_and_normalize(self):
        logger.info(f"-----> prep-rocessing image ...")
        from photutils.background import Background2D, MedianBackground
        from astropy.stats import sigma_clipped_stats, SigmaClip
        if self.config.background:
            smoothed = gaussian_filter(self.image.data, sigma = self.config.smoothing_window)
            _, median, _ = sigma_clipped_stats(self.image.data, sigma = self.config.stats_sigma, maxiters = 5)
            use_mask = smoothed > median
            labeled, num_features = label(use_mask)
            sizes = np.bincount(labeled.ravel())[1:]
            largest_mask = labeled == (np.argmax(sizes) + 1)
            background_mask = ~binary_fill_holes(largest_mask)
            
            self.image.data[background_mask] = median_filter(self.image.data, size = self.config.median_window)[background_mask]

        bkg = Background2D(self.image.data, 
                           box_size = self.config.smoothing_window, 
                           filter_size = 3, 
                           sigma_clip = SigmaClip(sigma = self.config.stats_sigma),
                           mask = np.isnan(self.image.data),
                           bkg_estimator = MedianBackground())
        median = np.clip(bkg.background, self.config.background_val, None)
        
        mask_invalid = (self.image.data <= 0) | np.isnan(self.image.data)
        mask_invalid = binary_opening(mask_invalid, iterations = self.config.opening_iteration)

        mask_outliers = binary_opening(self.image.data > median, iterations = 1)
        mask_outliers = mask_outliers ^ (self.image.data > median)
        
        self.image.data = self.image.data / median
        self.image.data[mask_invalid | mask_outliers] = self.config.background_val
        logger.info(f"   ✔   preprocessing done.")
    
    def prepare(self):
        iy, ix = self.image.valid_pixels
        intensities = self.image.data[iy, ix] 
        scaler = RobustScaler()
        features = scaler.fit_transform(np.log(intensities).reshape(-1, 1))
        edge_index = torch.tensor(self.image.neighbors.T, dtype=torch.long)
        edge_index = to_undirected(edge_index)
        node_features = torch.tensor(features, dtype=torch.float32)        
        return Data(x=node_features, edge_index=edge_index)
        
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
                - rotate (bool): Align image to minimize blank regions
                - background (bool): Subtract background
                - background_val (float, default=1.0): Background value
                - smoothing_window (int, default=30): Smoothing kernel size
                - median_window (int, default=20): Median filter size
                - stats_sigma (int, default=5): Sigma for clipping statistics
                - verbose (bool, default=True): Show logging info
                (Additional parameters to be documented)

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
                probs = torch.softmax(output, dim=1)
            all_probs[batch.n_id[:batch.batch_size]] = probs[:batch.batch_size, 0].cpu().numpy()
        return all_probs
    
    @staticmethod
    def save_results(gas_ids, gas_cfd, save_pt, snap_num, model_name):
        logger.info('●  ●  ● aerated ●  ●  ● \n-----> saving the bubbles ... ')
        with h5py.File(save_pt, 'w') as f:
            f.create_dataset('gas_ids', data=gas_ids, dtype='u4', 
                             compression='gzip', compression_opts = 4, chunks=True)
            f.create_dataset('confidence', data=gas_cfd, dtype='f4',
                             compression='gzip', compression_opts = 4, chunks=True)
            f.attrs['snapshot'] = snap_num
            f.attrs['model_name'] = model_name
        logger.info(f'   ✔    saved to {save_pt}')
    
    def aerate_snapshot(self, i_SnapData, save_results):
        save_pt = i_SnapData._file_config.bubble_file(i_SnapData.snap_num, self.config.model_name)
        i_SnapData.inner_results_path = Path(save_pt)
        if os.path.exists(save_pt) and not self.config.processing_config.force_recompute:
            logger.info(f"●  ●  ● aerated ●  ●  ●\n bubbles for snapshot {i_SnapData.snap_num} "
                        f" with model '{self.config.model_name}' already exist at:\n  {save_pt}\n skipping computation ...")
            with h5py.File(save_pt, 'r') as f:
                confidence = f['confidence'][()]
            i_SnapData._set_inner_bubble(confidence)
            return 
        preparator = ProcessingSnapshot(i_SnapData, self.config.processing_config)
        phy_cons = preparator.prepare_features(self.required_features)        
        voronoi_files = i_SnapData.vor_files
        n_jobs = self.config.processing_config.n_jobs
        logger.info(f"◦ ○ ◎ ◉ ● aerating: {len(voronoi_files)} Voronoi chunks with {n_jobs} parallel jobs ... ● ◉ ◎ ○ ◦")
        def process_chunk(vor_file):
            graph_data, mask = ProcessingSnapshot.prepare_chunk(i_SnapData.id, phy_cons, vor_file)
            probs = self.predict(graph_data)
            return probs, mask
        
        # Batch classify bubbles for each Voronoi chunk
        results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=10)(delayed(process_chunk)(vf) for vf in voronoi_files)
        
        confidence = np.zeros(len(i_SnapData.id))
        for probs, mask in results:
            confidence[mask] = probs
        i_SnapData._set_inner_bubble(confidence)
        if save_results:
            BubbleClassifier.save_results(i_SnapData.id, confidence, save_pt, i_SnapData.snap_num, self.config.model_name)
        return
    
    def aerate_image(self, image_data, preprocess_only = False):
        preparator = ProcessingImage(image_data, self.config.processing_config)
        preparator.process()
        if preprocess_only:
            return
        logger.info(f"◦ ○ ◎ ◉ ● aerating with a batch size of {self.config.processing_config.batch_size} ... ● ◉ ◎ ○ ◦")
        graph_data = preparator.prepare()
        probs = self.predict_batch(graph_data)
        inner_prob_map = np.zeros(image_data.data.shape, dtype = float)
        iy, ix = image_data.valid_pixels
        inner_prob_map[iy, ix] = probs
        inner_map = inner_prob_map > 0.5
        inner_map = remove_small_holes(inner_map, area_threshold = self.config.processing_config.small_hole_size)
        image_data.inner = inner_map
        image_data.labels = inner_map.flatten()
        image_data.probs = inner_prob_map
        logger.info('●  ●  ● aerated ●  ●  ● ')
        return