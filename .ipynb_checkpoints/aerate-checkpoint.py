import time, torch, pickle, os, h5py
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from typing import Union, Optional, Dict, Any
import argparse
from math import ceil
from scipy.interpolate import RegularGridInterpolator
from sklearn.preprocessing import RobustScaler

from scipy.ndimage import rotate, binary_fill_holes, binary_opening, label, median_filter, gaussian_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.background import Background2D, MedianBackground

from utils import load_masked_snapshot
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

import logging
logger = logging.getLogger(__name__)

from config import ModelConfig, SnapConfig, ImageConfig

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
        if os.path.exists(path):
            logger.info(f"loading vertical {quantity} profile...")
            with open(path, 'rb') as f:
                interp_func = pickle.load(f)
        else:
            logger.info(f"creating vertical {quantity} profile...")
            #print(f"Creating vertical {quantity} profile...")
            mass_per_bin, _, _ = np.histogram2d(self.snap.pos_R, self.snap.pos[:, 2],
                                                bins=[R_bins, z_bins], weights=self.snap.snap_data['mass'])
            weights = data * self.snap.snap_data['mass']
            weighted_sum, _, _ = np.histogram2d(self.snap.pos_R, self.snap.pos[:, 2], 
                                                bins=[R_bins, z_bins], weights=weights)
            mean_per_bin = weighted_sum / mass_per_bin
            interp_func = RegularGridInterpolator((R_centers, z_centers), mean_per_bin, method='nearest', bounds_error=False, fill_value=None)
            if self.config.save_scaler:
                with open(path, 'wb') as f:
                    pickle.dump(interp_func, f)
        return data, interp_func, quantity
    
    def scale_quantity(self, quantity: str):
        data, profile_func, quantity = self.get_vertical_profile(quantity)
        path = self.snap._file_config.scaler_file(quantity)
        
        positions = np.c_[self.snap.pos_R, self.snap.pos[:, 2]]
        corrected = data / profile_func(positions)
        log_corrected = np.log(corrected)

        if os.path.exists(path):
            with open(path, 'rb') as f:
                logger.info(f"loading {quantity} scaler...")
                scaler = pickle.load(f)
            scaled_data = scaler.transform(np.log(corrected).reshape(-1, 1)).flatten()
        else:
            logger.info(f"creating {quantity} scaler...")
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(np.log(corrected).reshape(-1, 1)).flatten()
            if self.config.save_scaler:
                with open(path, 'wb') as f:
                    pickle.dump(scaler, f)
        logger.info(f"{quantity} scaled.")
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
    def __init__(self, i_ImageData, image_config, save_scaler=True):
        self.image = i_ImageData
        self.required_features = self.config_obj.get_required_features()
        self.config = image_config
        self._vor_pairs = None
        
    def process(self):
        if self.config.rotate:
            self._align()
        self._smooth_and_normalize()
        return self._prepare_graph_data()
    
    def _align(self):
        self.image[np.isnan(self.image)] = 0
        valid_mask = binary_fill_holes(self.image != 0).astype(int)
        coords = np.argwhere(valid_mask > 0)
        pca = PCA(n_components=2).fit(coords)
        angle = np.arctan2(*pca.components_[0])
        
        rotated = rotate(img, angle/np.pi*180-90, order = 1)
        rows = np.any(rotated != 0, axis=1)
        cols = np.any(rotated != 0, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        self.image = rotated[rmin:rmax+1, cmin:cmax+1]
        
    def _smooth_and_normalize(self):
        if self.config.background:
            smoothed = gaussian_filter(self.image, sigma = self.config.smoothing_window)
            _, median, _ = sigma_clipped_stats(self.image, sigma = self.config.stats_sigma, maxiters=5)
            use_mask = smoothed > median
            labeled, num_features = label(use_mask)
            sizes = np.bincount(labeled.ravel())[1:]
            largest_mask = labeled == (np.argmax(sizes) + 1)
            background_mask = ~binary_fill_holes(largest_mask)
            
            self.image[background_mask] = median_filter(image, size = self.config.median_window)[background_mask]

        bkg = Background2D(self.image, 
                           box_size = self.config.smoothing_window, 
                           filter_size = 3, 
                           sigma_clip = SigmaClip(sigma = self.config.stats_sigma),
                           bkg_estimator = MedianBackground())
        #median = bkg.background
        #median[median <= 0] = self.config.background_val
        median = np.clip(bkg.background, self.config.background_val, None)
        
        mask_valid = (self.image <= 0) | np.isnan(self.image)
        mask_valid = binary_opening(mask_valid, iterations = 10)

        mask_outliers = binary_opening(self.image > median, iterations = 1)
        mask_outliers = mask_outliers ^ (self.image > median)
        
        self.image = self.image / median
        self.image[mask_valid | mask_outliers] = self.config.background_val
        
    @property
    def valid_pixels(self):
        if self._valid_pixels is None:
            iy, ix = np.nonzero(self.image > 0)  # Note: nonzero returns (row, col)
            self._valid_pixels = (ix, iy)
        return self._valid_pixels
        
    @property
    def neighbors(self):
        if self._neighbors is None:
            ix, iy = self.valid_pixels
            pixel_map = {(x, y): idx for idx, (x, y) in enumerate(zip(ix, iy))}            
            offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
            edges = []
            for idx, (x, y) in enumerate(zip(ix, iy)):
                for dx, dy in offsets:
                    neighbor = (x + dx, y + dy)
                    if neighbor in pixel_map:
                        edges.append([idx, pixel_map[neighbor]])
            
            self._neighbors = np.array(edges)
        return self._neighbors
    
    def _prepare_graph_data(self):
        ix, iy = self.valid_pixels
        intensities = self.image[iy, ix] 
        scaler = RobustScaler()
        features = scaler.fit_transform(np.log(intensities).reshape(-1, 1))
        
        edge_index = torch.tensor(self.neighbors.T, dtype=torch.long)
        edge_index = to_undirected(edge_index)
        node_features = torch.tensor(features, dtype=torch.float32)        
        return Data(x=node_features, edge_index=edge_index), scaler
        
class BubbleClassifier:
    def __init__(self, model_name: str,
                 processing_config: Optional[Union[SnapConfig, ImageConfig]] = None,
                 device: Optional[str] = None,
                 save_results: bool = True):
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = ModelConfig(model_name=model_name,
                                  device=self.device,
                                  processing_config=processing_config)
        self.model = self.config.load_model()
        self.required_features = self.config.get_required_features()
        self.save_results = save_results
        self.verbose = self.config.processing_config.verbose
        
        #if self.verbose:
         #   logging.basicConfig(level=logging.INFO)
        if self.verbose:
            logging.getLogger().setLevel(logging.INFO)
            if not logging.getLogger().handlers:
                logging.getLogger().addHandler(logging.StreamHandler())
        
        print(f"✅ Loaded: {self.config.model_name}")
        print(f"   Features: {self.required_features}")
        print(f"   Device: {self.device}")
    
    @classmethod
    def list_models(cls):
        from registry import ModelRegistry
        return ModelRegistry().list_models()
    
    def predict(self, graph_data):
        graph_data = graph_data.to(self.device)
        with torch.no_grad():
            output = self.model(graph_data)
            probs = torch.softmax(output, dim=1)
        return probs[:, 0].cpu().numpy()
    
    @staticmethod
    def save_snap_inner_results(gas_ids, gas_cfd, save_pt, snap_num, model_name):
        logger.info('● ● ● aerated ● ● ● \n saving the bubbles ... ')
        with h5py.File(save_pt, 'w') as f:
            f.create_dataset('gas_ids', data=gas_ids, dtype='u4',
                             compression='gzip', compression_opts=4, chunks=True)
            f.create_dataset('confidence', data=gas_cfd, dtype='f4',
                             compression='gzip', compression_opts=4, chunks=True)
            f.attrs['snapshot'] = snap_num
            f.attrs['model_name'] = model_name
        logger.info(f'Saved to {save_pt}')
        #print(f'Saved to {save_pt}')
    
    def predict_snapshot(self, i_SnapData):
        save_pt = i_SnapData._file_config.bubble_file(i_SnapData.snap_num, self.config.model_name)
        if os.path.exists(save_pt):
            logger.info(f"● ● ● aerated ● ● ●\n bubbles for snapshot {i_SnapData.snap_num}  \n"
                        f"with model '{self.config.model_name}' already exist at:\n  {save_pt}\nSkipping computation...")
            with h5py.File(save_pt, 'r') as f:
                confidence = f['confidence'][()]
            i_SnapData._set_inner_bubble(confidence)
            return 
        preparator = ProcessingSnapshot(i_SnapData, self.config.processing_config)
        phy_cons = preparator.prepare_features(self.required_features)        
        voronoi_files = i_SnapData.vor_files
        n_jobs = self.config.processing_config.n_jobs
        logger.info(f"🫧 🫧 🫧 aerating: {len(voronoi_files)} Voronoi chunks with {n_jobs} parallel jobs... 🫧 🫧 🫧")
        def process_chunk(vor_file):
            graph_data, mask = ProcessingSnapshot.prepare_chunk(i_SnapData.id, phy_cons, vor_file)
            probs = self.predict(graph_data)
            return probs, mask
        
        results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=10)(delayed(process_chunk)(vf) for vf in voronoi_files)
        
        confidence = np.zeros(len(i_SnapData.id))
        for probs, mask in results:
            confidence[mask] = probs
        i_SnapData._set_inner_bubble(confidence)
        if self.save_results:
            BubbleClassifier.save_snap_inner_results(i_SnapData.id, confidence, save_pt, i_SnapData.snap_num, self.config.model_name)
        return 
    
    def predict_image(self, image_data):
        preparator = ProcessingImage()
        graph_data = preparator.prepare()
        probs = self.predict(graph_data)
        inner_prob_map = np.zeros(image.shape, dtype = float)
        inner_prob_map[ix, iy] = probs[:, 0]
        predicted_con =  probs[:, 0] <= 0.5
        inner_map = np.zeros(image.shape, dtype = bool)
        inner_map[ix, iy] = ~np.array(predictions, dtype = bool) 
        inner_map = remove_small_holes(inner_map, area_threshold = size**2*2)
        inner_mask = (inner_map > 0)
        ix, iy = np.nonzero(inner_mask)
        return image, inner_map, inner_prob_map, neighbors
    