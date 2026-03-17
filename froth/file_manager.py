from dataclasses import dataclass
import os, re
import numpy as np
from pathlib import Path
from typing import Optional

@dataclass
class PathsConfig:
    snap_path: Path
    vor_path: Optional[Path] = None
    scaler_path: Optional[Path] = None
    bubble_res_path: Optional[Path] = None
    num_of_vor_file: int = None
    inner_file_suffix: str = 'inner'
    
    def __post_init__(self):
        self.snap_path = Path(self.snap_path)        
        if self.vor_path is None:
            self.vor_path = self.snap_path / "voronoi"
        if self.scaler_path is None:
            self.scaler_path = self.snap_path / "scalers"
        if self.bubble_res_path is None:
            self.bubble_res_path = self.snap_path / "bubbles"
        
        self.vor_path = Path(self.vor_path)
        self.scaler_path = Path(self.scaler_path)
        self.bubble_res_path = Path(self.bubble_res_path)
    
    def create_directories(self):
        for path in [self.vor_path, self.scaler_path, self.bubble_res_path]:
            path.mkdir(parents=True, exist_ok=True)
            
    @classmethod
    def from_yaml(cls, yaml_path):
        import yaml
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"   ⚠   Config file not found: {yaml_path} \n"
                f"Create one using: froth.create_default_config()")
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get('paths', {}))

class FileManager:
    def __init__(self, paths: PathsConfig):
        self.paths = paths
        
    def profile_file(self, quantity: str) -> str:
        filename = f"mass_weighted_{quantity}_profile.pkl"
        return os.path.join(self.paths.scaler_path, filename)

    def scaler_file(self, quantity: str) -> str:
        filename = f"{quantity}_scaler.pkl"
        return os.path.join(self.paths.scaler_path, filename)
    
    def snapshot_file(self, snap_num):
        fname = 'snap_%s.hdf5'%(snap_num)
        return os.path.join(self.paths.snap_path, fname)
    
    def bubble_file(self, snap_num, model_name):
        fname = f"bubbles_{snap_num}_{self.paths.inner_file_suffix}_{model_name}.h5"
        return os.path.join(self.paths.bubble_res_path, fname)
    
    @staticmethod
    def parse_vor_filename(fname):
        pattern = r"(\d+)_(\d+)_.*_sec_vor\.npz"
        m = re.match(pattern, fname)
        if not m:
            raise ValueError(f"   ⚠   Invalid filename: {fname} ¡ ¡ ¡")
        snap_num, section_idx = m.groups()
        return int(snap_num), int(section_idx)
    
    def _validate_vor_files(self, file_paths):
        if not file_paths:
            return False

        pattern = r"\d+_\d+_([\d.\-]+)_([\d.\-]+)_sec_vor\.npz"
        phi_ranges = []
        for f in file_paths:
            m = re.search(pattern, os.path.basename(f))
            if m:
                phi_start = float(m.group(1)) * np.pi
                phi_end = float(m.group(2)) * np.pi
                phi_ranges.append((phi_start, phi_end))
        phi_ranges.sort()
        if not np.isclose(phi_ranges[0][0], -np.pi, atol=1e-6):
            return False
        if not np.isclose(phi_ranges[-1][1], np.pi, atol=1e-6):
            return False

        for i in range(len(phi_ranges) - 1):
            if not np.isclose(phi_ranges[i][1], phi_ranges[i+1][0], atol=1e-6):
                return False
        return True

    def vor_file(self, snap_num):
        vor_save_path = self.paths.vor_path
        files = [i for i in os.listdir(vor_save_path) if str(snap_num) in i]
        files_sorted = sorted(files, key=lambda f: FileManager.parse_vor_filename(f)[1])
        file_paths = [os.path.join(vor_save_path, f) for f in files_sorted]

        if self.paths.num_of_vor_file is None:
            return file_paths
        return file_paths[:self.paths.num_of_vor_file]