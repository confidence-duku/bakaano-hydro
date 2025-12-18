
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from bakaano.utils import Utils

class RoutedRunoff:
    def __init__(self, working_dir, study_area):
        self.working_dir = working_dir
        self.study_area = study_area
        self.uw = Utils(self.working_dir, self.study_area)
        self.out_path = f'{self.working_dir}/elevation/dem_clipped.tif'


    def map_routed_runoff(self, date, vmax=8):
        # Function to map routed runoff for a specific date
        data = sorted(glob.glob(f'{self.working_dir}/runoff_output/*.pkl'))[0]
        if os.path.exists(data) is False:
            raise FileNotFoundError("Routed runoff output directory not found. Please run veget module first.")
        else:
            with open(data, 'rb') as f:
                wfa_list = pickle.load(f)

        entry = next((item for item in wfa_list if item['time'] == date), None)
        del wfa_list

        if entry is None:
            raise ValueError(f"No matrix found for date {date}")

        # Extract sparse matrix and convert to dense
        mat = entry['matrix'].toarray()

        dem_data = self.uw.clip(raster_path=self.out_path, out_path=None, save_output=False, crop_type=True)[0]
        dem_data = np.where(dem_data > 0, 1, np.nan)
        dem_data = np.where(dem_data < 32000, 1, np.nan)

        ro = dem_data * mat
        # Plot
        plt.figure(figsize=(7, 5))
        plt.imshow(np.log1p(ro), cmap='viridis', vmax=vmax)
        plt.colorbar(label='Value')
        plt.title(f"Routed runoff for {date}")
        plt.xlabel("X index")
        plt.ylabel("Y index")
        plt.show()


