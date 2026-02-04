"""Potential evapotranspiration helper functions.

Role: Compute daily PET parameters for VegET.
"""

import numpy as np
from bakaano.utils import Utils

class PotentialEvapotranspiration:
    def __init__(self, project_name, study_area, start_date, end_date):
        """Initialize PET helper for a study area and period.

        Role: Compute daily PET based on temperature inputs.

        Args:
            project_name (str): Working directory for outputs.
            study_area (str): Path to the basin/watershed shapefile.
            start_date (str): Start date (YYYY-MM-DD).
            end_date (str): End date (YYYY-MM-DD).
        """
        self.study_area = study_area

        self.start_date = start_date
        self.end_date = end_date
        self.uw = Utils(project_name, self.study_area)
        self.uw.get_bbox('EPSG:4326')
    
    def compute_PET(self, day_pet_params, tan_lat, cos_lat, sin_lat, doy):
        """Compute daily potential evapotranspiration (Hargreaves-style).

        Args:
            day_pet_params (np.ndarray): Daily PET parameters array.
            tan_lat (np.ndarray): Tangent of latitude grid (radians).
            cos_lat (np.ndarray): Cosine of latitude grid (radians).
            sin_lat (np.ndarray): Sine of latitude grid (radians).
            doy (int): Day of year (1-366).

        Returns:
            np.ndarray: PET array for the day.
        """
        
        p2 = 1 + 0.033 * np.cos((2 * np.pi * doy) / 365)
        p3 = 0.409 * np.sin(((2 * np.pi * doy) / 365) - 1.39)
        
        # grid-based radiation terms (only 1 heavy operation)
        p4 = np.arccos(-tan_lat * np.tan(p3))   # only heavy grid-operation per day
        p5 = sin_lat * np.sin(p3)
        p6 = cos_lat * np.cos(p3)
        
        Ra = ((24 * 60) / np.pi) * 0.0820 * p2 * ((p4 * p5) + p6 * np.sin(p4))
        this_et = day_pet_params * Ra
        return this_et
        
    
