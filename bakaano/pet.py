import numpy as np
from bakaano.utils import Utils


class PotentialEvapotranspiration:
    def __init__(self, project_name, study_area, start_date, end_date):
        self.study_area = study_area

        self.start_date = start_date
        self.end_date = end_date
        self.uw = Utils(project_name, self.study_area)
        self.uw.get_bbox('EPSG:4326')
    
    def compute_PET(self, day_pet_params, latgrids, tmean):
        
        # Convert latgrids to CuPy array and calculate radians
        p1 = np.radians(latgrids)

        # Extract day of the year and convert to CuPy array
        
        dayNum = tmean['time'].dt.dayofyear
        dayNum = dayNum.values

        # Calculate solar radiation components using CuPy
        p2 = 1 + 0.033 * np.cos((2 * np.pi * dayNum) / 365)
        p3 = 0.409 * np.sin(((2 * np.pi * dayNum) / 365) - 1.39)  # sol_dec
        p4 = np.arccos(-1 * np.tan(p1) * np.tan(p3))
        p5 = np.sin(p1) * np.sin(p3)
        p6 = np.cos(p1) * np.cos(p3)
        Ra = ((24 * 60) / np.pi) * 0.0820 * p2 * ((p4 * p5) + p6 * np.sin(p4))

        pet = day_pet_params * Ra  # units in mm/day
        return pet
        
    

