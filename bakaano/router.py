"""Runoff routing utilities using pysheds.

Role: Compute flow directions and (weighted) flow accumulation.
"""

import rasterio
import pysheds.grid
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp")

class RunoffRouter:
    """
    Runoff routing utilities built on top of Pysheds.

    Role: Compute flow directions and route runoff across the DEM grid.

    This class:
    - Loads a clipped DEM and its metadata.
    - Hydrologically conditions the DEM (fill depressions, resolve flats).
    - Computes flow directions and (optionally weighted) flow accumulation.

    Typical usage:
        1) Instantiate with a DEM and routing method.
        2) Call ``compute_flow_dir()`` once to set ``self.fdir2``.
        3) Convert a runoff raster via ``convert_runoff_layers()`` and
           call ``compute_weighted_flow_accumulation()`` to route runoff.

    Notes:
    - ``compute_weighted_flow_accumulation()`` requires flow directions to be
      computed first, so call ``compute_flow_dir()`` before it.
    - Input runoff rasters must align with the DEM grid (same extent,
      resolution, and CRS) or routing results will be incorrect.
    """
    
    def __init__(self, working_dir, dem, routing_method):
        """
        Initialize the RunoffRouter object.
        
        Args:
            working_dir (str): Working directory for scratch outputs.
            dem (str): Path to clipped DEM GeoTIFF.
            routing_method (str): Routing method (e.g., ``"mfd"``, ``"d8"``, ``"dinf"``).
        """
        self.working_dir = working_dir
        self.grid = None
        self.dem_filepath = dem
        self.inflated_dem = None
        self.routing_method = routing_method

        with rasterio.open(self.dem_filepath) as dm:
            self.dem_ras = dm.read(1)
            self.dem_profile = dm.profile
            self.dem_nodata = dm.nodata
            
    
    def convert_runoff_layers(self, runoff_array):
        """
        Convert a runoff array to a GeoTIFF aligned to the DEM grid.

        The returned raster is written to ``{working_dir}/scratch/runoff_scratch.tif``
        using the DEM's metadata (extent, resolution, CRS).
        
        Parameters:
        -----------
        runoff_array : numpy array
            Array containing the simulated runoff. Must match the DEM grid shape.
        
        Returns:
            str: Path to the temporary runoff GeoTIFF.
        """

        profile = self.dem_profile.copy()
        profile.update(dtype=rasterio.float32, count=1)
        # GDAL only accepts BLOCKXSIZE/BLOCKYSIZE when TILED=YES.
        if not profile.get("tiled", False):
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)
        runoff_tiff = f'{self.working_dir}/scratch/runoff_scratch.tif'
        with rasterio.open(runoff_tiff, 'w', **profile) as dst:
            dst.write(runoff_array, 1)
                
        return runoff_tiff
    
    def fill_dem(self):
        """Fill depressions and resolve flats in the DEM.

        This hydrologically conditions the DEM to improve flow routing.

        Returns:
            np.ndarray: Hydrologically conditioned DEM array.
        """
        self.grid = pysheds.grid.Grid.from_raster(self.dem_filepath, nodata=self.dem_nodata)
        dem = self.grid.read_raster(self.dem_filepath, nodata=self.dem_nodata)
        
        flooded_dem = self.grid.fill_depressions(dem)

        # Resolve flats
        inflated_dem = self.grid.resolve_flats(flooded_dem)
        return inflated_dem

    def compute_flow_dir(self):
        """
        Compute flow directions and unweighted flow accumulation.

        Returns:
            tuple[np.ndarray, np.ndarray]: (flow_directions, accumulation)
        """
        inflated_dem = self.fill_dem()
        self.fdir2 = self.grid.flowdir(inflated_dem, routing=self.routing_method)
        acc = self.grid.accumulation(fdir=self.fdir2, routing=self.routing_method)
        return self.fdir2, acc
            
    def compute_weighted_flow_accumulation(self, runoff_tiff):
        """Compute weighted flow accumulation using a runoff raster.

        Args:
            runoff_tiff (str): Path to a runoff GeoTIFF aligned to the DEM grid.

        Returns:
            np.ndarray: Weighted flow accumulation array.
        """
        
        weight = self.grid.read_raster(runoff_tiff)
        wacc = self.grid.accumulation(fdir=self.fdir2, weights=weight, routing=self.routing_method)
        return wacc
 
