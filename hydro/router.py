import rasterio
import pysheds.grid
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp")

class RunoffRouter:
    """
    A class that performs runoff routing using Pysheds library.
    """
    
    def __init__(self, project_name, dem):
        """
        Initialize the RunoffRouter object.
        
        Parameters:
        -----------
        datadir : str
            Directory path of the data.
        """
        self.project_name = project_name
        self.grid = None
        self.dem_filepath = dem
        self.inflated_dem = None

        with rasterio.open(self.dem_filepath) as dm:
            self.dem_ras = dm.read(1)
            self.dem_profile = dm.profile
            self.dem_nodata = dm.nodata
            
    
    def convert_runoff_layers(self, runoff_array, runoff_out_name):
        """
        Convert simulated daily runoff numpy array to geotiff files, the format required by Pysheds module for weighted flow accumulation computation.
        
        Parameters:
        -----------
        runoff_array : numpy array
            Array containing the simulated runoff.
        swc_filename : str
            Filepath of an existing raster/geotiff template with the desired extent, coordinate, resolution and data type
        """

        self.dem_profile.update(dtype=rasterio.float32, count=1)
        with rasterio.open(runoff_out_name, 'w', **self.dem_profile) as dst:
                dst.write(runoff_array, 1)
                
        return runoff_out_name
    
    def fill_dem(self):
        self.grid = pysheds.grid.Grid.from_raster(self.dem_filepath)
        dem = self.grid.read_raster(self.dem_filepath)
        
        flooded_dem = self.grid.fill_depressions(dem)

        # Resolve flats
        inflated_dem = self.grid.resolve_flats(flooded_dem)
        return inflated_dem

    def compute_flow_dir(self):
        """
        Compute the weighted flow accumulation.

        Parameters:
        -----------
        dem_filename : str, optional
            Filename of the digital elevation model raster. If not provided, it will be computed.
        fdir_filename : str, optional
            Filename of the flow direction raster. If not provided, it will be computed.
        """
        inflated_dem = self.fill_dem()
        self.fdir2 = self.grid.flowdir(inflated_dem, routing='mfd')
        acc = self.grid.accumulation(fdir=self.fdir2, routing='mfd')
        return self.fdir2, acc
            
    def compute_weighted_flow_accumulation(self, runoff_tiff):
        
        weight = self.grid.read_raster(runoff_tiff)
        wacc = self.grid.accumulation(fdir=self.fdir2, weights=weight, routing='mfd')
        return wacc
 
