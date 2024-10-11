from pathlib import Path
import sys
import requests
import geopandas as gpd
#from tqdm.auto import tqdm
from argparse import ArgumentParser
from shapely.geometry import Polygon

class WorldCoverDownloader:
    def __init__(self, output_folder='.', country=None, bounds=None, year=2021, overwrite=False, dryrun=False):
        self.output_folder = Path(output_folder)
        self.year = year
        self.country = country
        self.bounds = bounds
        self.overwrite = overwrite
        self.dryrun = dryrun
        self.version = {2020: 'v100', 2021: 'v200'}[year]
        self.s3_url_prefix = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
        self.ne = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        self.grid_url = 'https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/esa_worldcover_2020_grid.geojson'

    def _get_selected_geom(self):
        geom = None
        if self.country is not None:
            if self.country not in self.ne.name.values:
                sys.exit()
            geom = self.ne[self.ne.name == self.country].iloc[0].geometry

        if self.bounds is not None:
            geom_bounds = Polygon.from_bounds(*self.bounds)
            if geom is None:
                geom = geom_bounds
            else:
                geom = geom.intersection(geom_bounds)
        return geom

    def download(self):
        grid = gpd.read_file(self.grid_url)
        geom = self._get_selected_geom()

        if geom is not None:
            tiles = grid[grid.intersects(geom)]
        else:
            tiles = grid

        if tiles.shape[0] == 0:
            sys.exit()

        for tile in tiles.ll_tile:
            url = self.s3_url_prefix + "/" + self.version + "/" + str(self.year) + "/map/ESA_WorldCover_10m_" + str(self.year) + "_" + self.version + "_" + tile + "_Map.tif"
            out_fn = self.output_folder / ("ESA_WorldCover_10m_" + str(self.year) + "_" + self.version + "_" + tile + "_Map.tif")

            if out_fn.is_file() and not self.overwrite:
                continue

            if not self.dryrun:
                r = requests.get(url, allow_redirects=True)
                with open(out_fn, 'wb') as f:
                    f.write(r.content)
            else:
                print('Downloading ' + url + ' to ' + str(out_fn))

if __name__ == '__main__':
    parser = ArgumentParser(description="ESA WorldCover download helper")
    parser.add_argument('-o', '--output', default='.', help="Output folder path, defaults to current folder.")
    parser.add_argument('-c', '--country', help="Optionally specify a country")
    parser.add_argument('-b', '--bounds', nargs=4, type=float, help="Optionally specify a set of lat lon bounds (4 values: xmin ymin xmax ymax)")
    parser.add_argument('-y', '--year', default=2021, type=int, choices=[2020, 2021], help="Map year, defaults to the most recent 2021 map")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing files")
    parser.add_argument('--dry', action='store_true', help="Perform a dry run")
    args = parser.parse_args()

    downloader = WorldCoverDownloader(args.output, args.country, args.bounds, args.year, args.overwrite, args.dry)
    downloader.download()
