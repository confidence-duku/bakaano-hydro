"""Runoff generation and routing using the VegET formulation.

Role: Compute daily runoff and route flow to river network.
"""

import numpy as np
import pandas as pd
import os
import shutil
from datetime import datetime
from bakaano.core.utils import Utils
from bakaano.hydrology.pet import PotentialEvapotranspiration
import pickle
import scipy as sp
from bakaano.data.meteo import Meteo
from tqdm import tqdm
from datetime import datetime, timedelta

from numba import njit, prange


def _load_runoff_router():
    """Import RunoffRouter only when routing is actually requested."""
    from bakaano.hydrology.router import RunoffRouter

    return RunoffRouter


@njit(parallel=True, fastmath=True)
def update_soil_and_runoff(soil_moisture, eff_rain, ETa, max_allowable_depletion, whc):
    """
    Corrected Numba-optimized soil moisture and runoff update.
    All inputs must be float32 NumPy arrays (2D).

    Args:
        soil_moisture (np.ndarray): Current soil moisture grid (ny, nx).
        eff_rain (np.ndarray): Effective rainfall grid (ny, nx).
        ETa (np.ndarray): Actual evapotranspiration grid (ny, nx).
        max_allowable_depletion (np.ndarray): Max allowable depletion grid.
        whc (np.ndarray): Water holding capacity grid.

    Returns:
        tuple[np.ndarray, np.ndarray]: (updated soil_moisture, surface runoff).
    """
    ny, nx = soil_moisture.shape
    q_surf = np.empty((ny, nx), dtype=np.float32)

    for i in prange(ny):
        for j in prange(nx):

            # soil update
            sm = soil_moisture[i, j] + eff_rain[i, j] - ETa[i, j]

            # no negative soil moisture
            if sm < 0:
                sm = 0.0

            # compute runoff (excess water)
            excess = sm - whc[i, j]

            if excess > 0.0:
                q_surf[i, j] = excess        # runoff is excess
                sm = whc[i, j]               # enforce WHC cap (critical!)
            else:
                q_surf[i, j] = 0.0

            # save updated soil moisture
            soil_moisture[i, j] = sm

    return soil_moisture, q_surf

class VegET:
    """Role: Orchestrate VegET runoff generation and routing."""
    def __init__(
        self,
        working_dir,
        study_area,
        start_date,
        end_date,
        climate_data_source,
        routing_method='mfd',
        runoff_output_dir=None,
        tree_cover_tiff=None,
        herb_cover_tiff=None,
        ndvi_pickle_path=None,
    ):
        """Initialize a VegET object.

        Args:
            working_dir (str): The parent working directory where files and outputs will be stored.
            study_area_path (str): The path to the shapefile of the river basin or watershed.
            start_date (str): The start date of the simulation period in YYYY-MM-DD format
            end_date (str): The end date of the simulation period in YYYY-MM-DD format
            climate_data_source (str): The source of climate data. Options are 'CHELSA', 'ERA5', or 'CHIRPS'.
            routing_method (str): The method used for routing runoff. Options are 'mfd', 'd8' or 'dinf'. Default is 'mfd'.

        Methods
        -------
        __init__(working_dir, study_area_path, start_date, end_date, climate_data_source):
            Initializes the VegET object with project details.
        compute_veget_runoff_route_flow(prep_nc, tasmax_nc, tasmin_nc, tmean_nc):
            Computes the vegetation evapotranspiration and runoff routing flow.
        """
         # Initialize the project name
        self.working_dir = working_dir
        
        # Initialize the study area
        self.study_area = study_area
        
        # Initialize utility class with project name and study area.
        self.uw = Utils(self.working_dir, self.study_area)
        self.times = pd.date_range(start_date, end_date)
        
        # Set the start and end dates for the project
        self.start_date = start_date
        self.end_date = end_date
        self.routing_method = routing_method

        # Create necessary directories for the project structure   
        os.makedirs(f'{self.working_dir}/models', exist_ok=True)
        os.makedirs(f'{self.working_dir}/runoff_output', exist_ok=True)
        os.makedirs(f'{self.working_dir}/scratch', exist_ok=True)
        os.makedirs(f'{self.working_dir}/shapes', exist_ok=True)
        os.makedirs(f'{self.working_dir}/catchment', exist_ok=True)

        self.clipped_dem = f'{self.working_dir}/elevation/dem_clipped.tif'
        self.climate_data_source = climate_data_source
        self.runoff_output_dir = runoff_output_dir or f'{self.working_dir}/runoff_output'
        self.tree_cover_tiff = tree_cover_tiff or f'{self.working_dir}/vcf/mean_tree_cover.tif'
        self.herb_cover_tiff = herb_cover_tiff or f'{self.working_dir}/vcf/mean_herb_cover.tif'
        self.ndvi_pickle_path = ndvi_pickle_path or f'{self.working_dir}/ndvi/daily_ndvi_climatology.pkl'
        os.makedirs(self.runoff_output_dir, exist_ok=True)

    def _validate_climate_data_source(self):
        """Raise a clear error for unsupported climate input sources."""
        valid_sources = {"CHELSA", "ERA5", "CHIRPS"}
        if self.climate_data_source not in valid_sources:
            valid_str = ", ".join(sorted(valid_sources))
            raise ValueError(
                f"Unsupported climate_data_source '{self.climate_data_source}'. "
                f"Expected one of: {valid_str}."
            )

    def _resume_signature(self):
        """Return a lightweight signature used to validate resume state."""
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            **self._extension_signature(),
        }

    def _extension_signature(self):
        """Return inputs that must stay fixed for a valid extension run."""
        return {
            "routing_method": self.routing_method,
            "climate_data_source": self.climate_data_source,
            "study_area": str(self.study_area),
            "tree_cover_tiff": os.path.abspath(self.tree_cover_tiff),
            "herb_cover_tiff": os.path.abspath(self.herb_cover_tiff),
            "ndvi_pickle_path": os.path.abspath(self.ndvi_pickle_path),
        }

    def _load_resume_state(self, state_file, expected_signature):
        """Load and validate resume state. Returns dict or None."""
        if not os.path.exists(state_file):
            return None
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        if state.get("signature") != expected_signature:
            return None
        return state

    def _save_resume_state(self, state_file, signature, next_index, soil_moisture):
        """Persist resume state after checkpoint flush."""
        state = {
            "signature": signature,
            "next_index": next_index,
            "soil_moisture": np.asarray(soil_moisture, dtype=np.float32),
        }
        with open(state_file, "wb") as f:
            pickle.dump(state, f)

    def _load_output_metadata(self, metadata_file):
        """Load completed-run metadata if it exists."""
        if not os.path.exists(metadata_file):
            return None
        with open(metadata_file, "rb") as f:
            return pickle.load(f)

    def _save_output_metadata(self, metadata_file, completed_end_date, soil_moisture):
        """Persist enough state to extend a completed runoff file later."""
        metadata = {
            "start_date": self.start_date,
            "end_date": completed_end_date,
            "last_completed_date": completed_end_date,
            "signature": self._extension_signature(),
            "soil_moisture": np.asarray(soil_moisture, dtype=np.float32),
        }
        with open(metadata_file, "wb") as f:
            pickle.dump(metadata, f)

    def compute_veget_runoff_route_flow(self, resume=True, checkpoint_days=30):  
        """Compute VegET runoff and route flow to the river network.

        This routine loads climate inputs, computes PET, simulates soil moisture
        and runoff, and performs routing to produce daily routed runoff outputs.

        Args:
            resume (bool): Resume from checkpoint state if available.
            checkpoint_days (int): Number of simulated days between checkpoints.

        Returns:
            None. Writes routed runoff outputs to ``{working_dir}/runoff_output``.
        """
        if checkpoint_days < 1:
            raise ValueError("checkpoint_days must be >= 1.")
        self._validate_climate_data_source()

        final_file = f'{self.runoff_output_dir}/wacc_sparse_arrays.pkl'
        metadata_file = f'{self.runoff_output_dir}/wacc_output_metadata.pkl'
        state_file = f'{self.runoff_output_dir}/wacc_resume_state.pkl'
        chunks_dir = f'{self.runoff_output_dir}/wacc_resume_chunks'
        os.makedirs(chunks_dir, exist_ok=True)
        signature = self._resume_signature()
        existing_output = os.path.exists(final_file)
        output_metadata = self._load_output_metadata(metadata_file) if existing_output else None

        extension_mode = False
        existing_wacc_list = []
        existing_end_dt = None

        if existing_output:
            if output_metadata is None:
                with open(final_file, "rb") as f:
                    existing_wacc_list = pickle.load(f)
                if existing_wacc_list:
                    existing_start = existing_wacc_list[0].get("time")
                    existing_end = existing_wacc_list[-1].get("time")
                    if self.start_date == existing_start and self.end_date == existing_end:
                        print(f'Routed runoff data already covers {existing_start} to {existing_end} in {final_file}. Skipping processing')
                        return
                raise ValueError(
                    "Existing runoff output does not include metadata required to validate or extend it. "
                    "Delete the existing runoff output and rerun the full date range once to enable future extensions."
                )

            existing_start = output_metadata.get("start_date")
            existing_end = output_metadata.get("end_date")
            existing_sig = output_metadata.get("signature")

            if existing_sig != self._extension_signature():
                raise ValueError(
                    "Existing runoff output was generated with different VegET inputs. "
                    "Use a different runoff_output_dir or delete the existing runoff output before rerunning."
                )

            if self.start_date != existing_start:
                raise ValueError(
                    "VegET extension only supports the original start_date. "
                    f"Existing output starts on {existing_start}, but the requested run starts on {self.start_date}."
                )

            existing_end_dt = datetime.strptime(existing_end, "%Y-%m-%d")
            requested_end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
            if requested_end_dt < existing_end_dt:
                raise ValueError(
                    "VegET cannot shorten an existing completed runoff output. "
                    f"Existing end_date is {existing_end}."
                )
            if requested_end_dt == existing_end_dt:
                print(f'Routed runoff data already covers {existing_start} to {existing_end} in {final_file}. Skipping processing')
                return

            if "soil_moisture" not in output_metadata:
                raise ValueError(
                    "Existing runoff output does not include the saved terminal state required for extension. "
                    "Delete the existing runoff output and rerun the full date range once to enable future extensions."
                )

            with open(final_file, "rb") as f:
                existing_wacc_list = pickle.load(f)
            extension_mode = True
            print(
                "Extending VegET runoff output from "
                f"{existing_end_dt.strftime('%Y-%m-%d')} to {self.end_date}"
            )
        else:
            print('Computing VegET runoff and routing flow to river network')
            if resume:
                print(
                    "VegET resume mode is enabled. Matching checkpoints are written "
                    f"every {checkpoint_days} simulated day(s)."
                )

        eto = PotentialEvapotranspiration(self.working_dir, self.study_area, self.start_date, self.end_date)

        cd = Meteo(
            self.working_dir,
            self.study_area,
            start_date=self.start_date,
            end_date=self.end_date,
            local_data=False,
            data_source=self.climate_data_source,
            local_prep_path=None,
            local_tasmax_path=None,
            local_tasmin_path=None,
            local_tmean_path=None,
        )
        prep_nc, tasmax_nc, tasmin_nc, tmean_nc = cd.get_meteo_data()

        if self.climate_data_source == 'CHELSA':
            tasmax_period = tasmax_nc.tasmax.sel(time=slice(self.start_date, self.end_date)) - 273.15
            tasmin_period = tasmin_nc.tasmin.sel(time=slice(self.start_date, self.end_date)) - 273.15
            tmean_period = tmean_nc.tas.sel(time=slice(self.start_date, self.end_date)) - 273.15
            rf = prep_nc.pr.sel(time=slice(self.start_date, self.end_date)) * 86400
            rf = rf.astype(np.float32).assign_coords(lat=rf['lat'].astype(np.float32), lon=rf['lon'].astype(np.float32))
            self.rf = rf
        elif self.climate_data_source == 'ERA5':
            tasmax_period = tasmax_nc.tasmax.sel(time=slice(self.start_date, self.end_date)) - 273.15
            tasmin_period = tasmin_nc.tasmin.sel(time=slice(self.start_date, self.end_date)) - 273.15
            tmean_period = tmean_nc.tas.sel(time=slice(self.start_date, self.end_date)) - 273.15
            rf = prep_nc.pr.sel(time=slice(self.start_date, self.end_date)) * 1000
            rf = rf.astype(np.float32).assign_coords(lat=rf['lat'].astype(np.float32), lon=rf['lon'].astype(np.float32))
            self.rf = rf
        elif self.climate_data_source == 'CHIRPS':
            tasmax_period = tasmax_nc.tasmax.sel(time=slice(self.start_date, self.end_date)) - 273.15
            tasmin_period = tasmin_nc.tasmin.sel(time=slice(self.start_date, self.end_date)) - 273.15
            tmean_period = tmean_nc.tas.sel(time=slice(self.start_date, self.end_date)) - 273.15
            rf = prep_nc.pr.sel(time=slice(self.start_date, self.end_date))
            rf = rf.astype(np.float32).assign_coords(lat=rf['lat'].astype(np.float32), lon=rf['lon'].astype(np.float32))
            self.rf = rf

        td = np.sqrt(tasmax_period - tasmin_period)
        pet_params = 0.408 * 0.0023 * (tmean_period + 17.8) * td
        pet_params = pet_params.astype(np.float32)
        self.pet_params = pet_params.assign_coords(
            lat=pet_params['lat'].astype(np.float32),
            lon=pet_params['lon'].astype(np.float32)
        )

        latsg = tmean_period[0]['lat']
        latsg = latsg.astype(np.float32)
        self.latgrids = latsg.expand_dims(lon=tmean_period[0]['lon'], axis=[1]).values
        lat_rad = np.radians(self.latgrids)
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        tan_lat = np.tan(lat_rad)
        doys = tmean_period['time'].dt.dayofyear.values

        with open(self.ndvi_pickle_path, 'rb') as f:
            ndvi_array = pickle.load(f)

        water_holding_capacity = self.uw.align_rasters(
            f'{self.working_dir}/soil/clipped_AWCh3_M_sl6_1km_ll.tif',
            israster=True
        ) * 10
        water_holding_capacity = np.asarray(water_holding_capacity[0], dtype=np.float32)
        water_holding_capacity[~np.isfinite(water_holding_capacity)] = 0.0
        water_holding_capacity = np.maximum(water_holding_capacity, 0.0)
        max_allowable_depletion = 0.5 * water_holding_capacity

        tree_cover = self.uw.align_rasters(self.tree_cover_tiff, israster=True)[0]
        herb_cover = self.uw.align_rasters(self.herb_cover_tiff, israster=True)[0]
        tree_cover = np.where(tree_cover > 100, 0, tree_cover)
        herb_cover = np.where(herb_cover > 100, 0, herb_cover)

        interception = ((0.15 * tree_cover) + (0.1 * herb_cover)) / 100
        interception = np.asarray(interception)
        one_minus_interception = 1.0 - interception

        RunoffRouter = _load_runoff_router()
        rout = RunoffRouter(self.working_dir, self.clipped_dem, self.routing_method)
        _, acc = rout.compute_flow_dir()
        facc_thresh = np.nanmax(acc) * 0.0001
        facc_mask = np.where(acc < facc_thresh, 0, 1)

        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        date_list = [
            (start + timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range((end - start).days + 1)
        ]
        available_days = min(
            len(date_list),
            int(rf.sizes.get("time", len(rf))),
            int(self.pet_params.sizes.get("time", len(self.pet_params))),
            len(doys),
        )
        if available_days <= 0:
            raise ValueError("No overlapping daily climate inputs were found for the requested simulation period.")
        if available_days != len(date_list):
            missing_start = date_list[available_days]
            missing_end = date_list[-1]
            raise ValueError(
                "Required climate inputs are not available for the full requested VegET date range. "
                f"Available coverage ends before {missing_end}; first missing date is {missing_start}."
            )
        sim_dates = date_list

        init_sm = rf[0] * 0
        init_sm = self.uw.align_rasters(init_sm, israster=False)
        init_sm = np.asarray(init_sm, dtype=np.float32)

        if (not resume) and (os.path.exists(state_file) or os.listdir(chunks_dir)):
            if os.path.exists(state_file):
                os.remove(state_file)
            for name in os.listdir(chunks_dir):
                os.remove(os.path.join(chunks_dir, name))

        if extension_mode:
            resume_state = self._load_resume_state(state_file, signature) if resume else None
            if resume_state is not None:
                start_idx = int(resume_state.get("next_index", 0))
                soil_moisture = np.asarray(resume_state["soil_moisture"], dtype=np.float32)
                if start_idx > available_days:
                    raise ValueError(
                        f"Resume state points to day {start_idx + 1}, but only {available_days} aligned days are available. "
                        "Delete the resume files or rerun with resume=False."
                    )
                print(f"Resuming VegET extension from day {start_idx + 1} of {available_days}")
            else:
                start_idx = (existing_end_dt - start).days + 1
                if start_idx <= 0:
                    raise ValueError(
                        "VegET extension requires the requested end_date to extend beyond the last completed date."
                    )
                soil_moisture = np.asarray(output_metadata["soil_moisture"], dtype=np.float32)
                if resume and (os.path.exists(state_file) or os.listdir(chunks_dir)):
                    print("Resume state invalid or stale for extension; restarting extension from the saved completed state.")
                    if os.path.exists(state_file):
                        os.remove(state_file)
                    for name in os.listdir(chunks_dir):
                        os.remove(os.path.join(chunks_dir, name))
                elif resume:
                    print(
                        "No matching VegET extension checkpoint was found; "
                        f"continuing from the last completed output day. "
                        f"New checkpoints will be written every {checkpoint_days} simulated day(s)."
                    )
            self.wacc_list = list(existing_wacc_list)
        else:
            resume_state = self._load_resume_state(state_file, signature) if resume else None
            if resume_state is not None:
                start_idx = int(resume_state.get("next_index", 0))
                soil_moisture = np.asarray(resume_state["soil_moisture"], dtype=np.float32)
                if start_idx > available_days:
                    raise ValueError(
                        f"Resume state points to day {start_idx + 1}, but only {available_days} aligned days are available. "
                        "Delete the resume files or rerun with resume=False."
                    )
                print(f"Resuming VegET from day {start_idx + 1} of {available_days}")
            else:
                start_idx = 0
                soil_moisture = init_sm
                if resume and (os.path.exists(state_file) or os.listdir(chunks_dir)):
                    print("Resume state invalid or stale; starting from day 1 and clearing old checkpoints.")
                    if os.path.exists(state_file):
                        os.remove(state_file)
                    for name in os.listdir(chunks_dir):
                        os.remove(os.path.join(chunks_dir, name))
                elif resume:
                    print(
                        "No matching VegET checkpoint was found; starting from day 1. "
                        f"Checkpoints will be written every {checkpoint_days} simulated day(s)."
                    )
            self.wacc_list = []

        ref_shape = soil_moisture.shape

        def _align_or_values(arr):
            """Align arrays to DEM grid or return values if already aligned."""
            if hasattr(arr, "shape") and arr.shape == ref_shape:
                return np.asarray(arr, dtype=np.float32)
            aligned = self.uw.align_rasters(arr, israster=False)
            return np.asarray(aligned, dtype=np.float32)

        print('\n')
        chunk_start = start_idx
        wacc_buffer = []
        for count in tqdm(range(start_idx, available_days), desc="     Simulating and routing runoff", unit="day", total=available_days):
            date = sim_dates[count]
            if count % 365 == 0:
                year_num = (count // 365) + 1
                print(f'    Computing surface runoff and routing flow to river channels in year {year_num}')
            this_rf = _align_or_values(rf[count])
            eff_rain = this_rf * one_minus_interception
            eff_rain = np.where(eff_rain < 0, 0, eff_rain)

            doy = doys[count]
            this_et = eto.compute_PET(self.pet_params[count], tan_lat, cos_lat, sin_lat, doy)
            this_et = _align_or_values(this_et)

            day_num = int(doys[count])
            ndvi_day = _align_or_values(ndvi_array[day_num] * 0.0001)

            this_kcp = 1.25 * ndvi_day
            this_kcp += 0.2 * (ndvi_day > 0.4)

            ks = np.divide(
                soil_moisture,
                max_allowable_depletion,
                out=np.zeros_like(soil_moisture, dtype=np.float32),
                where=max_allowable_depletion > 0,
            )
            ks = np.clip(ks, 0.0, 1.0)

            ETa = this_et * ks * this_kcp

            soil_moisture, q_surf = update_soil_and_runoff(
                soil_moisture,
                eff_rain,
                ETa,
                max_allowable_depletion,
                water_holding_capacity
            )

            mask = ~np.isfinite(soil_moisture)
            soil_moisture[mask] = 0

            mask = ~np.isfinite(q_surf)
            q_surf[mask] = 0

            ro_tiff = rout.convert_runoff_layers(q_surf)
            wacc = rout.compute_weighted_flow_accumulation(ro_tiff)
            wacc = wacc * facc_mask
            wacc = sp.sparse.coo_array(wacc)
            wacc_buffer.append({"time": date, "matrix": wacc})

            flush = (len(wacc_buffer) >= checkpoint_days) or (count == available_days - 1)
            if flush:
                chunk_file = os.path.join(chunks_dir, f"chunk_{chunk_start:07d}_{count:07d}.pkl")
                with open(chunk_file, "wb") as f:
                    pickle.dump(wacc_buffer, f)
                wacc_buffer = []
                self._save_resume_state(state_file, signature, count + 1, soil_moisture)
                chunk_start = count + 1

        chunk_files = sorted(
            [os.path.join(chunks_dir, n) for n in os.listdir(chunks_dir) if n.endswith(".pkl")]
        )
        for chunk_file in chunk_files:
            with open(chunk_file, "rb") as f:
                self.wacc_list.extend(pickle.load(f))

        with open(final_file, 'wb') as f:
            pickle.dump(self.wacc_list, f)
        self._save_output_metadata(metadata_file, self.end_date, soil_moisture)
        print(f'Completed. Routed runoff data saved to {final_file}')

        if os.path.exists(state_file):
            os.remove(state_file)
        if os.path.exists(chunks_dir):
            shutil.rmtree(chunks_dir, ignore_errors=True)
