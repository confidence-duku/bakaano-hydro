"""Daily routed rainfall products.

Role: Compute daily routed rainfall accumulation products.
"""

import os
import pickle
import shutil
from datetime import datetime, timedelta

import numpy as np
import scipy as sp
from tqdm import tqdm

from bakaano.core.utils import Utils
from bakaano.data.meteo import Meteo


def _load_runoff_router():
    """Import RunoffRouter only when routed rainfall is requested."""
    from bakaano.hydrology.router import RunoffRouter

    return RunoffRouter


class RainfallFeatures:
    """Compute optional routed-rainfall feature products on the DEM grid."""

    def __init__(
        self,
        working_dir,
        study_area,
        start_date,
        end_date,
        climate_data_source,
        routing_method="mfd",
        runoff_output_dir=None,
    ):
        self.working_dir = working_dir
        self.study_area = study_area
        self.start_date = start_date
        self.end_date = end_date
        self.climate_data_source = climate_data_source
        self.routing_method = routing_method

        self.uw = Utils(self.working_dir, self.study_area)
        self.clipped_dem = f"{self.working_dir}/elevation/dem_clipped.tif"
        self.runoff_output_dir = runoff_output_dir or f"{self.working_dir}/runoff_output"

        os.makedirs(f"{self.working_dir}/models", exist_ok=True)
        os.makedirs(f"{self.working_dir}/runoff_output", exist_ok=True)
        os.makedirs(f"{self.working_dir}/scratch", exist_ok=True)
        os.makedirs(f"{self.working_dir}/shapes", exist_ok=True)
        os.makedirs(f"{self.working_dir}/catchment", exist_ok=True)
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
            "routing_method": self.routing_method,
            "climate_data_source": self.climate_data_source,
            "study_area": str(self.study_area),
        }

    def _load_resume_index(self, state_file, expected_signature):
        """Load and validate resume state for routed rainfall."""
        if not os.path.exists(state_file):
            return None
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        if state.get("signature") != expected_signature:
            return None
        return state

    def _save_resume_index(self, state_file, signature, next_index):
        """Persist only the next day index for resumable rainfall routing."""
        state = {
            "signature": signature,
            "next_index": next_index,
        }
        with open(state_file, "wb") as f:
            pickle.dump(state, f)

    def _load_output_metadata(self, metadata_file):
        """Load completed-run metadata if it exists."""
        if not os.path.exists(metadata_file):
            return None
        with open(metadata_file, "rb") as f:
            return pickle.load(f)

    def _save_output_metadata(self, metadata_file, completed_end_date):
        """Persist enough state to validate completed routed-rainfall outputs."""
        metadata = {
            "start_date": self.start_date,
            "end_date": completed_end_date,
            "signature": self._resume_signature(),
            "products": ("rainfall_sparse_arrays.pkl",),
        }
        with open(metadata_file, "wb") as f:
            pickle.dump(metadata, f)

    def _load_rainfall_inputs(self):
        """Load rainfall inputs aligned to the requested period."""
        self._validate_climate_data_source()

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
        prep_nc, _, _, _ = cd.get_meteo_data()

        if self.climate_data_source == "CHELSA":
            rf = prep_nc.pr.sel(time=slice(self.start_date, self.end_date)) * 86400
        elif self.climate_data_source == "ERA5":
            rf = prep_nc.pr.sel(time=slice(self.start_date, self.end_date)) * 1000
        else:
            rf = prep_nc.pr.sel(time=slice(self.start_date, self.end_date))

        rf = rf.astype(np.float32).assign_coords(
            lat=rf["lat"].astype(np.float32),
            lon=rf["lon"].astype(np.float32),
        )

        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        date_list = [
            (start + timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range((end - start).days + 1)
        ]

        available_days = min(
            len(date_list),
            int(rf.sizes.get("time", len(rf))),
        )
        if available_days <= 0:
            raise ValueError("No overlapping daily rainfall inputs were found for the requested period.")
        if available_days != len(date_list):
            print(
                "Warning: requested rainfall routing spans "
                f"{len(date_list)} days, but climate inputs provide only {available_days} daily steps. "
                "Processing will stop at the last aligned day."
            )

        return {
            "rf": rf,
            "dates": date_list[:available_days],
            "available_days": available_days,
        }

    def compute_routed_rainfall(self, resume=True, checkpoint_days=30):
        """Compute routed daily rainfall sparse products."""
        if checkpoint_days < 1:
            raise ValueError("checkpoint_days must be >= 1.")

        rainfall_final = os.path.join(self.runoff_output_dir, "rainfall_sparse_arrays.pkl")
        metadata_file = os.path.join(self.runoff_output_dir, "rainfall_output_metadata.pkl")
        state_file = os.path.join(self.runoff_output_dir, "rainfall_resume_state.pkl")
        rainfall_chunks_dir = os.path.join(self.runoff_output_dir, "rainfall_resume_chunks")
        os.makedirs(rainfall_chunks_dir, exist_ok=True)

        signature = self._resume_signature()
        products = ("rainfall_sparse_arrays.pkl",)
        signature["products"] = products

        if os.path.exists(rainfall_final):
            output_metadata = self._load_output_metadata(metadata_file)
            if output_metadata is None:
                raise ValueError(
                    "Existing routed rainfall output does not include metadata required to validate it. "
                    "Delete the existing rainfall output and rerun the full date range once."
                )

            existing_start = output_metadata.get("start_date")
            existing_end = output_metadata.get("end_date")
            existing_sig = output_metadata.get("signature")
            existing_products = tuple(output_metadata.get("products", ()))

            if (
                existing_start == self.start_date
                and existing_end == self.end_date
                and existing_sig == self._resume_signature()
                and existing_products == products
            ):
                print(
                    f"Routed rainfall already covers {existing_start} to "
                    f"{existing_end} in {self.runoff_output_dir}. Skipping processing"
                )
                return {"rainfall": rainfall_final}

            raise ValueError(
                "Existing routed rainfall output was generated for a different date range or routing configuration. "
                "Use a different runoff_output_dir or delete the existing rainfall output before rerunning."
            )

        print("Computing routed rainfall")
        if resume:
            print(
                "Routed-rainfall resume mode is enabled. Matching checkpoints are written "
                f"every {checkpoint_days} simulated day(s)."
            )
        climate = self._load_rainfall_inputs()
        rf = climate["rf"]
        sim_dates = climate["dates"]
        available_days = climate["available_days"]

        ref_shape = self.uw.align_rasters(rf[0], israster=False).shape

        def _align_or_values(arr):
            if hasattr(arr, "shape") and arr.shape == ref_shape:
                return np.asarray(arr, dtype=np.float32)
            aligned = self.uw.align_rasters(arr, israster=False)
            return np.asarray(aligned, dtype=np.float32)

        RunoffRouter = _load_runoff_router()
        rout = RunoffRouter(self.working_dir, self.clipped_dem, self.routing_method)
        _, acc = rout.compute_flow_dir()
        facc_thresh = np.nanmax(acc) * 0.0001
        facc_mask = np.where(acc < facc_thresh, 0, 1)

        if (not resume) and (
            os.path.exists(state_file)
            or os.listdir(rainfall_chunks_dir)
        ):
            if os.path.exists(state_file):
                os.remove(state_file)
            for name in os.listdir(rainfall_chunks_dir):
                os.remove(os.path.join(rainfall_chunks_dir, name))

        resume_state = self._load_resume_index(state_file, signature) if resume else None
        if resume_state is not None:
            start_idx = int(resume_state.get("next_index", 0))
            if start_idx > available_days:
                raise ValueError(
                    f"Resume state points to day {start_idx + 1}, but only {available_days} aligned days are available. "
                    "Delete the resume files or rerun with resume=False."
                )
            print(f"Resuming routed rainfall from day {start_idx + 1} of {available_days}")
        else:
            start_idx = 0
            if resume and (os.path.exists(state_file) or os.listdir(rainfall_chunks_dir)):
                print("Resume state invalid or stale; starting from day 1 and clearing old checkpoints.")
                if os.path.exists(state_file):
                    os.remove(state_file)
                for name in os.listdir(rainfall_chunks_dir):
                    os.remove(os.path.join(rainfall_chunks_dir, name))
            elif resume:
                print(
                    "No matching routed-rainfall checkpoint was found; starting from day 1. "
                    f"Checkpoints will be written every {checkpoint_days} simulated day(s)."
                )

        rainfall_buffer = []
        chunk_start = start_idx

        for count in tqdm(
            range(start_idx, available_days),
            desc="     Routing rainfall",
            unit="day",
            total=available_days,
        ):
            date = sim_dates[count]
            if count % 365 == 0:
                year_num = (count // 365) + 1
                print(f"    Routing rainfall in year {year_num}")

            rainfall = _align_or_values(rf[count])
            rainfall[~np.isfinite(rainfall)] = 0.0
            rainfall = np.maximum(rainfall, 0.0)

            rainfall_wacc = rout.compute_weighted_flow_accumulation(
                rout.convert_runoff_layers(rainfall)
            )
            rainfall_wacc = sp.sparse.coo_array(rainfall_wacc * facc_mask)
            rainfall_buffer.append({"time": date, "matrix": rainfall_wacc})

            flush = (len(rainfall_buffer) >= checkpoint_days) or (count == available_days - 1)
            if flush:
                rainfall_chunk = os.path.join(
                    rainfall_chunks_dir, f"chunk_{chunk_start:07d}_{count:07d}.pkl"
                )
                with open(rainfall_chunk, "wb") as f:
                    pickle.dump(rainfall_buffer, f)

                rainfall_buffer = []
                self._save_resume_index(state_file, signature, count + 1)
                chunk_start = count + 1

        rainfall_list = []
        rainfall_chunk_files = sorted(
            os.path.join(rainfall_chunks_dir, name)
            for name in os.listdir(rainfall_chunks_dir)
            if name.endswith(".pkl")
        )
        for chunk_file in rainfall_chunk_files:
            with open(chunk_file, "rb") as f:
                rainfall_list.extend(pickle.load(f))

        with open(rainfall_final, "wb") as f:
            pickle.dump(rainfall_list, f)
        self._save_output_metadata(metadata_file, sim_dates[-1])

        print(f"Completed. Routed rainfall data saved to {rainfall_final}")

        if os.path.exists(state_file):
            os.remove(state_file)
        shutil.rmtree(rainfall_chunks_dir, ignore_errors=True)

        return {"rainfall": rainfall_final}
