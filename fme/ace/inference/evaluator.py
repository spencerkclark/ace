import copy
import dataclasses
import datetime
import logging
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence

import dacite
import numpy as np
import torch
import xarray as xr

import fme
import fme.core.logging_utils as logging_utils
from fme.ace.aggregator.inference import InferenceEvaluatorAggregatorConfig
from fme.ace.data_loading.batch_data import BatchData
from fme.ace.data_loading.getters import get_inference_data
from fme.ace.data_loading.inference import ExplicitIndices, InferenceDataLoaderConfig
from fme.ace.inference.data_writer import DataWriterConfig, PairedDataWriter
from fme.ace.inference.data_writer.dataset_metadata import DatasetMetadata
from fme.ace.inference.default_metadata import get_default_variable_metadata
from fme.ace.inference.loop import DeriverABC, run_dataset_comparison
from fme.ace.stepper import (
    Stepper,
    StepperOverrideConfig,
    load_stepper,
    load_stepper_config,
)
from fme.ace.stepper.single_module import StepperConfig
from fme.core.cli import prepare_config, prepare_directory
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset.xarray import XarrayDataset
from fme.core.dataset_info import IncompatibleDatasetInfo
from fme.core.derived_variables import get_derived_variable_metadata
from fme.core.dicts import to_flat_dict
from fme.core.distributed import Distributed
from fme.core.generics.inference import get_record_to_wandb, run_inference
from fme.core.logging_utils import LoggingConfig
from fme.core.timing import GlobalTimer
from fme.core.typing_ import TensorDict, TensorMapping


def resolve_variable_metadata(
    dataset_metadata: Mapping[str, VariableMetadata],
    stepper_metadata: Mapping[str, VariableMetadata],
    stepper_all_names: Sequence[str],
) -> dict[str, VariableMetadata]:
    """
    Resolve variable metadata by merging from the following sources: derived variables,
    the dataset, the stepper, and finally a set of defaults. If there are conflicts on
    variable metadata values, preference is given first to values from the stepper,
    then from the dataset, and finally from default values.

    Note that if not saved with the stepper, the variable metadata is not guaranteed to
    be the same as that in the dataset used for training the stepper.

    Args:
        dataset_metadata: Metadata from the dataset.
        stepper_metadata: Metadata from the stepper.
        stepper_all_names: Variable names associated with the stepper.

    Returns:
        A mappping of variable names to metadata.
    """
    default_metadata = get_default_variable_metadata(version="era5_v1")
    names_from_default = (
        set(stepper_all_names) - (dataset_metadata.keys() | stepper_metadata.keys())
    ) & default_metadata.keys()
    if names_from_default:
        logging.warning(
            "Variable metadata for the following stepper variables were not found in "
            "the variable metadata of the forcing dataset or stepper: "
            f"{names_from_default}. Using default values for these variables instead. "
            "Users should ensure that the default values are consistent with the "
            "training dataset of the stepper."
        )
    resolved_metadata = (
        default_metadata | dict(dataset_metadata) | dict(stepper_metadata)
    )
    resolved_metadata = {
        name: resolved_metadata[name]
        for name in stepper_all_names
        if name in resolved_metadata
    }
    return get_derived_variable_metadata() | resolved_metadata


@dataclasses.dataclass
class InferenceEvaluatorConfig:
    """
    Configuration for running inference including comparison to reference data.

    Parameters:
        experiment_dir: Directory to save results to.
        n_forward_steps: Number of steps to run the model forward for.
        checkpoint_path: Path to stepper checkpoint to load.
        logging: configuration for logging.
        loader: Configuration for data to be used as initial conditions, forcing, and
            target in inference.
        prediction_loader: Configuration for prediction data to evaluate. If given,
            model evaluation will not run, and instead predictions will be evaluated.
            Model checkpoint will still be used to determine inputs and outputs.
        forward_steps_in_memory: Number of forward steps to complete in memory
            at a time, will load one more step for initial condition.
        data_writer: Configuration for data writers.
        aggregator: Configuration for inference evaluator aggregator.
        stepper_override: Configuration for overriding select stepper configuration
            options at inference time (optional).
        allow_incompatible_dataset: If True, allow the forcing dataset used
            for inference to be incompatible with the dataset used for stepper training.
            This should be used with caution, as it may allow the stepper to make
            scientifically invalid predictions, but it can allow running inference with
            incorrectly formatted or missing grid information.
    """

    experiment_dir: str
    n_forward_steps: int
    checkpoint_path: str
    logging: LoggingConfig
    loader: InferenceDataLoaderConfig
    prediction_loader: InferenceDataLoaderConfig | None = None
    forward_steps_in_memory: int = 1
    data_writer: DataWriterConfig = dataclasses.field(
        default_factory=lambda: DataWriterConfig()
    )
    aggregator: InferenceEvaluatorAggregatorConfig = dataclasses.field(
        default_factory=lambda: InferenceEvaluatorAggregatorConfig()
    )
    stepper_override: StepperOverrideConfig | None = None
    allow_incompatible_dataset: bool = False

    def __post_init__(self):
        if self.data_writer.time_coarsen is not None:
            self.data_writer.time_coarsen.validate(
                self.forward_steps_in_memory,
                self.n_forward_steps,
            )
        if self.data_writer.files is not None:
            for file_config in self.data_writer.files:
                if file_config.time_coarsen is not None:
                    file_config.time_coarsen.validate(
                        self.forward_steps_in_memory,
                        self.n_forward_steps,
                    )

    def configure_logging(self, log_filename: str):
        self.logging.configure_logging(self.experiment_dir, log_filename)

    def configure_wandb(
        self, env_vars: dict | None = None, resumable: bool = False, **kwargs
    ):
        config = to_flat_dict(dataclasses.asdict(self))
        self.logging.configure_wandb(
            config=config, env_vars=env_vars, resumable=resumable, **kwargs
        )

    def load_stepper(self) -> Stepper:
        logging.info(f"Loading trained model checkpoint from {self.checkpoint_path}")
        return load_stepper(self.checkpoint_path, self.stepper_override)

    def load_stepper_config(self) -> StepperConfig:
        logging.info(f"Loading trained model checkpoint from {self.checkpoint_path}")
        return load_stepper_config(self.checkpoint_path, self.stepper_override)

    def get_data_writer(
        self,
        timestep: datetime.timedelta,
        variable_metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
    ) -> PairedDataWriter:
        return self.data_writer.build_paired(
            experiment_dir=self.experiment_dir,
            n_initial_conditions=self.loader.n_initial_conditions,
            n_timesteps=self.n_forward_steps,
            timestep=timestep,
            variable_metadata=variable_metadata,
            coords=coords,
            dataset_metadata=DatasetMetadata.from_env(),
        )


def main(yaml_config: str, override_dotlist: Sequence[str] | None = None):
    config_data = prepare_config(yaml_config, override=override_dotlist)
    config = dacite.from_dict(
        data_class=InferenceEvaluatorConfig,
        data=config_data,
        config=dacite.Config(strict=True),
    )
    prepare_directory(config.experiment_dir, config_data)
    with GlobalTimer(), torch.no_grad():
        return run_evaluator_from_config(config)


class _Deriver(DeriverABC):
    """
    DeriverABC implementation for dataset comparison.
    """

    def __init__(
        self,
        n_ic_timesteps: int,
        derive_func: Callable[[TensorMapping, TensorMapping], TensorDict],
    ):
        self._n_ic_timesteps = n_ic_timesteps
        self._derive_func = derive_func

    @property
    def n_ic_timesteps(self) -> int:
        return self._n_ic_timesteps

    def get_forward_data(
        self, data: BatchData, compute_derived_variables: bool = False
    ) -> BatchData:
        if compute_derived_variables:
            timer = GlobalTimer.get_instance()
            with timer.context("compute_derived_variables"):
                data = data.compute_derived_variables(
                    derive_func=self._derive_func,
                    forcing_data=data,
                )
        return data.remove_initial_condition(self._n_ic_timesteps)


def run_evaluator_from_config(config: InferenceEvaluatorConfig):
    timer = GlobalTimer.get_instance()
    timer.start_outer("inference")
    timer.start("initialization")

    if not os.path.isdir(config.experiment_dir):
        os.makedirs(config.experiment_dir, exist_ok=True)
    config.configure_logging(log_filename="inference_out.log")
    env_vars = logging_utils.retrieve_env_vars()
    beaker_url = logging_utils.log_beaker_url()
    config.configure_wandb(env_vars=env_vars, notes=beaker_url)

    if fme.using_gpu():
        torch.backends.cudnn.benchmark = True

    logging_utils.log_versions()
    logging.info(f"Current device is {fme.get_device()}")

    stepper_config = config.load_stepper_config()
    logging.info("Initializing data loader")
    window_requirements = stepper_config.get_evaluation_window_data_requirements(
        n_forward_steps=config.forward_steps_in_memory
    )
    initial_condition_requirements = (
        stepper_config.get_prognostic_state_data_requirements()
    )
    data = get_inference_data(
        config=config.loader,
        total_forward_steps=config.n_forward_steps,
        window_requirements=window_requirements,
        initial_condition=initial_condition_requirements,
    )

    stepper = config.load_stepper()
    stepper.set_eval()

    if not config.allow_incompatible_dataset:
        try:
            stepper.training_dataset_info.assert_compatible_with(data.dataset_info)
        except IncompatibleDatasetInfo as err:
            raise IncompatibleDatasetInfo(
                "Inference dataset is not compatible with dataset used for stepper "
                "training. Set allow_incompatible_dataset to True to ignore this "
                f"error. The incompatiblity found was: {str(err)}"
            ) from err

    aggregator_config: InferenceEvaluatorAggregatorConfig = config.aggregator
    for batch in data.loader:
        initial_time = batch.time.isel(time=0)
        break
    variable_metadata = resolve_variable_metadata(
        dataset_metadata=data.variable_metadata,
        stepper_metadata=stepper.training_variable_metadata,
        stepper_all_names=stepper_config.all_names,
    )
    dataset_info = data.dataset_info.update_variable_metadata(variable_metadata)
    aggregator = aggregator_config.build(
        dataset_info=dataset_info,
        record_step_20=config.n_forward_steps >= 20,
        n_timesteps=config.n_forward_steps + stepper_config.n_ic_timesteps,
        initial_time=initial_time,
        channel_mean_names=stepper.loss_names,
        normalize=stepper.normalizer.normalize,
        output_dir=config.experiment_dir,
    )

    writer = config.get_data_writer(
        timestep=data.timestep,
        variable_metadata=variable_metadata,
        coords=data.coords,
    )

    timer.stop()
    logging.info("Starting inference")
    record_logs = get_record_to_wandb(label="inference")
    if config.prediction_loader is not None:
        prediction_data = get_inference_data(
            config.prediction_loader,
            total_forward_steps=config.n_forward_steps,
            window_requirements=window_requirements,
            initial_condition=initial_condition_requirements,
        )
        deriver = _Deriver(
            n_ic_timesteps=stepper_config.n_ic_timesteps,
            derive_func=stepper.derive_func,
        )
        run_dataset_comparison(
            aggregator=aggregator,
            prediction_data=prediction_data,
            target_data=data,
            deriver=deriver,
            writer=writer,
            record_logs=record_logs,
        )
    else:
        run_inference(
            predict=stepper.predict_paired,
            data=data,
            aggregator=aggregator,
            writer=writer,
            record_logs=record_logs,
        )

    timer.start("final_writer_flush")
    logging.info("Starting final flush of data writer")
    writer.finalize()
    logging.info("Writing reduced metrics to disk in netcdf format.")
    aggregator.flush_diagnostics()
    timer.stop()

    timer.stop_outer("inference")
    total_steps = config.n_forward_steps * config.loader.n_initial_conditions
    inference_duration = timer.get_duration("inference")
    wandb_logging_duration = timer.get_duration("wandb_logging")
    total_steps_per_second = total_steps / (inference_duration - wandb_logging_duration)
    timer.log_durations()
    logging.info(
        "Total steps per second (ignoring wandb logging): "
        f"{total_steps_per_second:.2f} steps/second"
    )

    summary_logs = {
        "total_steps_per_second": total_steps_per_second,
        **timer.get_durations(),
        **aggregator.get_summary_logs(),
    }
    record_logs([summary_logs])


def batched(iterable, n=1):
    # Since we do not use python 3.12 yet, which includes itertools.batched,
    # we need to implement this ourselves.
    # https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def set_chunks_and_shards_encoding(
    ds: xr.Dataset, sample_chunks: int, sample_shards: int | None
):
    for name in ds.variables:
        chunks = []
        if sample_shards is not None:
            shards = []
        for dim, size in ds[name].sizes.items():
            if dim == "sample":
                chunks.append(sample_chunks)
                if sample_shards is not None:
                    shards.append(sample_shards)
            else:
                chunks.append(size)
                if sample_shards is not None:
                    shards.append(size)
        ds[name].encoding["chunks"] = tuple(chunks)
        if sample_shards is not None:
            ds[name].encoding["shards"] = tuple(shards)
    return ds


@dataclasses.dataclass
class BatchedEnsembleEvaluatorConfig:
    base_evaluator_config: InferenceEvaluatorConfig
    batch_size: int
    sample_chunks: int = 120
    sample_shards: int | None = None


def run_batched_ensemble_evaluator_from_config(config: BatchedEnsembleEvaluatorConfig):
    base_evaluator_config = config.base_evaluator_config
    batch_size = config.batch_size

    timer = GlobalTimer.get_instance()
    timer.start_outer("inference")
    timer.start("initialization")

    if not os.path.isdir(base_evaluator_config.experiment_dir):
        os.makedirs(base_evaluator_config.experiment_dir, exist_ok=True)
    base_evaluator_config.configure_logging(log_filename="inference_out.log")
    env_vars = logging_utils.retrieve_env_vars()
    beaker_url = logging_utils.log_beaker_url()
    base_evaluator_config.configure_wandb(env_vars=env_vars, notes=beaker_url)

    if fme.using_gpu():
        torch.backends.cudnn.benchmark = True

    logging_utils.log_versions()
    logging.info(f"Current device is {fme.get_device()}")

    stepper_config = base_evaluator_config.load_stepper_config()
    logging.info("Initializing data loader")
    window_requirements = stepper_config.get_evaluation_window_data_requirements(
        n_forward_steps=base_evaluator_config.forward_steps_in_memory
    )
    initial_condition_requirements = (
        stepper_config.get_prognostic_state_data_requirements()
    )

    dataset = XarrayDataset(
        base_evaluator_config.loader.dataset,
        window_requirements.names,
        window_requirements.n_timesteps_schedule,
    )

    stepper = base_evaluator_config.load_stepper()
    stepper.set_eval()
    timer.stop()
    for i, batch in enumerate(
        batched(base_evaluator_config.loader.start_indices.list, n=batch_size)
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            batch_config = copy.deepcopy(base_evaluator_config)
            batch_config.loader.start_indices = ExplicitIndices(list(batch))
            batch_config.experiment_dir = os.path.join(temp_dir)
            os.makedirs(batch_config.experiment_dir, exist_ok=True)
            data = get_inference_data(
                config=batch_config.loader,
                total_forward_steps=batch_config.n_forward_steps,
                window_requirements=window_requirements,
                initial_condition=initial_condition_requirements,
                xarray_dataset=dataset,
            )

            aggregator_config: InferenceEvaluatorAggregatorConfig = (
                batch_config.aggregator
            )
            for batch in data.loader:
                initial_time = batch.time.isel(time=0)
                break
            variable_metadata = resolve_variable_metadata(
                dataset_metadata=data.variable_metadata,
                stepper_metadata=stepper.training_variable_metadata,
                stepper_all_names=stepper_config.all_names,
            )
            dataset_info = data.dataset_info.update_variable_metadata(variable_metadata)
            aggregator = aggregator_config.build(
                dataset_info=dataset_info,
                record_step_20=batch_config.n_forward_steps >= 20,
                n_timesteps=batch_config.n_forward_steps
                + stepper_config.n_ic_timesteps,
                initial_time=initial_time,
                channel_mean_names=stepper.loss_names,
                normalize=stepper.normalizer.normalize,
                output_dir=batch_config.experiment_dir,
            )

            writer = batch_config.get_data_writer(
                timestep=data.timestep,
                variable_metadata=variable_metadata,
                coords=data.coords,
            )

            logging.info("Starting inference")
            record_logs = get_record_to_wandb(label="inference")

            run_inference(
                predict=stepper.predict_paired,
                data=data,
                aggregator=aggregator,
                writer=writer,
                record_logs=record_logs,
            )

            timer.start("final_writer_flush")
            logging.info("Starting final flush of data writer")
            writer.finalize()
            logging.info("Writing reduced metrics to disk in netcdf format.")
            aggregator.flush_diagnostics()
            timer.stop()

            timer.start("zarr_append")
            predictions_netcdf = os.path.join(temp_dir, "autoregressive_predictions.nc")
            predictions_zarr = os.path.join(
                base_evaluator_config.experiment_dir, "autoregressive_predictions.zarr"
            )

            target_netcdf = os.path.join(temp_dir, "autoregressive_target.nc")
            target_zarr = os.path.join(
                base_evaluator_config.experiment_dir, "autoregressive_target.zarr"
            )

            if os.path.exists(predictions_zarr):
                ds = xr.open_dataset(predictions_netcdf, decode_timedelta=False)
                ds.to_zarr(predictions_zarr, append_dim="sample", mode="a")

                ds = xr.open_dataset(target_netcdf, decode_timedelta=False)
                ds.to_zarr(target_zarr, append_dim="sample", mode="a")
            else:
                ds = xr.open_dataset(predictions_netcdf, decode_timedelta=False)
                ds = set_chunks_and_shards_encoding(
                    ds,
                    sample_chunks=config.sample_chunks,
                    sample_shards=config.sample_shards,
                )
                ds.to_zarr(predictions_zarr)

                ds = xr.open_dataset(target_netcdf, decode_timedelta=False)
                ds = set_chunks_and_shards_encoding(
                    ds,
                    sample_chunks=config.sample_chunks,
                    sample_shards=config.sample_shards,
                )
                ds.to_zarr(target_zarr)
            timer.stop()

    timer.stop_outer("inference")
    total_steps = (
        base_evaluator_config.n_forward_steps * base_evaluator_config.loader.n_initial_conditions
    )
    inference_duration = timer.get_duration("inference")
    wandb_logging_duration = timer.get_duration("wandb_logging")
    total_steps_per_second = total_steps / (
        inference_duration - wandb_logging_duration
    )
    timer.log_durations()
    logging.info(
        "Total steps per second (ignoring wandb logging): "
        f"{total_steps_per_second:.2f} steps/second"
    )

    summary_logs = {
        "total_steps_per_second": total_steps_per_second,
        **timer.get_durations(),
        **aggregator.get_summary_logs(),
    }
    record_logs([summary_logs])
    Distributed.get_instance().shutdown()

                
def main_batched_ensemble_evaluator(
    yaml_config: str, override_dotlist: Sequence[str] | None = None
):
    config_data = prepare_config(yaml_config, override=override_dotlist)
    config = dacite.from_dict(
        data_class=BatchedEnsembleEvaluatorConfig,
        data=config_data,
        config=dacite.Config(strict=True),
    )
    prepare_directory(config.base_evaluator_config.experiment_dir, config_data)
    with GlobalTimer(), torch.no_grad():
        return run_batched_ensemble_evaluator_from_config(config)
