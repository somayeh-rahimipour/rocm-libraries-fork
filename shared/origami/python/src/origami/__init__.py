# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Origami: Analytical GEMM Solution Selection

Python bindings for the Origami C++ library.
"""

import os
import platform

_IS_WINDOWS = platform.system() == "Windows"

_ROCM_WHEEL_SHORTNAMES = ["amd_comgr", "amdhip64", "hiprtc", "origami"]


def _preload_via_rocm_sdk() -> bool:
    """Preload ROCm runtime libraries through rocm_sdk before importing the
    compiled extension so its ``NEEDED liborigami.so.1`` and ``libamdhip64.so``
    resolve to the SDK copies.

    In a ROCm-wheel environment the native libraries ship inside sibling
    ``_rocm_sdk_*`` packages that sit off the loader path and carry a build-time
    version nonce in their names, so only rocm_sdk knows their locations; use its
    public API rather than reimplementing that discovery. ``liborigami`` is one
    of these off-path libraries -- it lives in the ``_rocm_sdk_libraries`` wheel,
    not on ``LD_LIBRARY_PATH`` or on the extension's ``$ORIGIN`` RPATH -- so it is
    named alongside the core runtime. Preloading it RTLD_GLOBAL both resolves the
    extension's ``NEEDED liborigami.so.1`` and guarantees a single shared copy in
    the process, so a co-loaded consumer (hipBLASLt, PyTorch) binds to the same
    object rather than a second, layout-divergent one.

    Returns True when rocm_sdk is installed and drove the preload, False
    otherwise so the caller can fall back.
    """
    try:
        import rocm_sdk
    except ImportError:
        return False
    try:
        rocm_sdk.initialize_process(preload_shortnames=_ROCM_WHEEL_SHORTNAMES)
    except Exception:
        pass
    return True


def _register_rocm_path_dir() -> None:
    """Non-wheel installs -- a system ``/opt/rocm``, a ``.deb``, the Windows HIP
    SDK, or a build tree -- where the runtime lives in one directory named by the
    ``ROCM_PATH`` / ``HIP_PATH`` / ``ROCM_HOME`` environment variables.

    On Windows that directory's ``bin/`` must be registered via
    ``os.add_dll_directory`` because extension modules load with
    ``LOAD_LIBRARY_SEARCH_DEFAULT_DIRS``, which excludes ``PATH`` and has no
    RPATH equivalent. On Linux the dynamic loader already searches RPATH /
    ldconfig / ``LD_LIBRARY_PATH``, so there is nothing to do.
    """
    if not _IS_WINDOWS:
        return
    for var in ("ROCM_PATH", "HIP_PATH", "ROCM_HOME"):
        root = os.environ.get(var)
        if root:
            bin_dir = os.path.join(root, "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
                return


if not _preload_via_rocm_sdk():
    _register_rocm_path_dir()

try:
    # Import the compiled extension module
    from .origami import (
        # Enums
        architecture_t,
        data_type_t,
        transpose_t,
        grid_selection_t,
        reduction_t,
        hybrid_mode_t,
        prediction_modes_t,
        model_t,
        # Data structures
        dim3_t,
        dim4_t,
        tensile_params_t,
        config_t,
        prediction_result_t,
        workgroup_mapping_t,
        staggerU_t,
        problem_t,
        hardware_t,
        context_t,
        # Hardware functions
        get_hardware_for_device,
        get_hardware_for_arch,
        # Data type functions
        int_to_data_type,
        datatype_to_bits,
        string_to_datatype,
        datatype_to_string,
        # Configuration selection functions
        select_config,
        rank_configs,
        select_config_mnk,
        select_topk_configs,
        # Performance functions
        compute_perf_gflops,
        compute_total_latency,
        compute_number_matrix_instructions,
        compute_mt_compute_latency,
        # Memory functions
        check_lds_capacity,
        estimate_l2_hit,
        estimate_mall_hit,
        compute_memory_latency,
        compute_l2_tiles,
        compute_mall_tiles,
        predict_workgroup_mapping,
        wgm_to_grid,
        count_unique_tiles,
        count_unique_tiles_timestep,
        estimate_cache_hit_rates,
        # Latency functions
        compute_tile_latency,
        compute_timestep_latency,
        # StreamK functions
        select_grid_size,
        select_reduction,
        select_workgroup_mapping,
        compute_number_of_output_tiles,
        # Reduction functions
        int_to_reduction_t,
        hybrid_mode_to_string,
        # Attention functions
        att_compute_total_latency,
        att_compute_number_matrix_instructions,
        att_compute_mt_compute_latency,
        att_check_lds_capacity,
        att_estimate_l2_hit,
        att_estimate_mall_hit,
        att_compute_memory_latency,
        att_compute_tile_latency,
        att_compute_timestep_latency,
        att_calculate_work_utilization,
        att_calculate_output_utilization,
        att_compute_cu_occupancy,
        att_arithmetic_intensity,
        att_emulated_tf32_arithmetic_intensity,
        att_round_elements_to_128B,
        att_compute_mem_bw_from_occupancy,
        att_compute_l2_hit_rate_global,
    )
except ImportError as e:
    raise ImportError(
        "Failed to import the origami compiled extension. Its ROCm "
        "dependencies (liborigami, libamdhip64) were not found. Install the "
        "ROCm wheels (`pip install rocm[libraries]`), or set ROCM_PATH / "
        "HIP_PATH to a ROCm install or build tree (on Windows the directory "
        f"containing the ROCm DLLs under bin/).\nOriginal error: {e}"
    ) from e

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Enums
    "architecture_t",
    "data_type_t",
    "transpose_t",
    "grid_selection_t",
    "reduction_t",
    "hybrid_mode_t",
    "prediction_modes_t",
    "model_t",
    # Data structures
    "dim3_t",
    "dim4_t",
    "tensile_params_t",
    "config_t",
    "prediction_result_t",
    "workgroup_mapping_t",
    "problem_t",
    "hardware_t",
    "context_t",
    # Hardware functions
    "get_hardware_for_device",
    "get_hardware_for_arch",
    # Data type functions
    "int_to_data_type",
    "datatype_to_bits",
    "string_to_datatype",
    "datatype_to_string",
    # Configuration selection functions
    "select_config",
    "rank_configs",
    "select_config_mnk",
    "select_topk_configs",
    # Performance functions
    "compute_perf_gflops",
    "compute_total_latency",
    "compute_number_matrix_instructions",
    "compute_mt_compute_latency",
    # Memory functions
    "wgm_to_grid",
    "compute_l2_tiles",
    "compute_mall_tiles",
    "count_unique_tiles",
    "count_unique_tiles_timestep",
    "estimate_cache_hit_rates",
    "check_lds_capacity",
    "estimate_l2_hit",
    "estimate_mall_hit",
    "compute_memory_latency",
    # Latency functions
    "compute_tile_latency",
    "compute_timestep_latency",
    # StreamK functions
    "select_grid_size",
    "select_reduction",
    "select_workgroup_mapping",
    "compute_number_of_output_tiles",
    # Reduction functions
    "int_to_reduction_t",
    "hybrid_mode_to_string",
    # Attention functions
    "att_compute_total_latency",
    "att_compute_number_matrix_instructions",
    "att_compute_mt_compute_latency",
    "att_check_lds_capacity",
    "att_estimate_l2_hit",
    "att_estimate_mall_hit",
    "att_compute_memory_latency",
    "att_compute_tile_latency",
    "att_compute_timestep_latency",
    "att_calculate_work_utilization",
    "att_calculate_output_utilization",
    "att_compute_cu_occupancy",
    "att_arithmetic_intensity",
    "att_emulated_tf32_arithmetic_intensity",
    "att_round_elements_to_128B",
    "att_compute_mem_bw_from_occupancy",
    "att_compute_l2_hit_rate_global",
]

try:
    # Import the python selectors if possible (requires torch)
    from .selector import OrigamiMatmulSelector, OrigamiAttentionSelector
    __all__.append("OrigamiMatmulSelector")
    __all__.append("OrigamiAttentionSelector")
except ImportError:
    # Do not raise this error if import fails - compiled Origami bindings still
    # work without the dedicated Python selectors
    pass

