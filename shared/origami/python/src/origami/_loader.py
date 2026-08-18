# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Runtime preparation for the origami compiled extension.

Makes the extension's ROCm ``NEEDED`` dependencies (``liborigami.so.1``,
``libamdhip64.so`` and the core runtime) resolvable before the extension is
imported, for both ROCm-wheel installs (via ``rocm_sdk``) and non-wheel installs
(via ``ROCM_PATH`` / ``HIP_PATH`` / ``ROCM_HOME``).
"""

import os
import platform

_IS_WINDOWS = platform.system() == "Windows"

_ROCM_WHEEL_SHORTNAMES = ["amd_comgr", "amdhip64", "hiprtc", "origami"]

_dll_directory_handle = None


def preload_via_rocm_sdk() -> bool:
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

    Returns True only when rocm_sdk drove a successful preload. Returns False
    when rocm_sdk is absent or its initialization failed, so the caller falls
    through to the non-wheel resolution path instead of assuming the runtime is
    ready.
    """
    try:
        import rocm_sdk
    except ImportError:
        return False
    try:
        rocm_sdk.initialize_process(preload_shortnames=_ROCM_WHEEL_SHORTNAMES)
    except Exception:
        return False
    return True


def register_rocm_path_dir() -> bool:
    """Non-wheel installs -- a system ``/opt/rocm``, a ``.deb``, the Windows HIP
    SDK, or a build tree -- where the runtime lives in one directory named by the
    ``ROCM_PATH`` / ``HIP_PATH`` / ``ROCM_HOME`` environment variables.

    On Windows that directory's ``bin/`` must be registered via
    ``os.add_dll_directory`` because extension modules load with
    ``LOAD_LIBRARY_SEARCH_DEFAULT_DIRS``, which excludes ``PATH`` and has no
    RPATH equivalent. The handle returned by ``os.add_dll_directory`` controls
    the registration lifetime -- the directory is removed from the search path
    when the handle is closed or garbage-collected -- so it is retained in a
    module-level variable until process exit, well past the extension import.
    On Linux the dynamic loader already searches RPATH / ldconfig /
    ``LD_LIBRARY_PATH``, so there is nothing to do.

    Returns True when a directory was registered, False otherwise.
    """
    if not _IS_WINDOWS:
        return False
    global _dll_directory_handle
    for var in ("ROCM_PATH", "HIP_PATH", "ROCM_HOME"):
        root = os.environ.get(var)
        if root:
            bin_dir = os.path.join(root, "bin")
            if os.path.isdir(bin_dir):
                _dll_directory_handle = os.add_dll_directory(bin_dir)
                return True
    return False


def prepare_runtime() -> None:
    """Make the extension's ROCm dependencies resolvable before it is imported.

    Prefer the rocm_sdk preload (the wheel-install path). If rocm_sdk is absent,
    or is present but its preload failed, fall through to the ``ROCM_PATH``
    directory registration (the non-wheel path) as a best effort. Any dependency
    still missing after this surfaces as an actionable ``ImportError`` from the
    extension import itself.
    """
    if not preload_via_rocm_sdk():
        register_rocm_path_dir()
