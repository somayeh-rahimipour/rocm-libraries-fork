Description

Origami's C++ shared library landed in TheRock via PR #5650 (Jul 2026) with Python bindings explicitly disabled (-DORIGAMI_ENABLE_PYTHON=OFF). This ticket tracks enabling proper Python bindings for Origami in TheRock so that the out-of-tree rocm-origami PyPI workaround can be deprecated.

Why this matters -- the rocm-origami PyPI collision (ROCM-29472)

Because TheRock does not ship Origami Python bindings, a third-party rocm-origami==0.0.2 package was published to PyPI as a workaround. That package statically embeds an older copy of liborigami. When ROCm 10.0.0rc2 began shipping its own liborigami.so.1 (which hipBLASLt depends on), both versions coexist in the same process. The older PyPI copy under-allocates for a struct that grew from 112 to 128 bytes in the newer version, causing heap corruption and segfaults in PyTorch CI (see ROCM-29472, Critical S1). The only durable fix is to ship official Python bindings from TheRock so that consumers don't need the PyPI workaround.

History -- three prior land/revert cycles

Origami was landed and reverted from TheRock three times, each time due to packaging/integration failures (not code bugs):

1. PR #2813 landed Feb 2, reverted by #3222 Feb 3: The Python extension had no RPATH pointing to lib/ where liborigami.so.1 is installed, and origami was not registered in _dist_info.py, so rocm-sdk's testSharedLibrariesLoad failed with "cannot open shared object file."

2. PR #3237 landed Feb 20, reverted by #3583 Feb 24: The Python extension was compiled against CPython 3.12 only (origami.cpython-312-*.so). When rocm-sdk tests ran under Python 3.10 or 3.11, the extension failed with "undefined symbol: PyObject_Vectorcall." The version-specific .so was shipped in a version-agnostic location.

3. PR #3820 landed Apr 16, reverted by #4901 Apr 29: Artifact accounting in artifacts-blas.toml did not properly claim origami's Python files and license docs, causing them to leak into incorrect wheels or be silently dropped from the SDK.

PR #5650 (Jul 2026) solved these problems by side-stepping them: it builds origami as a shared library only, with Python bindings disabled. That PR has stuck and is the current state.

What needs to happen

1. Nanobind provider in TheRock: PR #5650 deferred Python bindings until a nanobind provider lands in TheRock (mirrors the hipDNN direction, TheRock #6425). This is a prerequisite. Coordinate with the hipDNN team if they are already working on this.

2. Re-enable Python bindings: Once nanobind is available, set -DORIGAMI_ENABLE_PYTHON=ON in the TheRock origami build (math-libs/BLAS/CMakeLists.txt).

3. Proper Python packaging: The origami Python module must:    - Call rocm_sdk.preload_libraries() in its __init__.py to ensure ROCm shared libraries are loaded before the extension (flagged as missing in PR #3237 review by @astrelsky).    - Build a version-matched .so for each supported Python version (3.10, 3.11, 3.12, 3.13), not just 3.12.    - Install into the correct site-packages path with proper RPATH so liborigami.so.1 is found at runtime.

4. Artifact accounting: Scope all origami Python artifacts in artifacts-blas.toml correctly -- exclude Python files from dbg/dev/lib components, include them only in the test component (or a dedicated python component). The fix pattern is described in PR #4901's body.

5. Deprecate rocm-origami PyPI package: Once TheRock ships the official bindings, coordinate with the rocm-origami package maintainer to deprecate it, eliminating the dual-library collision described in ROCM-29472.

Related links

- ROCM-29472: Origami segfault caused by PyPI/SDK collision (Critical S1) - TheRock PR #5650: Current state -- origami as shared lib, Python disabled - TheRock #6425: hipDNN nanobind provider (prerequisite pattern) - TheRock PRs #2813, #3222, #3237, #3583, #3820, #4901: Prior land/revert history - PyPI package: https://pypi.org/project/rocm-origami/