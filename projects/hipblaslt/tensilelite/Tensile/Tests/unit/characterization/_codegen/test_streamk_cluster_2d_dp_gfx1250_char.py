# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 StreamK=3 ForceDPOnly=0 ClusterDim=[2,2] 2-D DP multicast codegen.

2-D spatial A+B multicast in the DP window (same meaning as ForceDPOnly=1),
ordinary SK-tail loads, cluster-barrier reduction of SK partials.
"""

import os

import pytest

from config_harness import (
    assert_assembles,
    assert_real_gfx1250_kernels,
    emit_kernels_from_config,
)

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1250",
    "streamk_cluster_2d_dp.yaml",
)


def test_streamk_cluster_2d_dp_gfx1250_emits_assembly():
    results = emit_kernels_from_config(_CONFIG, limit=4, arch=_ARCH)
    assert_real_gfx1250_kernels(results)
    for base, src, _err in results:
        assert_assembles(src, base)
        assert "RemapWorkGroupDone" in src, (
            f"Kernel {base!r}: missing cluster WG-id decode"
        )
        assert "workaround" not in src, (
            f"Kernel {base!r}: ttmp reread emitted under ClusterDim != [1, 1]"
        )
        assert "DP fold: StreamKIdx = batch*(nWG0*nWG1) + N*nWG0 + M" in src, (
            f"Kernel {base!r}: missing M-fastest 2-D DP StreamK index fold"
        )
        assert "2-D cluster: StreamKIdx = WorkGroup0*Ck + WorkGroup1" not in src, (
            f"Kernel {base!r}: 2-D DP multicast must not use the K-fastest [1,C] fold"
        )
        assert "StreamKFactored: B-multicast along Cs=" not in src, (
            f"Kernel {base!r}: 2-D DP multicast must not emit K-slice B-masks"
        )
        assert "k = StreamKIdx & (Ck-1)" not in src, (
            f"Kernel {base!r}: 2-D DP multicast must not decode a K-slice rank from StreamKIdx"
        )
        assert "remap StreamKIdx to cluster-linear SK rank" in src, (
            f"Kernel {base!r}: missing DP->SK cluster-linear StreamKIdx remap"
        )
        assert "clear BOTH A & B broadcast masks at DP->SK boundary" in src, (
            f"Kernel {base!r}: missing DP->SK dual-mask clear"
        )
        assert "cluster_barrier signal (arrive)" in src, (
            f"Kernel {base!r}: missing multicast / reduction cluster arrive"
        )
        assert "cluster_barrier wait (all peers arrived)" in src, (
            f"Kernel {base!r}: missing SK-tail cluster-reduction wait"
        )
        assert "s_barrier_signal -3" in src, (
            f"Kernel {base!r}: missing cluster-scope barrier signal (-3)"
        )
        assert "s_barrier_wait -3" in src, (
            f"Kernel {base!r}: missing cluster-scope barrier wait (-3)"
        )
