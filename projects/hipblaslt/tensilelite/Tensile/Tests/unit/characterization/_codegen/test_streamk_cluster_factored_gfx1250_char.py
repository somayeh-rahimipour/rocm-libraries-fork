# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 StreamK=3 ForceDPOnly=0 ClusterDim=[2,2] factored cluster codegen.

B-multicast along Cs plus K-split along Ck. Multicast owns the cluster
barrier; the K-split reduction uses the global-flag path.
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
    "streamk_cluster_factored.yaml",
)


def test_streamk_cluster_factored_gfx1250_emits_assembly():
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
        assert "2-D cluster: StreamKIdx = WorkGroup0*Ck + WorkGroup1" in src, (
            f"Kernel {base!r}: missing 2-D StreamK index fold"
        )
        assert "StreamKFactored: B-multicast along Cs=" in src, (
            f"Kernel {base!r}: missing factored B-multicast mask compute"
        )
        assert "maskB_base=0x3 (shifted by k*Cs)" in src, (
            f"Kernel {base!r}: factored B-mask must be X-fast ((1<<Cs)-1) << (k*Cs), not StreamK-linear"
        )
        assert "cluster_barrier signal (arrive)" in src, (
            f"Kernel {base!r}: missing multicast prologue cluster arrive"
        )
        assert "cluster_barrier wait (all peers arrived)" not in src, (
            f"Kernel {base!r}: factored multicast already owns -3; K-split "
            "reduction must use the global-flag path, not a second cluster handshake"
        )
        assert "s_barrier_signal -3" in src, (
            f"Kernel {base!r}: missing cluster-scope barrier signal (-3)"
        )
        assert "s_barrier_wait -3" in src, (
            f"Kernel {base!r}: missing cluster-scope barrier wait (-3)"
        )
