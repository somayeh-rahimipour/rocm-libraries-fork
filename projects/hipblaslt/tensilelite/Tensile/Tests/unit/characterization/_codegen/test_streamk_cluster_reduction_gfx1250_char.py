# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 StreamK=3 ForceDPOnly=0 ClusterDim=[1,2] cluster-reduction codegen."""

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
    "streamk_cluster_reduction.yaml",
)


def test_streamk_cluster_reduction_gfx1250_emits_assembly():
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
        assert "cluster_barrier signal (arrive)" in src, (
            f"Kernel {base!r}: missing cluster-reduction arrive"
        )
        assert "cluster_barrier wait (all peers arrived)" in src, (
            f"Kernel {base!r}: missing cluster-reduction wait"
        )
        assert "cluster_last" in src, (
            f"Kernel {base!r}: missing intra-cluster predicate"
        )
        assert "MulticastMaskB" not in src, (
            f"Kernel {base!r}: pure reduction must not emit B multicast masks"
        )
