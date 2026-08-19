#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Unit tests for StreamK=3 ForceDPOnly=0 cluster reduction and
# cluster multicast+reduction.
#
# ClusterDim = [Cs, Ck] on the two-tile path:
#   [C, 1] -> existing non-multicast 1-D SK3 cluster
#   [1, C] -> pure K-split cluster reduction (cluster-barrier fast path)
#   [Cs,Ck] both > 1 -> cluster multicast+reduction (A+B multicast in DP,
#                        cluster-barrier reduction on the SK tail)
#
# Usage:
#   pytest test_streamk_cluster_reduction.py -v
################################################################################

import copy
import os

import pytest

pytestmark = pytest.mark.unit

from Tensile.Common import (
    streamKClusterFactors,
    streamKClusterReduction,
    streamKMulticast,
    streamK2DMulticast,
)

_DESIGNED = os.path.join(
    os.path.dirname(__file__), "characterization",
    "_codegen", "data", "test_data", "_designed", "gfx1250")
_STREAMK_MULTICAST = os.path.join(_DESIGNED, "streamk_cluster_multicast.yaml")

_ARCH = "gfx1250"


def _write_variant(tmp_path, name, *, fork_overrides=None):
    from Tensile import LibraryIO
    import yaml

    cfg = copy.deepcopy(LibraryIO.read(_STREAMK_MULTICAST))
    if fork_overrides:
        fork = cfg["BenchmarkProblems"][0][1]["ForkParameters"]
        for key, val in fork_overrides.items():
            replaced = False
            for entry in fork:
                if key in entry:
                    entry[key] = val
                    replaced = True
                    break
            if not replaced:
                fork.append({key: val})
    out = tmp_path / name
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=None)
    return str(out)


def _derive_states(cfg_path):
    from config_harness import derive_states
    return derive_states(cfg_path, arch=_ARCH, limit_solutions=8)


class TestPredicates:
    def test_factors(self):
        assert streamKClusterFactors({"ClusterDim": [4, 1]}) == (4, 1, 4, False)
        assert streamKClusterFactors({"ClusterDim": [1, 4]}) == (1, 4, 4, True)
        assert streamKClusterFactors({"ClusterDim": [2, 2]}) == (2, 2, 4, True)

    def test_fdpo1_still_multicast(self):
        st = {"StreamK": 3, "StreamKForceDPOnly": 1, "ClusterDim": [4, 1]}
        assert streamKMulticast(st)
        assert not streamKClusterReduction(st)
        assert not streamK2DMulticast(st)
        st22 = {"StreamK": 3, "StreamKForceDPOnly": 1, "ClusterDim": [2, 2]}
        assert streamKMulticast(st22)
        assert streamK2DMulticast(st22)
        assert not streamKClusterReduction(st22)

    def test_fdpo0_cs_only_is_not_multicast(self):
        st = {"StreamK": 3, "StreamKForceDPOnly": 0, "ClusterDim": [4, 1]}
        assert not streamKMulticast(st)
        assert not streamKClusterReduction(st)
        assert not streamK2DMulticast(st)

    def test_fdpo0_pure_reduction(self):
        st = {"StreamK": 3, "StreamKForceDPOnly": 0, "ClusterDim": [1, 4]}
        assert not streamKMulticast(st)
        assert streamKClusterReduction(st)
        assert not streamK2DMulticast(st)

    def test_fdpo0_cluster_multicast_reduction(self):
        st = {"StreamK": 3, "StreamKForceDPOnly": 0, "ClusterDim": [2, 2]}
        assert streamKMulticast(st)
        assert streamKClusterReduction(st)
        assert streamK2DMulticast(st)


class TestDerivation:
    def test_pure_reduction_derives(self, tmp_path):
        cfg = _write_variant(tmp_path, "red.yaml",
                             fork_overrides={
                                 "StreamKForceDPOnly": [0],
                                 "ClusterDim": [[1, 2]],
                                 "StreamKFixupTreeReduction": [0],
                             })
        states = _derive_states(cfg)
        assert states, "[1,2] ForceDPOnly=0 must derive as cluster reduction"
        for st in states:
            assert st["ClusterDim"] == [1, 2]
            assert streamKClusterReduction(st)
            assert not streamKMulticast(st)
            assert st["Multicast"] is False
            assert st["ClusterBarrier"] is False

    def test_cluster_multicast_reduction_derives(self, tmp_path):
        cfg = _write_variant(tmp_path, "multicast_reduction.yaml",
                             fork_overrides={
                                 "StreamKForceDPOnly": [0],
                                 "ClusterDim": [[2, 2]],
                                 "StreamKFixupTreeReduction": [0],
                             })
        states = _derive_states(cfg)
        assert states, "[2,2] ForceDPOnly=0 must derive as cluster multicast+reduction"
        for st in states:
            assert st["ClusterDim"] == [2, 2]
            assert streamKClusterReduction(st)
            assert streamKMulticast(st)
            assert streamK2DMulticast(st)
            assert st["Multicast"] is True
            assert st["ClusterBarrier"] is True

    def test_fdpo0_1d_cluster_unchanged(self, tmp_path):
        cfg = _write_variant(tmp_path, "sk1d.yaml",
                             fork_overrides={
                                 "StreamKForceDPOnly": [0],
                                 "ClusterDim": [[4, 1]],
                             })
        states = _derive_states(cfg)
        assert states, "ForceDPOnly=0 [4,1] must still derive (non-multicast SK3 cluster)"
        for st in states:
            assert not streamKMulticast(st)
            assert not streamKClusterReduction(st)
            assert st["Multicast"] is False

    def test_fdpo1_ck_only_still_rejected(self, tmp_path):
        cfg = _write_variant(tmp_path, "fdpo1_1c.yaml",
                             fork_overrides={"ClusterDim": [[1, 2]]})
        assert _derive_states(cfg) == []

    def test_reject_tree_reduction(self, tmp_path):
        cfg = _write_variant(tmp_path, "tree.yaml",
                             fork_overrides={
                                 "StreamKForceDPOnly": [0],
                                 "ClusterDim": [[1, 2]],
                                 "StreamKFixupTreeReduction": [1],
                             })
        assert _derive_states(cfg) == []
