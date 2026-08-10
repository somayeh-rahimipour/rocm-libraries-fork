"""Tests for verify_support_claims.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from verify_support_claims import verify_all


@pytest.fixture()
def bundle_root(tmp_path: Path) -> Path:
    return tmp_path


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Schema — single graph
# ---------------------------------------------------------------------------


class TestSingleGraphSchema:
    def test_valid_sidecar_passes(self, bundle_root: Path) -> None:
        _write_json(bundle_root / "A" / "Small.json", {})
        _write_json(
            bundle_root / "A" / "Small.meta.json",
            {"enforcement_level": "full"},
        )
        _write_json(
            bundle_root / "A" / "Small.support.json",
            {
                "version": 1,
                "claims": {"ENGINE": {"gfx942": ["linux"]}},
            },
        )
        assert verify_all(bundle_root) == []

    def test_bad_version_fails(self, bundle_root: Path) -> None:
        _write_json(bundle_root / "A" / "Small.json", {})
        _write_json(
            bundle_root / "A" / "Small.support.json",
            {"version": 2, "claims": {}},
        )
        errors = verify_all(bundle_root)
        assert any("version must be 1" in e for e in errors)

    def test_invalid_platform_fails(self, bundle_root: Path) -> None:
        _write_json(bundle_root / "A" / "Small.json", {})
        _write_json(
            bundle_root / "A" / "Small.support.json",
            {
                "version": 1,
                "claims": {"ENGINE": {"gfx942": ["macos"]}},
            },
        )
        errors = verify_all(bundle_root)
        assert any("invalid platform" in e for e in errors)

    def test_empty_claims_passes(self, bundle_root: Path) -> None:
        _write_json(bundle_root / "A" / "Small.json", {})
        _write_json(
            bundle_root / "A" / "Small.support.json",
            {"version": 1, "claims": {}},
        )
        assert verify_all(bundle_root) == []


# ---------------------------------------------------------------------------
# Schema — sweep
# ---------------------------------------------------------------------------


class TestSweepSchema:
    def test_valid_sweep_passes(self, bundle_root: Path) -> None:
        sweep_dir = bundle_root / "B" / "Default"
        _write_json(
            sweep_dir / "sweep.json",
            {"version": 1, "cases": [{"id": "c1", "values": {}}]},
        )
        _write_json(
            sweep_dir / "support.json",
            {
                "version": 1,
                "claims": {
                    "ENGINE": [
                        {
                            "cases": ["c1"],
                            "support": {"gfx942": ["linux"]},
                        }
                    ]
                },
            },
        )
        assert verify_all(bundle_root) == []

    def test_duplicate_case_id_fails(self, bundle_root: Path) -> None:
        sweep_dir = bundle_root / "B" / "Default"
        _write_json(
            sweep_dir / "sweep.json",
            {
                "version": 1,
                "cases": [
                    {"id": "c1", "values": {}},
                    {"id": "c2", "values": {}},
                ],
            },
        )
        _write_json(
            sweep_dir / "support.json",
            {
                "version": 1,
                "claims": {
                    "ENGINE": [
                        {
                            "cases": ["c1"],
                            "support": {"gfx942": ["linux"]},
                        },
                        {
                            "cases": ["c1"],
                            "support": {"gfx90a": ["linux"]},
                        },
                    ]
                },
            },
        )
        errors = verify_all(bundle_root)
        assert any("duplicate case id" in e for e in errors)

    def test_missing_cases_array_fails(self, bundle_root: Path) -> None:
        sweep_dir = bundle_root / "B" / "Default"
        _write_json(
            sweep_dir / "sweep.json",
            {"version": 1, "cases": [{"id": "c1", "values": {}}]},
        )
        _write_json(
            sweep_dir / "support.json",
            {
                "version": 1,
                "claims": {"ENGINE": [{"support": {"gfx942": ["linux"]}}]},
            },
        )
        errors = verify_all(bundle_root)
        assert any("non-empty 'cases' array" in e for e in errors)

    def test_missing_support_object_fails(self, bundle_root: Path) -> None:
        sweep_dir = bundle_root / "B" / "Default"
        _write_json(
            sweep_dir / "sweep.json",
            {"version": 1, "cases": [{"id": "c1", "values": {}}]},
        )
        _write_json(
            sweep_dir / "support.json",
            {
                "version": 1,
                "claims": {"ENGINE": [{"cases": ["c1"]}]},
            },
        )
        errors = verify_all(bundle_root)
        assert any("missing 'support' object" in e for e in errors)


# ---------------------------------------------------------------------------
# enforcement_level
# ---------------------------------------------------------------------------


class TestEnforcementLevel:
    def test_missing_meta_json_fails(self, bundle_root: Path) -> None:
        _write_json(bundle_root / "A" / "Small.json", {})
        _write_json(
            bundle_root / "A" / "Small.support.json",
            {
                "version": 1,
                "claims": {"ENGINE": {"gfx942": ["linux"]}},
            },
        )
        errors = verify_all(bundle_root)
        assert any("enforcement_level" in e for e in errors)

    def test_missing_enforcement_level_in_meta_fails(self, bundle_root: Path) -> None:
        _write_json(bundle_root / "A" / "Small.json", {})
        _write_json(bundle_root / "A" / "Small.meta.json", {"seed": 42})
        _write_json(
            bundle_root / "A" / "Small.support.json",
            {
                "version": 1,
                "claims": {"ENGINE": {"gfx942": ["linux"]}},
            },
        )
        errors = verify_all(bundle_root)
        assert any("enforcement_level is required" in e for e in errors)

    def test_invalid_enforcement_level_fails(self, bundle_root: Path) -> None:
        _write_json(bundle_root / "A" / "Small.json", {})
        _write_json(
            bundle_root / "A" / "Small.meta.json",
            {"enforcement_level": "ultra"},
        )
        _write_json(
            bundle_root / "A" / "Small.support.json",
            {
                "version": 1,
                "claims": {"ENGINE": {"gfx942": ["linux"]}},
            },
        )
        errors = verify_all(bundle_root)
        assert any("invalid enforcement_level" in e for e in errors)

    def test_valid_enforcement_level_passes(self, bundle_root: Path) -> None:
        for level in ("applicability", "buildable", "full"):
            d = bundle_root / level
            _write_json(d / "B.json", {})
            _write_json(d / "B.meta.json", {"enforcement_level": level})
            _write_json(
                d / "B.support.json",
                {
                    "version": 1,
                    "claims": {"ENGINE": {"gfx942": ["linux"]}},
                },
            )
        assert verify_all(bundle_root) == []

    def test_empty_claims_skips_enforcement_check(self, bundle_root: Path) -> None:
        _write_json(bundle_root / "A" / "Small.json", {})
        _write_json(
            bundle_root / "A" / "Small.support.json",
            {"version": 1, "claims": {}},
        )
        assert verify_all(bundle_root) == []


# ---------------------------------------------------------------------------
# Sweep case id cross-check
# ---------------------------------------------------------------------------


class TestSweepCaseIds:
    def test_orphan_case_id_fails(self, bundle_root: Path) -> None:
        sweep_dir = bundle_root / "B" / "Default"
        _write_json(
            sweep_dir / "sweep.json",
            {"version": 1, "cases": [{"id": "c1", "values": {}}]},
        )
        _write_json(
            sweep_dir / "support.json",
            {
                "version": 1,
                "claims": {
                    "ENGINE": [
                        {
                            "cases": ["c1", "c_nonexistent"],
                            "support": {"gfx942": ["linux"]},
                        }
                    ]
                },
            },
        )
        errors = verify_all(bundle_root)
        assert any("c_nonexistent" in e and "not found" in e for e in errors)

    def test_valid_case_ids_pass(self, bundle_root: Path) -> None:
        sweep_dir = bundle_root / "B" / "Default"
        _write_json(
            sweep_dir / "sweep.json",
            {
                "version": 1,
                "cases": [
                    {"id": "c1", "values": {}},
                    {"id": "c2", "values": {}},
                ],
            },
        )
        _write_json(
            sweep_dir / "support.json",
            {
                "version": 1,
                "claims": {
                    "ENGINE": [
                        {
                            "cases": ["c1", "c2"],
                            "support": {"gfx942": ["linux"]},
                        }
                    ]
                },
            },
        )
        assert verify_all(bundle_root) == []


# ---------------------------------------------------------------------------
# Orphaned sidecars
# ---------------------------------------------------------------------------


class TestOrphanedSidecars:
    def test_orphaned_single_graph_sidecar_fails(self, bundle_root: Path) -> None:
        _write_json(
            bundle_root / "A" / "Missing.support.json",
            {"version": 1, "claims": {}},
        )
        errors = verify_all(bundle_root)
        assert any("orphaned sidecar" in e for e in errors)

    def test_orphaned_sweep_support_json_fails(self, bundle_root: Path) -> None:
        _write_json(
            bundle_root / "B" / "NotASweep" / "support.json",
            {"version": 1, "claims": {}},
        )
        errors = verify_all(bundle_root)
        assert any("not a sweep root" in e for e in errors)

    def test_sweep_support_json_with_sweep_json_passes(self, bundle_root: Path) -> None:
        sweep_dir = bundle_root / "B" / "Default"
        _write_json(
            sweep_dir / "sweep.json",
            {"version": 1, "cases": [{"id": "c1", "values": {}}]},
        )
        _write_json(
            sweep_dir / "support.json",
            {"version": 1, "claims": {}},
        )
        assert verify_all(bundle_root) == []


# ---------------------------------------------------------------------------
# No sidecars at all
# ---------------------------------------------------------------------------


class TestNoSidecars:
    def test_empty_tree_passes(self, bundle_root: Path) -> None:
        assert verify_all(bundle_root) == []

    def test_nonexistent_root_passes(self, tmp_path: Path) -> None:
        assert verify_all(tmp_path / "does_not_exist") == []
