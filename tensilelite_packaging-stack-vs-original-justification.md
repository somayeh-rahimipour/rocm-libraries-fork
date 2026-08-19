# PR-17 reconstructed code-stack differences from the original branch

This file explains the remaining code and consumer-closure differences between:

- original branch tip: `26254111b82a6734053ac7d32ced5fff5e942295`
- rebuilt stack endpoint (PR 17): `27e1d83c21b1c8f5805e6018865ada8190e6615d`

The commit-split artifacts are in
`tensilelite_packaging-stack-vs-original-by-commit/`. Their subject-named patch
files collect the PR-17 code and consumer-closure differences by owning current
commit; `manifest.tsv` records each hunk's ownership, and `verify_split.sh`
regenerates and verifies the complete partition. This comparison deliberately
ends at PR 17. It excludes the conditional TheRock re-enable PRs 18--20, which
are above that endpoint and are not part of the original-branch reconstruction.
The ten standalone decision, implementation-plan, and research records are
intentionally deferred from PR 17 to the docs-only PR 21 above PR 20; the split
artifacts therefore remain a code and consumer-closure audit. The rebuilt stack
is deliberately not byte-identical to the original tip: these are correctness
and consumer-closure fixes discovered while splitting the history. No feature
from the original branch was intentionally omitted.

## Installed workflow and public guidance

1. `.github/scripts/run_rocjitsu_hipblaslt_race_check.sh`

   The original runner invokes a deleted source `Tensile/bin/Tensile` driver and
   passes the removed `--prebuilt-client` option. It now installs the staged
   canonical wheel, binds the staged libexec client, and invokes
   `python -m tensilelite run` under rocjitsu.

2. `.github/workflows/component-ci-tensilelite-coverage.yml`

   The coverage workflow now describes the current coverage-unit behavior
   (client build plus binding) and calls the moved lower-case characterization
   summary script. The original path is silently ignored by `|| true`.

3. `projects/hipblaslt/.agent/docs/targets.json`

   Preserves the lower-case package and test roots introduced earlier in the
   stack. The original final tip incorrectly restores nonexistent `Tensile/`
   paths.

4. `projects/hipblaslt/CONTRIBUTING.md`

   Replaces the stale internal/uppercase generator reference with the supported
   public command `python -m tensilelite create-library`, while preserving the
   lower-case test paths.

5. `projects/hipblaslt/README.md`

   Replaces the deleted `Tensile/bin/TensileCreateLibrary` command with the
   current public create-library command.

6. `projects/hipblaslt/tensilelite/AGENTS.md`

   Documents the real `tensilelite()` function name and the current tox unit
   workflow, which installs editable packages, builds the client, and configures
   the binding before tests.

7. `projects/hipblaslt/tensilelite/AGENTS_reference.md`

   Updates the unit-test description and lower-case create-library directory
   reference so developer instructions match the current test and source layout.

8. `projects/hipblaslt/tensilelite/PythonBuildGrillingDecisions.md`

   Removes trailing Markdown whitespace on metadata lines so the final stack
   passes `git diff --check`; no decision content changes.

9. `projects/hipblaslt/tensilelite/README.md`

   States that strict version validation and Python-SDK runtime support are
   present but currently gated off pending TheRock support. It also gives the
   required explicit ROCm identity for manual editable installs.

10. `projects/hipblaslt/tensilelite/docs/PackagingDecisions.md`

    Removes trailing Markdown whitespace on status metadata. No decision
    content changes.

11. `projects/hipblaslt/tensilelite/docs/PackagingPlan.md`

    Removes trailing Markdown whitespace on plan metadata. No plan content
    changes.

12. `projects/hipblaslt/tensilelite/scripts/precommit_affected_tests.py`

    The original failure-report path interpolates undefined `runtime_root` and
    raises `NameError` when offering a snapshot-update command. It now uses the
    already-constructed `test_env["ROCM_PATH"]`.

13. `projects/hipblaslt/tensilelite/setup.py`

    Retains strict explicit `TENSILELITE_ROCM_VERSION` packaging, but gives tox
    package setup a narrowly scoped fallback to its selected
    `ROCM_PATH/.info/version`. This is needed because tox constructs an isolated
    editable package before its commands run. Normal wheel builds still fail
    without an explicit identity.

14. `projects/hipblaslt/tensilelite/tasks.py`

    Keeps the selected resolved ROCm root when invoking `build_client` from
    `invoke install`; the original final code passes the raw optional argument,
    which can select a different SDK from rocisa. The remaining changes are
    small formatting/help-text alignment with the final workflow.

## Runtime, generator, and test correctness

15. `projects/hipblaslt/tensilelite/tensilelite/Tests/unit/characterization/GenerateSummations/test_generate_summations_char.py`

    Removes stale pandas/mock assumptions after the generator moved to stdlib
    CSV plus NumPy. The characterization now describes the actual dispatch
    behavior and has no unused pandas-era imports.

16. `projects/hipblaslt/tensilelite/tensilelite/Tests/unit/characterization/Naming/test_mut_Naming_char.py`

    Retains the lower-case module path in the characterization fixture rather
    than reintroducing the removed uppercase namespace.

17. `projects/hipblaslt/tensilelite/tensilelite/Tests/unit/characterization/ParseArguments/test_parse_arguments_char.py`

    Keeps characterization expectations aligned with the removed
    `--prebuilt-client` option and the canonical command surface.

18. `projects/hipblaslt/tensilelite/tensilelite/Tests/unit/characterization/PublicInputSurface/test_pchaos_Tensile_L25_char.py`

    Updates the direct-execution redirect assertion to `python -m tensilelite
    run`, replacing the deleted source-bin launcher.

19. `projects/hipblaslt/tensilelite/tensilelite/Tests/unit/characterization/PublicInputSurfaceDeep/test_pchaos_Validators_L97_char.py`

    Replaces characterization of deleted ambient `_posixSearchPaths` behavior
    with the active selected-installation search-path and missing-tool diagnostic
    contracts.

20. `projects/hipblaslt/tensilelite/tensilelite/Tests/unit/characterization/SolutionStructsUtils/test_mut_Utilities_reject_char.py`

    Keeps the lower-case import/module target introduced by the namespace split.

21. `projects/hipblaslt/tensilelite/tensilelite/Tests/unit/characterization/ToolchainValidators/test_toolchain_validators_char.py`

    Replaces removed ambient Windows/POSIX search helper tests with direct live
    coverage for selected-runtime paths, absolute and relative executable
    validation, scalar/tuple public returns, accepted compiler/HIP/bundler/
    enumerator aliases, rejected near-misses, and the RHEL fallback path.

22. `projects/hipblaslt/tensilelite/tensilelite/Tests/unit/characterization/_codegen/char_paths.py`

    Retains lower-case package-root path resolution for installed artifacts;
    the original final text/path refers to a nonexistent `Tensile` package.

23. `projects/hipblaslt/tensilelite/tensilelite/Tests/unit/characterization/_codegen/test_r3_streamk_tdm_sgpr_budget_gfx1250_char.py`

    Keeps lower-case imports for live codegen coverage instead of stale
    `Tensile.*` imports.

24. `projects/hipblaslt/tensilelite/tensilelite/Tests/unit/characterization/coverage-baseline.json`

    Tracks the replacement characterization coverage after validator/search-path
    APIs changed. The coverage floor itself was not lowered or disabled.

25. `projects/hipblaslt/tensilelite/tensilelite/Tests/unit/source_only/README.md`

    Documents the actual source-only classification and avoids claiming stale
    installed-artifact behavior before it exists.

26. `projects/hipblaslt/tensilelite/tensilelite/Tests/unit/source_only/test_install_task.py`

    Adds direct orchestration coverage for one resolved ROCm root, the actual
    CMake client output path, editable install flags, explicit binding, missing
    or non-executable client rejection, and Windows naming/Linux-only handling.

27. `projects/hipblaslt/tensilelite/tensilelite/Tests/unit/source_only/test_precommit_affected_tests.py`

    Matches the installed/bound precommit workflow and protects the repaired
    precommit helper behavior.

28. `projects/hipblaslt/tensilelite/tensilelite/Tests/unit/source_only/test_release_metadata.py`

    Adds explicit ROCm identity, development-publication identity, and tox
    package-bootstrap coverage for the final metadata contract.

29. `projects/hipblaslt/tensilelite/tensilelite/Tests/unit/source_only/test_specs_amdsmi.py`

    Keeps the correct moved lower-case source path.

30. `projects/hipblaslt/tensilelite/tensilelite/Tests/unit/test_TensileLogic_Run.py`

    Retains explicit `_setup(argv)` and `main(argv)` coverage. The canonical CLI
    calls `main(args)`, so deleting these tests would lose a live contract.

31. `projects/hipblaslt/tensilelite/tensilelite/Tests/unit/test_generate_summations_csv.py`

    Keeps the stronger regression fixture: first-seen duplicate `SizeL` order,
    multiple `Cij*` columns, and NaN-aware maximum behavior.

32. `projects/hipblaslt/tensilelite/tensilelite/Tests/unit/test_gfx1250_asic_revision.py`

    Preserves all lower-case imports after the original final tip reintroduced
    dozens of stale `Tensile.*` imports. The full test module still collects 80
    cases under the rebuilt namespace.

33. `projects/hipblaslt/tensilelite/tensilelite/Tests/unit/test_rocm_runtime.py`

    Verifies the additional conventional-prefix libexec client search directory.

34. `projects/hipblaslt/tensilelite/tensilelite/__init__.py`

    Adds the final trailing newline. No runtime behavior changes.

35. `projects/hipblaslt/tensilelite/tensilelite/_rocm.py`

    Adds `<ROCM_PATH>/libexec/hipblaslt/tensilelite` to conventional-prefix
    executable search paths. This is required after removing explicit
    prebuilt-client forwarding because artifacts stage the client there.

36. `projects/hipblaslt/tensilelite/tensilelite/tensilelite.py`

    Updates a stale `Tensile()` code comment to the real `tensilelite()`
    function name.

37. `projects/hipblaslt/tensilelite/tensilelite/tensilelite_create_library/run.py`

    Removes duplicated legacy license blocks and a duplicate `printWarning`
    import inserted mid-module by the original final history. Functional code is
    unchanged by this cleanup.

## GEKO installed-interface closure

38. `projects/hipblaslt/utilities/geko/README.md`

    Replaces the nonexistent `tensilelite mergelibrary` command with the real
    installed `tensilelite.merge_library.avoidRegressions` API, including a
    working shell-variable example.

39. `projects/hipblaslt/utilities/geko/geko/config_generator/README.md`

    States that generated workflows bind the built client to the active
    installed Python package; they do not stage a source-tree client path.

40. `projects/hipblaslt/utilities/geko/geko/library/operations.py`

    Removes the residual `sys.path.append(<checkout>/tensilelite)` fallback from
    `normalize()`. `hipblaslt_path` remains validation-only and imports always
    come from the active installed environment.

41. `projects/hipblaslt/utilities/geko/geko/utils.py`

    Supplies an explicit ROCm identity when GEKO editable-installs TensileLite,
    rejects explicitly empty `ROCM_PATH`, and forwards that environment through
    the client build/install/binding sequence.

42. `projects/hipblaslt/utilities/geko/tests/conftest.py`

    Requires installed `tensilelite` and `rocisa` for integration fixtures
    instead of injecting a checkout into `sys.path`.

43. `projects/hipblaslt/utilities/geko/tests/test_library_operations_unit.py`

    Verifies that `normalize()` no longer mutates `sys.path` while preserving
    its optional hipBLASLt-path validation behavior.

44. `projects/hipblaslt/utilities/geko/tests/test_utils_unit.py`

    Covers the exact active-interpreter build, editable-install, binding, cache,
    explicit identity, fallback identity, missing metadata, and empty-ROCM_PATH
    contracts added in `geko.utils`.

## Validation record

- `invoke build-client` passed at the PR 17 endpoint.
- `ROCM_PATH=/opt/rocm TENSILELITE_ROCM_VERSION=7.2.4 python -m pip install -e .` passed.
- Package and client both reported `5.0.0+rocm7.2.4`.
- Focused TensileLite suite: `125 passed`.
- Full GEKO fast suite: `251 passed, 20 skipped`.
- `git diff --check 26254111b8 27e1d83c21` passed.
- The original branch remains unchanged at `26254111b82a6734053ac7d32ced5fff5e942295`.
