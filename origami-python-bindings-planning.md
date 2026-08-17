# Origami Python Bindings: Planning Research

## Status

- Planning phase: Phase 1 complete; Phase 2 **complete** (gating experiment run
  and observed; the two owner decisions made). The implementation plan is
  written: see [Step 7 implementation plan](#step-7-implementation-plan-in-tree-per-version-wheel).
  See also [Phase 2 findings](#phase-2-findings) and
  [Phase 2 verdict](#phase-2-verdict).
- Source ticket: [`ticket.md`](./ticket.md)
- Implementation status: No implementation changes have been made
- Evidence convention:
  - **verified** — supported by an inspected primary source or the current repository
  - **summary-only** — useful for orientation, but not established as factual support
  - **speculative** — a proposed interpretation or direction that still needs evidence
  - **UNKNOWN** — explicitly unresolved

## Goal

Provide official Origami Python bindings that consume the ROCm SDK's shared
`liborigami`, eliminate the need for the `rocm-origami` workaround package, and
avoid loading two incompatible Origami implementations in one process.

This document records the first planning/research phase. It does not yet prescribe
an implementation. Phase 1 found that the ticket's proposed prerequisite and
delivery path have changed upstream, so the packaging architecture and Python ABI
strategy must be resolved before an implementation plan is written.

## Phase 1 findings

### 1. TheRock currently builds only Origami's shared C++ library

**Evidence status: verified**

Current TheRock configuration builds Origami with:

```text
-DORIGAMI_BUILD_SHARED_LIBS=ON
-DORIGAMI_ENABLE_PYTHON=OFF
-DORIGAMI_ENABLE_FETCH=OFF
```

The configuration says bindings are built outside TheRock pending a home and a
nanobind provider. TheRock PR #5650 also states that Python bindings were disabled
and deferred.

Sources:

- TheRock PR #5650: <https://github.com/ROCm/TheRock/pull/5650>
- Current TheRock `math-libs/BLAS/CMakeLists.txt`, Origami declaration, lines
  95-126 at the time of research

### 2. TheRock PR #6425 did not land a reusable nanobind provider

**Evidence status: verified**

The ticket describes #6425 as a prerequisite pattern for a nanobind provider. The
merged PR did the opposite: it removed TheRock's nanobind and robin-map providers,
hipDNN Python binding builds, wheel packaging, and binding tests. Its rationale
states that the new architecture uses a different repository for bindings.

This invalidates the assumption that the immediate next step is simply to add a
nanobind provider to TheRock and turn `ORIGAMI_ENABLE_PYTHON` on.

Source:

- TheRock PR #6425: <https://github.com/ROCm/TheRock/pull/6425>

### 3. Origami's current bindings already use nanobind

**Evidence status: verified**

The local Origami Python build first searches for an installed nanobind package.
If it cannot find one, it fetches nanobind v2.0.0. TheRock disables dependency
fetching, so enabling the bindings in the current TheRock build would require a
provided nanobind package or a different build boundary.

Sources:

- [`shared/origami/python/CMakeLists.txt`](./shared/origami/python/CMakeLists.txt),
  lines 34-44
- [`shared/origami/CMakeLists.txt`](./shared/origami/CMakeLists.txt), lines 112-114

### 4. The Python package does not preload ROCm libraries

**Evidence status: verified**

The package imports its compiled extension immediately. It does not call
`rocm_sdk.preload_libraries()` first.

Review discussion on TheRock PR #3237 explicitly says that Origami's `__init__.py`
needs to use the `rocm_sdk` package function to ensure ROCm libraries are loaded
before the extension.

Sources:

- [`shared/origami/python/src/origami/__init__.py`](./shared/origami/python/src/origami/__init__.py),
  lines 10-97
- TheRock PR #3237: <https://github.com/ROCm/TheRock/pull/3237>

### 5. The Python-version compatibility failure is real, but the solution is open

**Evidence status: verified for the failure; UNKNOWN for the final solution**

TheRock issue #3581 records Python 3.10 and 3.11 processes attempting to load an
extension built for CPython 3.12 and failing with:

```text
undefined symbol: PyObject_Vectorcall
```

The issue discussion recommends a proper Python wheel using the limited or stable
Python ABI if possible, instead of placing a CPython-version-specific extension in
a Python-version-agnostic SDK package.

The ticket currently requires separate builds for Python 3.10, 3.11, 3.12, and
3.13. That is one possible solution, but it is not yet established as the preferred
one. Whether Origami's current nanobind bindings can use a stable ABI across all
supported versions remains UNKNOWN and requires documentation review plus a build
experiment.

Source:

- TheRock issue #3581: <https://github.com/ROCm/TheRock/issues/3581>

### 6. The original RPATH and SDK registration failures are supported

**Evidence status: verified**

TheRock issue #3220 records failure to load `liborigami.so.1`. PR #3820 attributes
the failure to two integration gaps:

1. The Python extension lacked an RPATH to the installed Origami library.
2. Origami was not registered in `rocm_sdk` distribution metadata.

The current local Origami Python build sets an install RPATH. Current TheRock
distribution metadata registers `liborigami` as a public library.

Sources:

- TheRock issue #3220: <https://github.com/ROCm/TheRock/issues/3220>
- TheRock PR #3820: <https://github.com/ROCm/TheRock/pull/3820>
- [`shared/origami/python/CMakeLists.txt`](./shared/origami/python/CMakeLists.txt),
  lines 87-95
- Current TheRock `rocm_sdk/_dist_info.py`

### 7. Artifact accounting caused the third revert

**Evidence status: verified**

TheRock PR #4901 says SDK tests still failed and proposes excluding Origami's
`lib/python*/**` files from `dbg`, `dev`, and `lib`, while including them in the
`test` component.

That correction explains the prior failure, but putting the bindings only in a
test component does not by itself define a supported user-facing distribution.
A dedicated Python wheel or Python artifact component is a plausible answer, but
that conclusion remains speculative until the current TheRock binding architecture
is confirmed.

Source:

- TheRock PR #4901: <https://github.com/ROCm/TheRock/pull/4901>

### 8. `rocm-origami` 0.0.2 builds its own Origami implementation

**Evidence status: verified**

PyPI publishes `rocm-origami` 0.0.2 as a source distribution. The source archive
contains Origami's C++ sources, headers, Python bindings, and build system. Its
Python build adds the bundled parent Origami project; in that nested configuration,
Origami defaults to a static library, and the nanobind extension links that target.

This supports the claim that an installed extension built from the package can
carry its own Origami implementation instead of consuming the SDK's shared
`liborigami`.

Sources:

- PyPI release: <https://pypi.org/project/rocm-origami/0.0.2/>
- Official `rocm_origami-0.0.2.tar.gz` source distribution

### 9. The exact incident diagnosis remains unverified

**Evidence status: summary-only**

The following claims currently come from `ticket.md` and reportedly ROCM-29472:

- the older implementation allocated 112 bytes for a structure that grew to 128
  bytes;
- the mismatch caused heap corruption and PyTorch CI segfaults;
- the incident is classified Critical S1.

The internal ROCM-29472 evidence was not available during Phase 1, and the public
sources inspected do not independently prove those exact sizes or the full causal
chain. These claims must not be promoted to verified plan rationale until the issue
is read or the ABI mismatch is reproduced.

## Architectural correction

The evidence does not currently support enabling a CPython-specific extension
inside TheRock's Python-version-agnostic `rocm-sdk-libraries` package.

The leading architecture to investigate in Phase 2 is:

```text
TheRock / ROCm SDK
└── liborigami.so.1

Official Origami Python distribution
├── pure Python package
├── nanobind extension
├── explicit Python ABI strategy
├── ROCm library preload integration
└── dynamic dependency on the SDK-provided liborigami
```

**Evidence status: speculative recommendation.** This direction fits the evidence
from TheRock #6425 and #3581, but the owning repository, artifact boundary, and
supported installation contract must be confirmed before it becomes a decision.

## Phase 2 objective

Resolve the delivery architecture and Python ABI strategy sufficiently to write a
grounded implementation plan. Phase 2 should produce decisions, not code.

## Phase 2 work plan

### Step 1: Confirm the official binding-distribution architecture

Determine where ROCm Python bindings are expected to live after TheRock #6425.

Questions to answer:

- What is the separate bindings repository mentioned by #6425?
- Is Origami expected to use it?
- Is TheRock expected to build the binding, package an externally built wheel,
  test it, or only provide `liborigami`?
- Is a dedicated Python component permitted inside TheRock, or explicitly
  discouraged?

Required evidence:

- Current first-party design or repository documentation
- A current example binding with the intended ownership and packaging boundary
- Confirmation from the responsible TheRock or Python-packaging owner if the
  documented direction is incomplete

Exit criterion:

- One selected artifact boundary with named owning repository and build system

### Step 2: Choose the Python ABI strategy

Compare these options:

1. One stable-ABI/limited-API extension supporting all required Python versions
2. One extension wheel per CPython version
3. Another supported nanobind packaging model identified by current documentation

For each option, verify:

- nanobind support and restrictions;
- compatibility with every bound Origami type and function;
- wheel tags and installation behavior;
- build-matrix and CI cost;
- compatibility with Python 3.10, 3.11, 3.12, and 3.13;
- Windows implications, if Windows remains in scope.

Diligence procedure:

- Survey primary nanobind and Python packaging documentation.
- Select at least two relevant primary sources with different scopes or methods.
- Extract a direct quote or specific compatibility statement for every
  load-bearing ABI claim.
- Build a minimal proof-of-concept extension for the leading option.
- Test import under each supported Python version.

Exit criterion:

- A selected ABI strategy backed by documentation and a reproducible experiment

### Step 3: Define the shared-library loading contract

The official extension must consume the SDK's `liborigami`, not compile another
copy into itself.

Determine:

- whether `rocm_sdk` is a mandatory runtime dependency;
- the exact `preload_libraries()` call and required library names;
- Linux RPATH relative to the final wheel/SDK installation layout;
- Windows DLL discovery behavior;
- behavior when ROCm is installed outside the wheel-based SDK;
- whether import should fail with an actionable error when the runtime is absent.

Verification should include:

- inspecting `DT_NEEDED`/RPATH on Linux or imports on Windows;
- proving that the extension resolves the SDK-provided library;
- proving that no second Origami implementation is statically present;
- importing the module before and after other consumers such as hipBLASLt or
  PyTorch.

Exit criterion:

- A written loader contract with platform-specific observable tests

### Step 4: Define artifact ownership and wheel contents

Map every generated file to exactly one artifact component or wheel:

- Python package sources;
- compiled extension;
- metadata and license files;
- tests, if shipped;
- shared library, headers, debug files, and development files.

Verify that Python files do not leak into `dbg`, `dev`, or generic `lib`
components and are not silently dropped. Decide whether test files belong in a
test component while the importable package belongs in a dedicated wheel or
Python component.

Exit criterion:

- A complete file-to-artifact table with no overlap and no unclaimed files

### Step 5: Verify the collision incident

Obtain ROCM-29472 or reproduce the relevant mismatch.

Confirm:

- the exact types and versions involved;
- the reported 112-byte and 128-byte sizes;
- how the older and newer implementations enter one process;
- the allocation and overwrite path;
- whether dynamically linking the official binding to the SDK library eliminates
  that path.

Exit criterion:

- Either a verified causal chain with logs/measurements or a weakened incident
  statement that accurately reflects the available evidence

### Step 6: Establish deprecation and migration ownership

PyPI metadata currently identifies AMD-OSS as the owning organization and lists
AMD maintainers. Determine:

- whether the existing `rocm-origami` project should be replaced in place,
  transferred, yanked, or marked deprecated;
- the official successor package name;
- the migration message and release sequence;
- how existing users avoid silently installing the old source distribution.

Exit criterion:

- Named owners and a release/deprecation sequence that prevents an overlap window

### Step 7: Produce the implementation plan

After Steps 1-6 are resolved, write a grounded implementation plan containing:

- selected architecture and rejected alternatives;
- exact repositories and files to modify;
- dependency-provider or external-wheel integration changes;
- ABI and build-matrix changes;
- preload and dynamic-linking changes;
- artifact-accounting rules;
- tests for all supported Python versions and platforms;
- collision regression test;
- rollout, deprecation, and rollback sequence;
- measurable completion criteria.

Every load-bearing repository claim must cite `file:line`. Every external claim
must retain its diligence status and primary-source evidence.

## Phase 2 decision record template

Use this table while researching so unresolved questions do not silently turn into
assumptions:

| Decision | Options considered | Evidence | Selected option | Owner | Status |
| --- | --- | --- | --- | --- | --- |
| Binding repository | | | | | OPEN |
| Artifact boundary | | | | | OPEN |
| Python ABI | Stable ABI / per-version / other | | | | OPEN |
| Linux loader contract | | | | | OPEN |
| Windows loader contract | | | | | OPEN |
| `rocm_sdk` dependency | Required / optional / absent | | | | OPEN |
| Package name | | | | | OPEN |
| PyPI migration | Replace / deprecate / yank / other | | | | OPEN |

## Risks to carry forward

- Repeating the prior approach without resolving the artifact boundary can cause a
  fourth land/revert cycle.
- Building under one CPython version and placing the result in a
  Python-version-agnostic artifact recreates the #3581 failure class.
- Correct RPATH alone does not prevent two Origami implementations from entering
  one process.
- Calling `preload_libraries()` does not by itself prove that the extension links
  the intended `liborigami`.
- Putting importable bindings only in a test component does not deliver an
  official user-facing package.
- Deprecating the old package before the successor is installable across supported
  environments creates a migration gap; doing it too late preserves the collision
  window.

## Definition of Phase 2 complete

Phase 2 is complete when all of the following are true:

- the owning repository and artifact boundary are confirmed;
- the Python ABI strategy is verified by documentation and experiment;
- the extension's dynamic-link and preload contract is specified for supported
  platforms;
- artifact ownership is exhaustive and non-overlapping;
- the incident diagnosis is either verified or accurately weakened;
- deprecation owners and sequencing are identified;
- enough evidence exists to write a file-level implementation plan without
  unresolved architecture decisions.

---

# Phase 2 findings

## Summary

Phase 2 is now **complete**. The three blockers the research left open have been
cleared: the gating build experiment ran and confirmed the static-embed premise
by observation (see [Result](#result-observed-2026-08-17-static-embed-confirmed)),
and the owner made the two policy decisions (in-tree wheel now; keep per-version
3.10-3.13). The file-level implementation plan follows in
[Step 7](#step-7-implementation-plan-in-tree-per-version-wheel). The research
narrative below is preserved as-written for provenance; the decision record above
reflects the resolved state.

Three results overturn the direction the ticket assumed:

1. **The root cause is a static-embedded second copy of Origami, not RPATH or ABI
   packaging.** The wheel build links Origami *statically* into the extension by
   default, so the extension carries no dynamic dependency on the SDK's
   `liborigami.so.1`. Preloading the SDK library on top of that loads a *second*
   copy into the process -- the exact two-implementations-in-one-process condition
   this work exists to remove. Verified by source; must be confirmed by one
   `readelf -d` run before any code is written.

2. **The ticket's proposed prerequisite -- "add a nanobind provider to TheRock and
   turn Python on" -- targets a build that TheRock has deliberately dismantled.**
   After TheRock #6425, TheRock builds no Python bindings and the intended home is
   a separate `rocm-bindings` repository that does not exist yet (its governance is
   an open, unaccepted RFC). There is no repository to write the implementation
   against today.

3. **The incident's headline number is wrong, though its mechanism is real.** The
   struct that grew is `problem_t`, and it grew by 16 bytes (two `std::size_t`
   fields, `num_cus` and `q_heads`). A local recompile measured `sizeof(problem_t)`
   at 80 -> 96 bytes, not the ticket's 112 -> 128. The internal issue ROCM-29472
   remains inaccessible, so the 112/128 figure and the PyTorch-CI segfault
   causation stay unverified.

Method note: these findings come from a fan-out of six research agents (one per
Phase 2 step), an adversarial re-check of the three load-bearing claims (ABI,
loader, incident), and a completeness review. All three adversarial checks
returned `holds_up = false` against the first-pass recommendations; the
corrections are folded in below. Every load-bearing claim is cited to `file:line`
or a URL with a quote; unverifiable claims are marked.

## Decision record (Phase 2 result)

| Decision | Selected option | Status | Note |
| --- | --- | --- | --- |
| Binding repository | **In-tree wheel from `rocm-libraries/shared/origami`** (interim), independent of RFC #6050 | **RESOLVED** | Owner decision (2026-08-17). Off the documented `rocm-bindings` direction; interim until RFC #6050 lands. |
| Artifact boundary | Wheel built in-tree from `shared/origami`; TheRock keeps shipping native `liborigami` (`ENABLE_PYTHON=OFF`) | **RESOLVED** | Interim. The extension is *not* placed in any TheRock SDK component; it ships as a standalone wheel. |
| Python ABI | **Per-CPython-version build, 3.10-3.13**, version-specific packaging | **RESOLVED** | Owner decision (2026-08-17). Corroborated by the observed `cp310-cp310` wheel tag. `abi3` split-mode revisited only after nanobind 3.0 GA. |
| Linux loader | Force **shared** Origami link (`NEEDED liborigami.so.1`) + `preload_libraries` in `__init__.py` + register in `_dist_info.py`; keep `$ORIGIN` `INSTALL_RPATH` | **RESOLVED** | Static embed confirmed by `readelf` (no `liborigami` entry). Regression gate = `readelf -d` shows `NEEDED liborigami.so.1`. |
| Windows loader | `os.add_dll_directory` / `ctypes` preload by analogy | **DEFERRED** | Interim is Linux-first; no Windows build inspected. Out of scope for the interim wheel. |
| `rocm_sdk` dependency | Required at runtime; preload before extension import | **RESOLVED** | Follows from the loader decision. |
| Package name | `rocm-origami` (revert in-tree `name = "origami"`) | **RESOLVED** | `origami` is an active unrelated PyPI project. |
| PyPI migration | Fix -> publish `rocm-origami 0.0.3` -> verify in consumer CI -> yank `0.0.2`/`0.0.1`/`0.0.1.dev0` | **RESOLVED (sequence)** | Sequence set; the credential holder (PyPI owner) must still be identified internally to execute. |

## Findings by decision area

### F1 -- Binding-distribution architecture (Step 1): OPEN

After TheRock #6425, TheRock reverted its hipDNN Python binding build, wheel
packaging, and binding CI, stating the new architecture puts all bindings in a
separate repository. That target repository, `ROCm/rocm-bindings`, does not exist
(`gh repo view ROCm/rocm-bindings` returns "Could not resolve to a Repository"),
and the RFC that names it (#6050) is open and self-labels the name a placeholder:
"This RFC uses `ROCm/rocm-bindings` as a placeholder; the repository does not
exist yet." The name appears only in #6425 review comments, not the PR body.

Settled sub-conclusions (verified):

- TheRock's role is to ship the native `liborigami` only, with
  `ORIGAMI_ENABLE_PYTHON=OFF` (`shared/origami/CMakeLists.txt:24`). It does not
  build the binding, package a wheel, or run binding CI.
- A dedicated Python artifact treated as an ordinary native artifact inside
  TheRock is the *documented defect class* (RFC #6050 cites TheRock #5678: a
  `cpython-312` extension shipped in `rocm-sdk-devel` with a CI-local RPATH,
  breaking `rocm-sdk test` on Python 3.13).

Consequence: there is no first-party repository to write an Origami binding
implementation against. Origami's placement in `rocm-bindings` is inferred only
from #5650's one line "mirrors the hipDNN direction." **Owner decision required.**

Sources: TheRock PR #6425, RFC #6050 (issue #6048), `gh repo view`.

### F2 -- Python ABI strategy (Step 2): OPEN; first-pass recommendation refuted

The first pass recommended a hybrid `STABLE_ABI` build. The adversarial check
refuted it (`major`), on verified grounds:

- **nanobind "split mode" -- the only way to get one `abi3` wheel covering
  3.10-3.13 -- is unreleased.** It ships in nanobind 3.0; PyPI's latest stable is
  2.15.0 and `nanobind-backend` has only `1.0.0.dev1`. The nanobind changelog:
  "please try your projects with the development release of nanobind (3.0.0.dev1)
  and nanobind-backend (1.0.0.dev1)." Not adoptable for a package pinned
  `nanobind>=2.0.0`.
- **The actual fix to #3581 was a TheRock packaging change, not an ABI-keyword
  change.** PR #3820 made the Origami artifact version-specific and added RPATH +
  `_dist_info` registration; the extension still uses plain
  `nanobind_add_module` with no `STABLE_ABI` (`shared/origami/python/CMakeLists.txt:57`).
- **`STABLE_ABI` linked mode has a 3.12 floor.** Under TheRock's confirmed
  build-once-under-3.12-and-copy model, a single `cp312-abi3` extension still fails
  to import on 3.10/3.11 -- reproducing #3581.

Two coherent options remain, and the choice is a policy call about the minimum
supported Python:

- **(a)** Keep per-CPython-version builds + version-specific packaging (what
  shipped for #3581).
- **(b)** Raise the minimum Python to 3.12 and ship one linked `abi3` wheel.

Split mode becomes option (c) only after nanobind 3.0 reaches general
availability. **Owner/consumer decision required on the Python floor.**

Sources: nanobind changelog and PyPI JSON (queried 2026-08-17); TheRock PR #3820;
issue #3581 comments (ScottTodd, marbre).

### F3 -- Loader contract (Step 3): OPEN; refuted for omitting the build-config half

The runtime direction is correct and every primary source verifies: the preload
API (`rocm_sdk/__init__.py`), the "Unknown rocm library" raise, the hipDNN preload
pattern, and Origami's genuine absence from `_dist_info.py`. The `__init__.py`
still imports the extension directly with no preload
(`shared/origami/python/src/origami/__init__.py:12`).

The adversarial check refuted the contract (`major`) because it assumed the
extension carries `DT_NEEDED liborigami.so.1` -- and the default build produces the
opposite:

- The nested wheel build sets `ORIGAMI_STANDALONE=OFF`, which defaults
  `ORIGAMI_LIBRARY_TYPE` to `STATIC` (`shared/origami/CMakeLists.txt:30-34`), and
  the extension links it `PRIVATE` (`shared/origami/python/CMakeLists.txt:67`).
- `pyproject.toml` sets no `BUILD_SHARED_LIBS`, so the wheel path statically
  embeds Origami and emits **no** dynamic dependency on `liborigami.so.1`.

So the contract must add a mandatory build step -- either
`ORIGAMI_BUILD_FROM_SOURCE=OFF` + `find_package(origami)` against the SDK's shared
library, or force `ORIGAMI_BUILD_SHARED_LIBS=ON` -- and gate on the observable
`readelf -d origami*.so` showing `NEEDED liborigami.so.1`. Following the preload
steps *without* this fix loads a second copy `RTLD_GLOBAL` and can reproduce the
segfault.

Correction carried forward: keep the explicit `$ORIGIN`-relative `INSTALL_RPATH`
that PR #3820 merged and tested; do not switch to `INSTALL_RPATH_USE_LINK_PATH`,
which can bake absolute host paths.

Sources: `shared/origami/CMakeLists.txt:30-34`, `python/CMakeLists.txt:11,23-25,67`,
`src/origami/bindings.cpp` (by-value marshalling), TheRock PR #3820, `_dist_info.py`.

### F4 -- Artifact ownership (Step 4): native side settled; user-facing side gated

Verified correction to the ticket's revert story: the locally reproduced cause of
the reverts was an **unmatched `share/doc/origami/LICENSE.md`**, not a Python
artifact leak (memory `therock-origami-sdk-revert-rootcause.md`; TheRock does not
enforce file accounting -- `scanner.verify()` is commented out). Under TheRock's
default component globs, `.py` files and the `cpython` `.so` never match the
`dbg`/`dev` globs, so the #4901 body's dbg/dev excludes were redundant; the only
mandatory exclude is `lib/python*/**` on the `lib` component, to stop the default
`**/*.so` glob from double-claiming the ABI-specific extension.

If TheRock builds the extension at all, the mapping is:

| File | Component |
| --- | --- |
| `lib/liborigami.so` | `lib` |
| `include/origami/*.hpp` | `dev` |
| `lib/cmake/origami/*.cmake` | `dev` |
| `share/doc/origami/LICENSE.md` | `doc` (must exist) |
| `bin/origami/tests/*` | `test` |
| `lib/pythonX.Y/site-packages/origami/*` | `test` (and excluded from `lib`) |

But this table describes the *reverted* #3820 shape. Under the current
architecture (F1) TheRock builds no extension, so these rows are moot for the
interim, and a user-facing importable package belongs in a separate wheel, not any
SDK test component. **Gated on the binding-repo decision.**

Sources: TheRock PR #4901 (revert of #3820), `artifact_builder.py:47-104,270-278`,
`shared/origami/CMakeLists.txt`, `python/CMakeLists.txt`, memory note.

### F5 -- Incident (Step 5): mechanism verified, numbers weakened

Verified independently:

- **Static embed is real.** The 0.0.2 sdist bundles the full C++ tree and, under
  the nested build, selects `add_library(origami STATIC)` linked `PRIVATE` into the
  extension.
- **The struct grew by 16 bytes.** `problem_t` gained `num_cus` and `q_heads`
  (`types.hpp:667,670`), bound in `bindings.cpp:228-229`; the 0.0.2 sdist has
  neither field. A recompile of both headers with one host `g++` measured
  `problem_t` 80 -> 96 bytes (`config_t` unchanged at 224).

Weakened / unverified:

- The ticket's **112 -> 128** figure does not reproduce; treat as summary-only.
- **ROCM-29472** (internal Jira) is inaccessible; its Critical S1 classification
  and the PyTorch-CI heap-corruption/segfault causation stay unverified.

Correction carried forward: the fix is **not** already present in the worktree.
The Origami-target reuse block and the `STANDALONE -> STATIC` parent logic are
byte-identical to 0.0.2, so the standalone wheel path still static-embeds exactly
as 0.0.2 did. Dynamic linking to a single SDK library happens only when an *outer*
build supplies `roc::origami` or `ORIGAMI_BUILD_FROM_SOURCE=OFF`. A plausible
alternative mechanism worth noting: symbol interposition between the embedded
static copy and a co-loaded shared copy, rather than explicit by-value passing --
the worktree adds hidden visibility to the Origami target
(`shared/origami/CMakeLists.txt:43`) that 0.0.2 lacked, which would mitigate
interposition but not by-value skew.

Sources: 0.0.2 sdist vs worktree diff, local `sizeof` recompile, `types.hpp`,
`bindings.cpp`.

### F6 -- Deprecation and migration (Step 6): sequence sound, owner unnamed

- **Successor name = `rocm-origami`** (replace in place). The in-tree
  `name = "origami"` (`pyproject.toml:17`) is unpublishable: `origami` is an active,
  unrelated PyPI project. This regression must be reverted. (This is the one
  RESOLVED row.)
- **Sequence with no install gap and no lingering collision:** fix name + ABI +
  loader -> publish `rocm-origami 0.0.3` -> verify install/import in the PyTorch CI
  matrix -> then yank `0.0.2`, `0.0.1`, `0.0.1.dev0`.
- **Owner not publicly determinable.** PyPI metadata lists author "Advanced Micro
  Devices, Inc." with an empty maintainer field; the credential holder must be
  identified internally.

Sources: PyPI JSON for `rocm-origami` and `origami`; `pyproject.toml:17`.

## The one experiment that unblocks the plan

Build the Origami wheel via the real `pip` / `scikit-build-core` path (standalone,
no outer `roc::origami` target, default `ORIGAMI_BUILD_FROM_SOURCE=ON`) and run
`readelf -d` on the resulting `origami.cpython-*.so`.

- **Expected:** no `NEEDED liborigami.so.1` entry -- empirically confirming the
  default wheel statically embeds Origami.
- **Why it is load-bearing:** this single result settles the static-vs-shared
  premise that simultaneously gates F3 (loader), F5 (incident root cause), and F2
  (record the extension's Python tag from the same build). The same harness then
  serves as the regression gate: any real fix must flip the extension to show
  `NEEDED liborigami.so.1`.

This needs a ROCm/HIP build environment (a dev container), so it is the first
Phase 3 action, not something completed in this planning pass.

### Result (observed 2026-08-17): static embed confirmed

**Evidence status: verified by observation.**

The wheel was built via the real `pip` / `scikit-build-core` standalone path
inside `rocm/dev-ubuntu-22.04` (ROCm on the image; no outer `roc::origami`
target; default `ORIGAMI_BUILD_FROM_SOURCE=ON`). Reproduction script and log:
[`.handoff/origami-py-phase2/readelf_experiment.sh`](./.handoff/origami-py-phase2/readelf_experiment.sh),
`readelf_experiment.log`.

- Built wheel: `origami-0.1.0-cp310-cp310-linux_x86_64.whl` -- a
  **per-CPython-version** tag (`cp310-cp310`), not `abi3`. This directly
  corroborates F2: the default build is version-specific.
- Extension: `origami/origami.cpython-310-x86_64-linux-gnu.so`.
- `readelf -d` on the extension shows `NEEDED` for `libamdhip64.so.7`,
  `libstdc++.so.6`, `libm.so.6`, `libgcc_s.so.1`, `libc.so.6`,
  `ld-linux-x86-64.so.2` -- and **no `liborigami` entry of any kind**.

This is the observation-grade confirmation of the static-embed premise that
F3 (loader) and F5 (incident root cause) depend on: the standalone wheel
statically embeds Origami and carries no dynamic dependency on the SDK's
`liborigami.so.1`. It also fixes the regression gate: a correct fix must flip
this same `readelf -d` output to include `NEEDED liborigami.so.1` (or
`liborigami.so.X`). Note the extension *does* dynamically link
`libamdhip64.so.7`, so the HIP-host runtime is already consumed dynamically;
only Origami itself is embedded.

## Phase 2 verdict

Phase 2 is **complete**. The three conditions the earlier verdict set as
prerequisites are all met:

1. The static-embed experiment ran and its result is recorded (verified by
   observation: no `NEEDED liborigami` on the built extension).
2. The binding-home decision is made: an **in-tree per-version wheel** from
   `rocm-libraries/shared/origami` as the interim, explicitly independent of the
   still-open RFC #6050. This is an interim choice off the documented
   `rocm-bindings` direction; when RFC #6050 lands, the wheel source can be
   relocated without changing the loader/ABI contract.
3. The minimum supported Python is set: **3.10-3.13, built per version**.

F5 (incident) remains accurately *weakened* rather than fully verified -- the
struct-growth mechanism and static embed are proven, but the ticket's exact
112/128 sizes and the ROCM-29472 causal chain stay unverified because the
internal issue is inaccessible. That does not block the plan: the fix (dynamic
link to one `liborigami`) removes the two-implementations condition regardless of
the exact byte counts.

Residual, non-blocking follow-ups carried into execution: the Windows loader is
deferred (Linux-first interim), and the PyPI credential holder must be identified
internally to execute the yank sequence.

---

# Step 7 implementation plan (in-tree per-version wheel)

## Summary

Ship an official `rocm-origami` wheel built in-tree from
`rocm-libraries/shared/origami`, one wheel per CPython version for 3.10-3.13.
The single change that removes the incident's root cause is to stop statically
embedding Origami in the extension: build `liborigami` as a shared library,
give the extension a real `NEEDED liborigami.so.1`, and preload the ROCm SDK's
`liborigami` **before** importing the extension so the dynamic linker resolves
both the extension and any co-loaded consumer to one shared object.

This is an **interim** architecture. It is off the documented `rocm-bindings`
direction (RFC #6050, still open) and was chosen to unblock a fix now. When the
binding repository lands, the wheel *source* can move without changing the
loader or ABI contract defined here.

## Selected architecture and why

The extension today carries its own static copy of Origami. The standalone
wheel build leaves `ORIGAMI_BUILD_SHARED_LIBS` at its default -- which is
`ORIGAMI_STANDALONE` (`shared/origami/CMakeLists.txt:23`), and standalone is
false inside the nested wheel build -- so `ORIGAMI_LIBRARY_TYPE` becomes
`STATIC` (`shared/origami/CMakeLists.txt:30-34`) and the extension links it
`PRIVATE` (`shared/origami/python/CMakeLists.txt:67`). The built extension
therefore has no `liborigami` dynamic dependency, confirmed by `readelf`
(see the experiment result above).

The fix is to force a shared Origami and bundle it in the wheel:

1. **Force shared link.** Pass `-DORIGAMI_BUILD_SHARED_LIBS=ON` at wheel-build
   time. This flips `ORIGAMI_LIBRARY_TYPE` to `SHARED`
   (`shared/origami/CMakeLists.txt:30-31`); the extension then emits
   `NEEDED liborigami.so.1`.
2. **Preload the SDK copy first, then import.** `__init__.py` currently imports
   the extension directly with no preload
   (`shared/origami/python/src/origami/__init__.py:12`). Preloading the SDK's
   `liborigami.so.1` with `RTLD_GLOBAL` before the import makes the dynamic
   linker satisfy the extension's `NEEDED liborigami.so.1` from the
   already-loaded object -- SONAME-based deduplication. A co-loaded consumer
   (hipBLASLt, PyTorch) linking the same SONAME resolves to the same object, so
   only one Origami implementation exists in the process.
3. **Bundle a fallback `liborigami.so.1` in the wheel** next to the extension,
   so the package imports even when the SDK is absent. `$ORIGIN` on the
   extension's RPATH finds the sibling copy only when nothing has already
   loaded that SONAME.

Why this removes the incident: the crash needs *two* Origami copies with
different `problem_t` layouts in one process. A static embed cannot be
deduplicated by the linker; a shared object with a stable SONAME can. Preload
ordering is what guarantees the SDK copy wins.

### Rejected / alternative approaches

- **Build the extension against the SDK library, ship no `liborigami` in the
  wheel** (`ORIGAMI_BUILD_FROM_SOURCE=OFF` + `find_package(origami)`,
  `shared/origami/python/CMakeLists.txt:27-29`). This is the *stronger* form --
  the wheel then carries zero Origami implementation and cannot diverge from the
  SDK. It was **not** chosen for the interim because it hard-requires the SDK's
  `liborigami` to be present and discoverable at build and run time, which the
  standalone-wheel install story does not yet guarantee. **This becomes the
  preferred design once the wheel is produced inside a ROCm SDK build** (e.g. in
  the eventual `rocm-bindings` home); revisit it then.
- **One `abi3` wheel across 3.10-3.13.** Rejected: nanobind's abi3 "split mode"
  is unreleased (nanobind 3.0; stable is 2.15.0, and the package pins
  `nanobind>=2.0.0`, `shared/origami/python/pyproject.toml:27`), and linked
  `STABLE_ABI` has a 3.12 floor that would drop 3.10/3.11. Owner set the floor
  at 3.10, so per-version builds stand.

## Repositories and files to modify

All changes are in `rocm-libraries/shared/origami`; no TheRock change is
required for the interim wheel (TheRock keeps `ORIGAMI_ENABLE_PYTHON=OFF`,
`shared/origami/CMakeLists.txt:24`).

| File | Change |
| --- | --- |
| `python/pyproject.toml:17` | `name = "origami"` -> `name = "rocm-origami"`. `origami` is an active unrelated PyPI project. |
| `python/pyproject.toml:48-53` | Add `-DORIGAMI_BUILD_SHARED_LIBS=ON` to the wheel CMake args so the nested Origami builds `SHARED`. |
| `python/pyproject.toml` `[tool.scikit-build]` | Install/stage the built `liborigami.so.1` into the wheel's `origami/` package dir so it ships beside the extension. |
| `python/CMakeLists.txt:88-92` | Keep the merged `$ORIGIN`-relative `INSTALL_RPATH`; add plain `$ORIGIN` so the extension finds the sibling bundled `liborigami.so.1`. Do **not** switch to `INSTALL_RPATH_USE_LINK_PATH` (bakes absolute host paths). |
| `python/src/origami/__init__.py:10-12` | Before `from .origami import ...`, attempt `rocm_sdk.preload_libraries("origami")` (or the exact SDK API/name); on `ImportError`/absence, fall back to the bundled copy. Keep the existing actionable `ImportError` message. |
| `python/pyproject.toml:21` | Leave `requires-python = ">=3.9"` or raise to `>=3.10` to match the supported floor; building 3.10-3.13 per version regardless. |

Cross-repo dependency (not in this repo): `rocm_sdk` must register `origami` in
`_dist_info.py` for `preload_libraries("origami")` to resolve the SDK copy.
F3 found Origami absent there. Until that registration lands, preload falls back
to the bundled `liborigami.so.1`; single-copy consumption of the SDK library is
only guaranteed once registration exists. Track this as an execution dependency.

## ABI and build matrix

- Per CPython version: 3.10, 3.11, 3.12, 3.13. Four Linux wheels, tagged
  `cpXY-cpXY` (the experiment produced `cp310-cp310`, confirming the tag shape).
- Each wheel bundles a `liborigami.so.1` built from the same source revision as
  its extension, so bundled extension and bundled library never disagree on
  `problem_t` layout.
- Windows: deferred. Not built in the interim.

## Loader contract (Linux, observable)

- The extension MUST show `NEEDED liborigami.so.1` under `readelf -d`.
- Import order MUST be preload-then-import inside `__init__.py`.
- With the SDK present and registered, importing `origami` after or before
  hipBLASLt/PyTorch MUST resolve to a single `liborigami.so.1` mapping (checkable
  via `/proc/self/maps` or `ctypes` handle identity in a test).

## Artifact accounting (wheel contents)

The wheel ships exactly: the per-version extension
(`origami/origami.cpython-XY-*.so`), the bundled `origami/liborigami.so.1`, the
Python sources (`__init__.py`, `selector.py`), and metadata/license. Tests are
excluded from the wheel (already excluded, `pyproject.toml:66-72`). No file is
placed in any TheRock SDK component in the interim.

## Tests

- **Per-version import test:** in each of 3.10-3.13, `import origami` succeeds
  and a representative binding call runs (extend the existing
  `tests/test_origami.py`, wired at `python/CMakeLists.txt:109-113`).
- **Linker gate:** `readelf -d` on the built extension asserts
  `NEEDED liborigami.so.1` is present -- the inverse of the current experiment.
  Reuse `.handoff/origami-py-phase2/readelf_experiment.sh` as the harness.
- **Collision regression test:** in one process, load a second Origami copy
  (or the SDK's `liborigami` alongside the extension) and exercise a `problem_t`
  round-trip through `select_config`; assert no crash and consistent
  `sizeof(problem_t)`. The struct that grew is `problem_t` (+16 bytes:
  `num_cus`, `q_heads`), bound in `bindings.cpp` -- this is the type whose layout
  skew caused the incident. The pass condition is a single mapped
  `liborigami.so.1`.
- Note on evidence: passing these records the behavior; the ticket's exact
  112/128 byte figures remain unverified (ROCM-29472 inaccessible), so the
  regression test asserts single-copy + no-crash, not those specific sizes.

## Rollout, deprecation, rollback

1. Land the source changes above; CI builds and import-tests all four wheels and
   runs the linker gate + collision regression.
2. Publish `rocm-origami 0.0.3`.
3. Verify install + import across the consumer (PyTorch) CI Python matrix.
4. Only then yank `rocm-origami 0.0.2`, `0.0.1`, `0.0.1.dev0` to close the
   window where the static-embed sdist can still be installed.
5. **Rollback:** if 0.0.3 regresses, do not yank the predecessors; the fix is
   self-contained to the wheel, so reverting the four wheels restores 0.0.2
   behavior. The PyPI credential holder must be identified internally to execute
   publish/yank.

## Measurable completion criteria

- `readelf -d` on every shipped extension shows `NEEDED liborigami.so.1`.
- `import origami` succeeds on 3.10, 3.11, 3.12, 3.13.
- The collision regression test passes with exactly one `liborigami.so.1`
  mapping in the process.
- `rocm-origami 0.0.3` is installable and importable in the consumer CI matrix;
  predecessors are yanked.
- No file from the wheel lands in a TheRock SDK component.
