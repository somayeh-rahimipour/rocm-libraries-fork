# Origami Python Bindings: Planning Research

## Status

- Planning phase: Phase 1 complete; Phase 2 **complete**. The active
  implementation plan is [Step 8](#step-8-implementation-plan-build-against-installed-liborigami-rocshmem-pattern):
  build the wheel against an installed `liborigami` via `find_package(origami)`,
  following the sanctioned rocSHMEM pattern (rocm-systems PR #9471, merged;
  TheRock PR #5638, open). See the
  [Session 3-4 correction](#session-3-4-correction-the-rocshmem-pattern-supersedes-the-interim)
  for why this replaces the earlier interim plan.
- **Superseded:** [Step 7](#step-7-implementation-plan-superseded)
  (in-tree per-version wheel with forced-shared link + bundled `liborigami` +
  preload deduplication) is retained for provenance only. Its premise -- that no
  first-party mechanism exists to build against the installed library -- was
  false; the rocSHMEM precedent is that mechanism.
- The gating experiment ran and its result stands: the default standalone wheel
  statically embeds Origami (no `NEEDED liborigami.so.1`). See
  [Phase 2 findings](#phase-2-findings) and [Phase 2 verdict](#phase-2-verdict).
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

> **Partially superseded (2026-08-18).** The delivery/loader rows below reflect
> the Session-2 interim decision. They are superseded by the
> [Step 8 decision record](#decision-record-step-8), which adopts the rocSHMEM
> build-against-installed pattern. The ABI (per-version 3.10-3.13), package-name
> (`rocm-origami`), and PyPI-migration rows still hold; the binding-repository,
> artifact-boundary, and loader rows are replaced. See the
> [Session 3-4 correction](#session-3-4-correction-the-rocshmem-pattern-supersedes-the-interim).

| Decision | Selected option | Status | Note |
| --- | --- | --- | --- |
| Binding repository | **In-tree wheel from `rocm-libraries/shared/origami`** (interim), independent of RFC #6050 | **SUPERSEDED** | Session-2 decision (2026-08-17); replaced by build-against-installed (Step 8, 2026-08-18). |
| Artifact boundary | Wheel built in-tree from `shared/origami`; TheRock keeps shipping native `liborigami` (`ENABLE_PYTHON=OFF`) | **RESOLVED** | Interim. The extension is *not* placed in any TheRock SDK component; it ships as a standalone wheel. |
| Python ABI | **Per-CPython-version build, 3.10-3.13**, version-specific packaging | **RESOLVED** | Owner decision (2026-08-17). Corroborated by the observed `cp310-cp310` wheel tag. `abi3` split-mode revisited only after nanobind 3.0 GA. |
| Linux loader | Force **shared** Origami link (`NEEDED liborigami.so.1`) + `preload_libraries` in `__init__.py` + register in `_dist_info.py`; keep `$ORIGIN` `INSTALL_RPATH` | **SUPERSEDED** | Regression gate (`readelf -d` shows `NEEDED liborigami.so.1`) still holds; the *mechanism* is now `find_package(origami)` link, not forced `BUILD_SHARED_LIBS` + bundle. See [Step 8 loader row](#decision-record-step-8). |
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

# Session 3-4 correction: the rocSHMEM pattern supersedes the interim

## What changed

The interim plan below ([Step 7](#step-7-implementation-plan-superseded)) rested
on one premise from finding [F1](#f1----binding-distribution-architecture-step-1-open):
there is no first-party repository or build mechanism that compiles an Origami
binding against the *installed* `liborigami`, so the wheel must instead build
Origami from source and carry its own copy. That premise is false.

TheRock already runs a sanctioned build-against-installed wheel pipeline, and
rocSHMEM is the worked precedent:

- **rocm-systems PR #9471 (merged).** Moved `rocshmem4py` out of the nested
  `projects/rocshmem/python` C++ subproject into a standalone PEP-517 project at
  `python/rocshmem`. It builds the nanobind extension against an installed
  rocSHMEM discovered through `find_package(rocshmem CONFIG)` /
  `CMAKE_PREFIX_PATH` -- no static embed, no bundled library. The wheel version
  is dynamic: a base release plus a PEP 440 local segment recording the linked
  library version (`0.1.0+rocshmem3.6.0`), parsed from the installed
  `rocshmem_config.h`, and `find_package(rocshmem 3.5 ...)` gates the minimum
  compatible library version.
- **TheRock PR #5638 (open).** Wires that project into TheRock's existing
  `build_portable_linux_python_packages.yml`: after the rocSHMEM artifacts are
  present, it flattens them into a prefix, `cd rocm-systems/python/rocshmem`, and
  runs `ROCM_PATH=<prefix> CMAKE_PREFIX_PATH=<prefix> python -m build --wheel
  --outdir <dist>`. The workflow's existing upload/index step then publishes the
  wheel.

## Why this changes Origami's plan

- **F1's "no mechanism" is answered.** The pipeline is the mechanism; rocSHMEM is
  the pattern. The choice of where the binding source lives (`rocm-bindings` vs
  today) is now independent of *how* the wheel is built.
- **Origami fits the pattern more cleanly than rocSHMEM does.** rocSHMEM ships a
  *static* library, so its binding does the final `-fgpu-rdc` device link and
  must emit `--offload-arch` for every GPU. Origami ships a *shared*
  `liborigami.so` and is host-only (the extension links `hip::host`, not device
  code), so it needs none of that machinery -- `find_package(origami)` gives the
  extension a real `NEEDED liborigami.so.1` and one implementation in the process
  by construction.
- **Origami is already substantially wired for it.** `shared/origami/python/CMakeLists.txt`
  already has the build-against-installed branch (`find_package(origami REQUIRED)`
  -> `roc::origami`), gated behind `ORIGAMI_BUILD_FROM_SOURCE`, which currently
  defaults `ON` (the static-embed path the experiment confirmed). And
  `shared/origami/CMakeLists.txt` already exports a consumable package
  (`rocm_install(TARGETS origami)`, `install(... EXPORT origami-targets)`,
  `rocm_export_targets(...)`). Adopting the pattern is largely a default flip plus
  packaging metadata, not new build infrastructure.
- **The interim's force-shared + bundle + preload-dedup design is a weaker
  reimplementation** of what the pattern gives directly. It was chosen to avoid
  requiring the SDK's `liborigami` at build time; the pipeline supplies exactly
  that (`CMAKE_PREFIX_PATH=<flattened artifacts>`), so the constraint no longer
  applies.

## Owner decision (2026-08-18)

David chose to **rewrite the plan around the rocSHMEM pattern**. This reverses the
Session-2 interim choice. The Python floor decision (per-version 3.10-3.13, no
`abi3`) is unchanged and orthogonal -- the rocSHMEM pattern also builds per
CPython version with plain `nanobind_add_module`.

The implementation plan follows in
[Step 8](#step-8-implementation-plan-build-against-installed-liborigami-rocshmem-pattern).

Sources (pulled live 2026-08-18): rocm-systems PR #9471 body +
`python/rocshmem/{pyproject.toml,setup.py,CMakeLists.txt}` on `develop`; TheRock
PR #5638 diff; `shared/origami/python/CMakeLists.txt`, `shared/origami/CMakeLists.txt`.

---

# Step 8 implementation plan (build against installed liborigami -- rocSHMEM pattern)

## Summary

Ship an official `rocm-origami` wheel built against the **installed**
`liborigami`, following the rocSHMEM precedent. The extension links the SDK's
shared library through `find_package(origami)`, so it carries a real
`NEEDED liborigami.so.1` and never embeds or bundles a second copy. The wheel is
built one per CPython version (3.10-3.13) by TheRock's existing
`build_portable_linux_python_packages.yml`, the same lane that builds
`rocshmem4py`.

The single change that removes the incident's root cause is to stop building
Origami from source inside the wheel: flip `ORIGAMI_BUILD_FROM_SOURCE` to `OFF`
so the extension resolves `roc::origami` from the installed package instead of
static-embedding it.

## Decision record (Step 8)

| Decision | Selected option | Status | Note |
| --- | --- | --- | --- |
| Binding delivery | **Standalone PEP-517 wheel built against installed `liborigami`** via `find_package(origami)`, following rocSHMEM (rocm-systems #9471) | **RESOLVED** | Owner decision (2026-08-18). Supersedes the interim in-tree/force-shared/bundle path. |
| Build mechanism | TheRock `build_portable_linux_python_packages.yml` builds the wheel from `rocm-libraries/shared/origami/python` with `CMAKE_PREFIX_PATH` = flattened Origami artifacts, mirroring the #5638 rocSHMEM step | **RESOLVED (design)** | Requires a new workflow step in TheRock (analogous to #5638). No `--offload-arch` step -- Origami is host-only. |
| Wheel contents | Extension + Python sources + metadata only; **no bundled `liborigami`** | **RESOLVED** | The extension links the SDK copy dynamically; nothing to bundle. |
| Linux loader | Extension shows `NEEDED liborigami.so.1`; SDK copy resolved via `rocm_sdk` preload + `_dist_info.py` registration; keep `$ORIGIN` `INSTALL_RPATH` | **RESOLVED** | Regression gate unchanged: `readelf -d` must show `NEEDED liborigami.so.1`. |
| Python ABI | **Per-CPython-version build, 3.10-3.13**, plain `nanobind_add_module` | **RESOLVED** | Unchanged from Session 2; orthogonal to delivery. `abi3` revisited only after nanobind 3.0 GA. |
| Version scheme | Wheel `0.1.0`, single-sourced from `__init__.py`; kept decoupled from the C++ `liborigami` SOVERSION (`1`, from `project(Origami VERSION 1.0.0)`) | **RESOLVED: `0.1.0` (owner-ratified)** | Minor bump over the broken third-party `0.0.2` -- an honest signal that this is a different, first-party, dynamically-linked package, not a patch. **Caveat:** a rocSHMEM-style PEP 440 local segment (`0.1.0+origami...`) is *rejected by public PyPI*; use plain `0.1.0` for the public-PyPI release, reserving any local segment for a TheRock-index build only. |
| Package name | `rocm-origami` (revert in-tree `name = "origami"`) | **RESOLVED** | `origami` is an active unrelated PyPI project. |
| `_dist_info.py` entry | Add an Origami `LibraryEntry` in TheRock so `rocm_sdk` preload resolves the SDK copy | **RESOLVED (design)** | Still required even under build-against-installed: preload needs the registration to find the SDK library at import time. |
| Windows loader | `os.add_dll_directory` / `ctypes` preload by analogy | **DEFERRED** | Linux-first; no Windows build inspected. |
| PyPI migration | Publish `rocm-origami 0.1.0` -> verify in consumer CI -> yank `0.0.2`/`0.0.1`/`0.0.1.dev0` | **RESOLVED (sequence)** | Credential holder still to be identified internally. |

## Selected architecture and why

Today the wheel builds Origami from source and static-embeds it: the standalone
build leaves `ORIGAMI_BUILD_FROM_SOURCE` at its `ON` default
(`shared/origami/python/CMakeLists.txt:11`), so the wheel calls
`add_subdirectory(..)` on the C++ tree with `ORIGAMI_ENABLE_PYTHON OFF` and links
the resulting target `PRIVATE`. The parent build then resolves
`ORIGAMI_LIBRARY_TYPE` to `STATIC` (standalone is false in that nested build,
`shared/origami/CMakeLists.txt:30-34`), so the extension carries no `liborigami`
dynamic dependency -- confirmed by `readelf` (see the
[experiment result](#result-observed-2026-08-17-static-embed-confirmed)).

The fix flips the extension to consume the installed library:

1. **Build against the installed package.** Set `ORIGAMI_BUILD_FROM_SOURCE=OFF`.
   The existing `else()` branch then runs `find_package(origami REQUIRED)` and
   links `roc::origami` (`shared/origami/python/CMakeLists.txt:27-29,67`). The
   extension emits `NEEDED liborigami.so.1` from the exported shared target.
2. **Supply the installed library at build time via the pipeline.** The wheel is
   built with `CMAKE_PREFIX_PATH` pointing at a prefix that contains
   `liborigami.so`, its headers, and `lib/cmake/origami/*.cmake` -- exactly what
   `rocm_export_targets` installs. TheRock's workflow flattens the Origami
   artifacts into that prefix before `python -m build`, the same way #5638 does
   for rocSHMEM.
3. **Resolve the SDK copy at run time.** Preload the SDK's `liborigami.so.1`
   before importing the extension (`rocm_sdk.preload_libraries("origami")` or the
   exact SDK API), backed by an Origami `LibraryEntry` in TheRock's
   `_dist_info.py`. Keep the `$ORIGIN`-relative `INSTALL_RPATH` as the fallback
   resolution path. Because the extension links a stable SONAME, a co-loaded
   consumer (hipBLASLt, PyTorch) resolves to the same object.

Why this removes the incident: the crash needs two Origami copies with different
`problem_t` layouts in one process. Building against the installed shared library
means the wheel contains zero Origami implementation of its own -- there is only
the SDK copy to resolve, so divergence is impossible by construction, not merely
deduplicated after the fact.

### Rejected / alternative approaches

- **Interim in-tree wheel: build from source, force `ORIGAMI_BUILD_SHARED_LIBS=ON`,
  bundle `liborigami.so.1`, dedup by preload** (the [Step 7](#step-7-implementation-plan-superseded)
  plan). Rejected: it reimplements, more weakly, what `find_package(origami)`
  gives directly, and it keeps an Origami copy in the wheel that can drift from
  the SDK. Its sole advantage -- not needing the SDK library at build time -- is
  moot once the pipeline supplies it.
- **One `abi3` wheel across 3.10-3.13.** Rejected as in Session 2: nanobind's
  abi3 "split mode" is unreleased (nanobind 3.0; stable is 2.15.0), and linked
  `STABLE_ABI` has a 3.12 floor that would drop 3.10/3.11. Owner set the floor at
  3.10.
- **Keep `scikit-build-core`, or switch to setuptools + `CMakeExtension` to match
  rocSHMEM exactly.** Keep `scikit-build-core`. rocSHMEM uses setuptools, but the
  pattern's essence is standalone-build + `find_package(installed)` + no bundle,
  not the specific backend. `scikit-build-core` already drives Origami's build,
  is PEP-517, and works with `python -m build`, so switching backends would be
  churn without a functional gain.

## Repositories and files to modify

Two repositories. The binding source and packaging live in
`rocm-libraries/shared/origami`; the build lane and SDK registration live in
TheRock.

### `rocm-libraries/shared/origami`

| File | Change |
| --- | --- |
| `python/CMakeLists.txt:11` | Change the `ORIGAMI_BUILD_FROM_SOURCE` default to `OFF` (or set it `OFF` from `pyproject.toml` CMake args) so the wheel build resolves `find_package(origami)` instead of static-embedding. |
| `python/pyproject.toml` `[project] name` | `name = "origami"` -> `name = "rocm-origami"`. `origami` is an active unrelated PyPI project. |
| `python/pyproject.toml` `[tool.scikit-build] cmake.define` | Add `ORIGAMI_BUILD_FROM_SOURCE = "OFF"` so the wheel path builds against the installed package regardless of the CMake default. |
| `python/src/origami/__init__.py` | Before `from .origami import ...`, preload the SDK `liborigami` (`rocm_sdk.preload_libraries("origami")` or the exact API). Keep the existing actionable `ImportError` message. Remove any bundled-fallback logic -- there is no bundled copy. |
| `python/pyproject.toml` version | Reconcile the reported version to a single number (see [version note](#version-reconciliation)); optionally record the linked `liborigami` version as a PEP 440 local segment, rocSHMEM-style. |
| `python/CMakeLists.txt` (RPATH) | Keep the merged `$ORIGIN`-relative `INSTALL_RPATH`; do **not** switch to `INSTALL_RPATH_USE_LINK_PATH` (bakes absolute host paths). No bundled `liborigami`, so `$ORIGIN` is only a fallback for the preload path. |
| `python/CMakeLists.txt` (min version) | Optionally add `find_package(origami <minver> CONFIG REQUIRED)` to gate the earliest `liborigami` whose `problem_t` the binding matches (rocSHMEM gates `>= 3.5.0`). Requires Origami's exported config-version file to carry a usable version. |

### TheRock

| File | Change |
| --- | --- |
| `.github/workflows/build_portable_linux_python_packages.yml` | Add a "Build rocm-origami wheel" step mirroring #5638's rocSHMEM step: guard on the presence of Origami dev artifacts; `git submodule update --init rocm-libraries`; `fileset_tool.py artifact-flatten` the Origami artifacts into a prefix; `cd rocm-libraries/shared/origami/python`; `ROCM_PATH=<prefix> CMAKE_PREFIX_PATH=<prefix> python -m build --wheel --outdir <dist>`. Omit the `librocshmem_device_*.bc` guard and `--offload-arch` handling -- Origami is host-only. |
| `_dist_info.py` | Add `LibraryEntry("origami", "libraries", "liborigami.so*", "origami*.dll")` (draft already exists in the standalone-tensilelite worktree at `_dist_info.py:287`) so `rocm_sdk` preload resolves the SDK copy at import. |

TheRock continues to build only the native `liborigami`
(`ORIGAMI_ENABLE_PYTHON=OFF`, `shared/origami/CMakeLists.txt:24`); the extension
is built by the new wheel step, not by the component build.

### Version reconciliation

**Ratified: `0.1.0`** (owner call, 2026-08-18). The built wheel already reports
`0.1.0` dynamically from `src/origami/__init__.py`; the earlier migration draft
had assumed `0.0.3`, which is dropped. `0.1.0` is a clean minor over the broken
`0.0.x` third-party workaround line -- an honest signal that this is a different,
first-party, dynamically-linked package rather than a patch to it -- and it
satisfies the only hard constraint, that the successor sort above `0.0.2`.

The wheel version stays **decoupled from the C++ `liborigami` SOVERSION** (`1`,
from `project(Origami VERSION 1.0.0)` / `rocm_set_soversion`): the binding's API
maturity is not the library's, and the loader gate depends on that `1`, not on
the wheel number.

**Local-segment caveat.** A rocSHMEM-style PEP 440 local segment
(`0.1.0+origami<ver>`) is **rejected by public PyPI uploads** -- rocSHMEM gets
away with it only because it publishes to TheRock's own index, not public PyPI.
Since closing ROCM-29472 requires superseding the third-party package *on public
PyPI*, the public release must be plain `0.1.0`; a local segment can be reserved
for a TheRock-index build if provenance tagging is wanted there.

## ABI and build matrix

- Per CPython version: 3.10, 3.11, 3.12, 3.13. Four Linux wheels, tagged
  `cpXY-cpXY` (the experiment produced `cp310-cp310`, confirming the tag shape).
- No wheel bundles `liborigami`; each links the SDK copy dynamically.
- Windows: deferred. Not built in this plan.

## Loader contract (Linux, observable)

- The extension MUST show `NEEDED liborigami.so.1` under `readelf -d`. This is now
  satisfied by construction (`find_package(origami)` link), not by a forced
  `BUILD_SHARED_LIBS`.
- `__init__.py` MUST preload the SDK `liborigami` before importing the extension.
- With the SDK present and registered in `_dist_info.py`, importing `origami`
  before or after hipBLASLt/PyTorch MUST resolve to a single `liborigami.so.1`
  mapping (checkable via `/proc/self/maps` or `ctypes` handle identity in a test).

## Result observed (2026-08-18): build-against-installed validated

Ran the Step 8 path end-to-end in `rocm/dev-ubuntu-22.04` (harness:
`.handoff/origami-py-phase2/build_against_installed_experiment.sh`, log
alongside): install `liborigami` shared with its CMake exports into a prefix,
then build the wheel with `ORIGAMI_BUILD_FROM_SOURCE=OFF` and
`CMAKE_PREFIX_PATH` pointing at that prefix.

- **Wheel:** `rocm_origami-0.1.0-cp310-cp310-linux_x86_64.whl` (name and version
  as decided; the extension links `/opt/origami-install/lib/liborigami.so.1.0`).
- **Linker gate PASS:** `readelf -d` on the extension shows
  `NEEDED liborigami.so.1` -- the static embed is gone. Installed lib SONAME is
  `liborigami.so.1`, confirming the loader gate keys on the SOVERSION `1`, not the
  wheel version.
- **`liborigami` is off the default loader path (confirmed):** with no
  `rocm_sdk` and nothing on `LD_LIBRARY_PATH`, `import origami` fails with
  `liborigami.so.1: cannot open shared object file`. The extension's `$ORIGIN`
  RPATH resolves to the venv, not the SDK library wheel, so the preload is
  required -- it is not optional on Linux. (This corrects the assumption, copied
  from a hipDNN wheel-package draft, that the binding library "resolves on its
  own" on Linux; `__init__.py` now names `origami` in the preload list on both
  platforms.)
- **Extension is sound once deps resolve:** with the `liborigami` and HIP runtime
  directories on `LD_LIBRARY_PATH` (standing in for the `rocm_sdk` RTLD_GLOBAL
  preload), `import origami` succeeds and reports version `0.1.0`.

Not exercised here: the literal `rocm_sdk.initialize_process(preload_shortnames=
[..., "origami"])` call in a real ROCm-wheel environment (no `rocm_sdk` in this
base image) and the single-copy `/proc/self/maps` assertion. The two import
results bracket it -- deps must be made resolvable (4a) and the extension imports
cleanly once they are (4b) -- and the preload API is the production mechanism that
makes them resolvable; end-to-end confirmation belongs to the TheRock wheel-lane
wiring.

## Artifact accounting (wheel contents)

The wheel ships exactly: the per-version extension
(`origami/origami.cpython-XY-*.so`), the Python sources (`__init__.py`,
`selector.py`), and metadata/license. No `liborigami` and no tests
(tests already excluded, `pyproject.toml`). No file is placed in any TheRock SDK
component; the wheel is published from the Python-packages lane's `dist/`.

## Tests

- **Linker gate:** `readelf -d` on the built extension asserts
  `NEEDED liborigami.so.1` is present. Reuse
  `.handoff/origami-py-phase2/readelf_experiment.sh` as the harness, inverting its
  pass condition (the experiment confirmed the entry is *absent* today).
- **Per-version import test:** in each of 3.10-3.13, `import origami` succeeds
  against an installed SDK and a representative binding call runs (extend the
  existing `tests/test_origami.py`, wired at `python/CMakeLists.txt`).
- **Single-copy regression test:** in one process, load the SDK `liborigami`
  alongside the extension and exercise a `problem_t` round-trip through
  `select_config`; assert exactly one `liborigami.so.1` mapping and no crash.
  `problem_t` is the type that grew (+16 bytes: `num_cus`, `q_heads`, bound in
  `bindings.cpp`), so it is the layout-skew canary. Note: the ticket's exact
  112/128 byte figures remain unverified (ROCM-29472 inaccessible), so the test
  asserts single-copy + no-crash, not those sizes.

## Rollout, deprecation, rollback

1. Land the `rocm-libraries` source changes; land the TheRock workflow step and
   `_dist_info.py` entry.
2. The Python-packages lane builds and import-tests all four wheels and runs the
   linker gate + single-copy regression.
3. Publish `rocm-origami` (reconciled version).
4. Verify install + import across the consumer (PyTorch) CI Python matrix.
5. Only then yank `rocm-origami 0.0.2`, `0.0.1`, `0.0.1.dev0` to close the window
   where the static-embed sdist can still be installed.
6. **Rollback:** if the release regresses, do not yank predecessors; reverting the
   four wheels restores prior behavior. The PyPI credential holder must be
   identified internally to execute publish/yank.

## Measurable completion criteria

- `readelf -d` on every shipped extension shows `NEEDED liborigami.so.1`.
- No shipped wheel contains a `liborigami.so*` file.
- `import origami` succeeds on 3.10-3.13 against an installed SDK.
- The single-copy regression test passes with exactly one `liborigami.so.1`
  mapping in the process.
- The reconciled `rocm-origami` release is installable and importable in the
  consumer CI matrix; predecessors are yanked.

---

# Step 7 implementation plan (SUPERSEDED)

> **Superseded (2026-08-18) by
> [Step 8](#step-8-implementation-plan-build-against-installed-liborigami-rocshmem-pattern).**
> Retained for provenance. This interim plan built Origami from source inside the
> wheel, forced a shared link, bundled `liborigami.so.1`, and relied on preload
> ordering to deduplicate copies. Its premise -- no first-party mechanism to build
> against the installed library -- was false (see the
> [Session 3-4 correction](#session-3-4-correction-the-rocshmem-pattern-supersedes-the-interim)).
> Do not implement this; it is preserved only to record the reasoning that led to
> Step 8.

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
2. Publish `rocm-origami 0.1.0` (plain public version -- no local segment).
3. Verify install + import across the consumer (PyTorch) CI Python matrix.
4. Only then yank `rocm-origami 0.0.2`, `0.0.1`, `0.0.1.dev0` to close the
   window where the static-embed sdist can still be installed.
5. **Rollback:** if 0.1.0 regresses, do not yank the predecessors; the fix is
   self-contained to the wheel, so reverting the four wheels restores 0.0.2
   behavior. The PyPI credential holder must be identified internally to execute
   publish/yank.

## Measurable completion criteria

- `readelf -d` on every shipped extension shows `NEEDED liborigami.so.1`.
- `import origami` succeeds on 3.10, 3.11, 3.12, 3.13.
- The collision regression test passes with exactly one `liborigami.so.1`
  mapping in the process.
- `rocm-origami 0.1.0` is installable and importable in the consumer CI matrix;
  predecessors are yanked.
- No file from the wheel lands in a TheRock SDK component.
