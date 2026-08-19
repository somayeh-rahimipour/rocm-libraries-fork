# Rebuilt stacked-PR plan and placement rationale through PR 17

This records the plan used to rebuild the original PR-2 sequence beginning at
`56c7cf4b9c` and continuing through `26254111b8` as one linear stacked branch:

```text
users/alvasile/tensilelite_packaging-stack
```

The original branch was preserved unchanged. Its initial source-namespace move
was replayed as the first numbered PR-2 commit, so the reconstructed PR-2
sequence is complete. This file describes why a change belongs in a particular
PR, not merely which historical commit it first came from. The aggregate final
tree and each PR-tip contract were treated as the source of truth; old commit
boundaries were used only as evidence.

## Scope of these review artifacts

The reconstructed-stack endpoint for this plan, its companion justification,
and the commit-split patch audit is PR 17:

```text
27e1d83c21b1c8f5805e6018865ada8190e6615d
```

PRs 18--20 are intentionally excluded. They are conditional TheRock re-enable
work above the reconstruction endpoint, to be landed only after the requisite
TheRock support. They are not part of the original-branch parity comparison or
the PR-17 split plan recorded here.

PR 21 is a docs-only review slice placed after PR 20 and directly below this
audit commit. It carries the ten standalone decision, implementation-plan, and
research records that were previously co-located with PR-15 workflow guidance.
The PR-17 comparison remains a code and consumer-closure audit; PR 21 restores
the deferred records at the top of the stack.

## Decision rules

1. Folder moves precede all import/reference moves.

   Python package/test directory moves are reviewable as mechanical changes.
   The immediate PR closure includes every import, test, CI selector, and
   downstream consumer repair needed to make the PR tip work.

2. A PR tip must work; an intermediate commit may be deliberately incomplete.

   For example, a pure move commit can precede import rewrites, but the final
   commit in that numbered PR restores tests, CI, and all known consumers.

3. Consumer changes travel with the breaking producer change.

   GEKO, CMake codegen, tox, installed-artifact runners, and the rocjitsu race
   runner were treated as real consumers. A producer-side removal could not be
   committed at a PR tip until its affected consumer had moved to the new
   interface.

4. Compatibility-sensitive public boundaries are split from internal renames.

   The `Tensile` package-directory/root migration, lower-case internal-module
   migration, canonical CLI, compatibility wheel, runtime binding, and GEKO
   cutover are separate review decisions.

5. Temporary TheRock gates are part of the reconstructed implementation.

   The original branch deliberately disables strict runtime-version validation
   and Python-SDK runtime selection. Those gates remain false until upstream
   TheRock forwards the required version/payload contract.

6. Tests stay with the behavior they protect.

   Unit, characterization, installed-artifact, and GEKO tests were placed with
   their owning code or compatibility boundary. Tests were not deleted merely to
   make a branch green.

## PR stack

### PR 2 — package folder and root namespace migration

Purpose: move the package/test/custom-kernel folder structure from the old root
namespace to `tensilelite/`.

Why this is first:

- Every later Python import and path assumes the moved physical tree.
- It is mostly mechanical review noise, so separating it keeps subsequent
  functional reviews readable.
- The PR closure repairs all moved imports, GEKO normalization imports, tox,
  pre-commit selectors, test paths, and the legacy package consumer paths.

Important boundary: external C++/wire contracts such as `include/Tensile`,
`src/Tensile`, `TensileLibrary`, and backend value `tensile` remain untouched.

### PR 3 — lower-case internal module migration

Purpose: lower-case private generator, logic, library-creation, and ext-op
module names after the physical package move.

Why it stacks on PR 2:

- The module paths only exist after the folder migration.
- All import call sites, retained `bin/Tensile*` wrappers, coverage paths, and
  GEKO normalization need the new module names together.

Important closure: direct lower-case generator calls in gfx1250 codegen tests
are included here rather than deferred to packaging work.

### PR 4 — wheel/handler/rocisa foundation

Purpose: establish the canonical wheel boundary and package-local generation
handlers while keeping rocisa independently supplied.

Why separate:

- It changes packaging/build ownership but not the public command surface.
- Source client autobuild removal is isolated from later runtime binding and
  client selection changes.

### PR 5 — additive canonical package command

Purpose: add `tensilelite` and `python -m tensilelite` commands and route CMake
codegen through the package interface.

Why separate:

- This is the requested Python-entrypoint seam.
- Legacy console scripts and source wrappers remain available, so downstream
  rebases encounter one command-surface change at a time.

### PR 6 — package/client release identity prerequisite

Purpose: define a shared component/client version and test
`tensilelite-client --version` against package identity.

Why before binding:

- A persistent client binding is unsafe unless it can validate that the client
  and package are the same release.

### PR 7 — runtime/binding/source-install foundation

Purpose: add the runtime helpers, per-installation client-binding store,
`tensilelite-configure-client`, and `invoke install` / `invoke configure-client`
source workflow.

Why additive:

- It establishes a binding mechanism without forcing old source consumers to
  use it yet.
- Legacy wrappers, `--prebuilt-client`, and GEKO source execution remain live.

### PR 8 — release wheels and installed-artifact foundation

Purpose: classify source-only tests, remove the undeclared pandas generator
dependency, package compatibility command aliases, build/validate canonical and
compatibility wheels, and stage installed artifacts.

Why this grouping:

- These changes define the wheel and installed-test boundary as one contract.
- Moving legacy console aliases into the compatibility wheel is separate from
  deleting source wrappers; current GEKO still uses source wrappers until PR 13.

### PR 9 — explicit ROCm build identity

Purpose: make the selected ROCm version an explicit package/wheel input across
CMake, source installs, compatibility wheels, and tox bootstrap.

Why after release wheels:

- The release-wheel target is the first consumer that needs publication
  identity rather than only a conventional prefix.
- The temporary TheRock `0.0.0` fallback belongs here because current TheRock
  does not forward a real identity.

### PR 10 — canonical-wheel client-free device generation

Purpose: make device-library generation use the canonical wheel plus raw
in-tree rocisa, not `tensilelite-client`.

Why separate from runtime work:

- CMake codegen must not be coupled to benchmark/client execution.
- Device generation can become client-free while GEKO and source development
  retain an explicit bound client.

### PR 11 — dormant ROCm installation/lazy-client model

Purpose: introduce the complete System ROCm/Python SDK model and lazy client
selection helpers under currently disabled strict/Python-SDK gates.

Why dormant first:

- It allows direct testing of the model without changing package initialization
  or breaking GEKO source wrappers.

### PR 12 — selected-installation toolchain/enumerator plumbing

Purpose: make validators, architecture detection, generators, and ext-ops use
one selected ROCm installation and ordered device-enumerator candidates.

Why before package cutover:

- Tool lookup needs to be coherent before the installed runtime becomes the
  production package surface.
- The validator seam initializes only the installation, not the client, so
  source/prebuilt GEKO behavior remains intact.

### PR 13 — atomic GEKO and installed package cutover

Purpose: move GEKO to installed TensileLite interfaces, then retire producer
surfaces that GEKO previously consumed.

Internal commit order:

1. Build, editable-install, and bind the GEKO client in the active interpreter.
2. Migrate GEKO create/merge/run/optimizer/generated-script consumers to direct
   installed APIs and `python -m tensilelite run`.
3. Activate installed package runtime, delete source `bin/Tensile*`, and remove
   `--prebuilt-client` / `PrebuiltClient`.
4. Migrate tox, race-check, coverage, docs, and other non-GEKO consumers.

Why atomic:

- Deleting the source wrappers before GEKO migration would break generated
  workload scripts and optimizer workers.
- GEKO migration before the package-side break is safe because the bridge has
  already installed and bound the correct client.

### PR 14 — generator and metadata cleanup

Purpose: land remaining durable generator, development-publication identity,
precommit, lockfile, and code-generation cleanup from the original final tree.

Why after PR 13:

- These are not part of the GEKO compatibility break.
- It keeps metadata/generator review separate from consumer migration.

### PR 15 — final installed-workflow documentation

Purpose: add the decision records and packaging documents, then reconcile all
public guidance with lower-case package paths, canonical commands, installed
binding, and currently dormant TheRock runtime gates.

Why last among behavior PRs:

- Documentation describes the final behavior, not intermediate compatibility
  states.

### PR 16 — strict runtime initialization cleanup

Purpose: return runtime search-path access to strict initialized-state behavior
once package initialization owns initialization, while retaining the tox-only
identity fallback required for isolated package setup.

### PR 17 — runtime and validator coverage repair

Purpose: repair stale original-final tests that would otherwise regress the
lower-case namespace, generator-version compatibility, canonical redirect,
selected-installation validation, compiler/HIP/device predicate coverage, and
RHEL fallback coverage.

Why separate:

- The tests are substantive correction work, not feature implementation.
- It makes the final non-byte-identical differences easy to audit.

## Review and validation protocol

For every committed slice:

1. A separate read-only subagent reviewed the staged diff before commit.
2. The PR boundary ran `invoke build-client` and editable installation in the
   shared venv, with the selected ROCm identity where required.
3. Focused unit, packaging, CMake, and GEKO tests were run in proportion to the
   boundary's blast radius.
4. The PR-17 reconstructed stack was checked for clean status, valid numbered
   commit subjects, original-branch preservation, and `git diff --check`
   against the original tip.

See `tensilelite_packaging-stack-vs-original-justification.md` for the
remaining intentional final-tree differences and their per-file rationale.
