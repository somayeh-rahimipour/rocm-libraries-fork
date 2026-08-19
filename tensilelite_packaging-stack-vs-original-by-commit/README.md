# TensileLite packaging stack audit, split by current commit


This directory replaces the monolithic `tensilelite_packaging-stack-vs-original.patch` audit artifact. It partitions the current PR-17-versus-original aggregate diff at unified-hunk granularity, so reviews can proceed one current commit artifact at a time.


- Source comparison: `26254111b82a6734053ac7d32ced5fff5e942295` (original tip) → `27e1d83c21b1c8f5805e6018865ada8190e6615d` (current PR-17 tip).
- Shared stack base: `2b94440914ea54bcff909105485381a8d53151de`.
- Source digest: SHA-256 `0d1b584e0267ab1c5cc8e821b3e420c669ee22aca7bbbd54ad96331c7a8f7cd8`; 2035 literal lines, 39 file diffs, and 109 unified hunks.
- Output: 19 artifacts. All 109 hunk bodies occur in exactly one artifact.

Each subject-named `.patch` file contains the hunk bodies owned by that current stack commit. A file header is repeated when hunk ownership splits one source file; that repetition makes each artifact readable and is explicitly excluded from duplicate-hunk accounting. `manifest.tsv` is the canonical source-to-artifact mapping.

## Attribution

For each hunk, the owner is the current-stack commit with the greatest exact overlap of nonblank added/deleted lines. Ties select the later current-stack commit and retain every candidate score in `manifest.tsv`. Whitespace-only hunks use a documented adjacent-edit assignment.

20 hunks contain text attributable to more than one current commit. Their owner is still singular at the requested hunk granularity; inspect the `candidates=` evidence before judging them.

Two hunks have no truthful current-stack owner and are isolated in `unattributed-vs-original-no-current-stack-commit.patch`:
- `projects/hipblaslt/tensilelite/tasks.py:318-326` — no current-stack commit: current inherited the f-string from shared PR 1; original changed it later.
- `projects/hipblaslt/tensilelite/tensilelite/tensilelite_create_library/run.py:1620-1695` — no current-stack commit: original contains duplicate license blocks absent from shared PR 1 and the rebuilt stack.

## Verification

Run:

```bash
./tensilelite_packaging-stack-vs-original-by-commit/verify_split.sh
```

The verifier regenerates the historical aggregate diff with `--full-index` and the ten approved PR-21 decision records excluded, verifies the recorded SHA-256, compares every manifest source range byte-for-byte with its designated artifact range, proves source ranges are contiguous and complete, checks hunk/header counts, then reconstructs the aggregate byte-for-byte from the mapped fragments. It does not need the removed monolithic patch file.

## Artifacts

| Artifact | Current commit | Hunk count |
| --- | --- | ---: |
| `13-docs-geko-describe-installed-tensilelite-workflows.patch` | `32eb3a8485` — 13-docs(geko): describe installed TensileLite workflows | 3 |
| `13-fix-geko-install-and-bind-the-built-tensilelite-client.patch` | `8eeb831539` — 13-fix(geko): install and bind the built TensileLite client | 7 |
| `13-refactor-geko-use-installed-tensilelite-interfaces.patch` | `cc433700a9` — 13-refactor(geko): use installed TensileLite interfaces | 6 |
| `13-refactor-tensilelite-activate-installed-runtime-and-retire-source-entrypoints.patch` | `1a1d3704da` — 13-refactor(tensilelite): activate installed runtime and retire source entrypoints | 10 |
| `13-test-tensilelite-configure-installed-tox-workflows.patch` | `dbb5ce3a2a` — 13-test(tensilelite): configure installed tox workflows | 3 |
| `14-refactor-tensilelite-finalize-generator-and-metadata-cleanup.patch` | `210ae6d798` — 14-refactor(tensilelite): finalize generator and metadata cleanup | 3 |
| `15-docs-tensilelite-finalize-installed-workflow-guidance.patch` | `5ac5a9cfbf` — 15-docs(tensilelite): finalize installed workflow guidance | 11 |
| `17-test-tensilelite-preserve-runtime-and-validator-coverage.patch` | `27e1d83c21` — 17-test(tensilelite): preserve runtime and validator coverage | 2 |
| `2-fix-tensilelite-complete-moved-package-consumer-closure.patch` | `42df169903` — 2-fix(tensilelite): complete moved package consumer closure | 19 |
| `2-refactor-tensilelite-move-test-namespace-paths-part-6.patch` | `2bf56bbfec` — 2-refactor(tensilelite): move test namespace paths (part 6) | 1 |
| `2-refactor-tensilelite-update-test-namespace-references-part-3.patch` | `449986d78d` — 2-refactor(tensilelite): update test namespace references (part 3) | 1 |
| `3-fix-tensilelite-update-consumers-for-lower-case-internal-modules.patch` | `ca4af3392a` — 3-fix(tensilelite): update consumers for lower-case internal modules | 20 |
| `4-build-tensilelite-make-rocisa-externally-supplied.patch` | `cb4bda2dec` — 4-build(tensilelite): make rocisa externally supplied | 1 |
| `4-refactor-tensilelite-invoke-library-generation-through-package-handlers.patch` | `b577f90cdb` — 4-refactor(tensilelite): invoke library generation through package handlers | 4 |
| `7-build-tensilelite-bind-source-installs-to-built-clients.patch` | `1a8e68a435` — 7-build(tensilelite): bind source installs to built clients | 5 |
| `8-fix-tensilelite-remove-pandas-from-the-generator-path.patch` | `d054f7a7a9` — 8-fix(tensilelite): remove pandas from the generator path | 3 |
| `8-refactor-tensilelite-classify-source-only-unit-tests.patch` | `c078e1ead9` — 8-refactor(tensilelite): classify source-only unit tests | 6 |
| `9-build-tensilelite-make-rocm-version-a-wheel-build-input.patch` | `15477ccba3` — 9-build(tensilelite): make ROCm version a wheel build input | 2 |
| `unattributed-vs-original-no-current-stack-commit.patch` | unattributed vs original: no current stack commit | 2 |
