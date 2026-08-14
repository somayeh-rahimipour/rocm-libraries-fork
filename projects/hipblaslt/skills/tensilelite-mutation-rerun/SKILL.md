---
name: tensilelite-mutation-rerun
description: Guide safe, repeatable TensileLite mutation-test reruns with mutmut. Use when Codex is asked to plan, run, resume, or audit a TensileLite mutation campaign; choose tests for a target Python module; inspect a mutation that tests did not detect; verify that a new test detects it; reproduce AIHPBLAS-3868 results; or hand off evidence. Running mutmut requires Linux or Windows Subsystem for Linux (WSL), normally through Docker.
---

# TensileLite Mutation Rerun

Mutation testing makes small, temporary changes to source code and checks
whether the tests fail. Each temporary change is a mutant. Tests kill a mutant
when they detect its changed behavior; a mutant survives when the tests still
pass.

Use mutmut to create and run the mutants. Use the repository scripts to record
the environment, select one bounded run, distinguish assertion failures from
tool failures, and restore every edited file. Mutmut can run without these
scripts, but the extra checks make a TensileLite result repeatable and safe to
hand to another developer.

## Establish the environment

1. Find the repository root and the `projects/hipblaslt/tensilelite` source
   directory.
2. Read `projects/hipblaslt/tensilelite/Tensile/Tests/unit/mutation/README.md`
   and the `[tool.mutmut]` table in
   `projects/hipblaslt/tensilelite/pyproject.toml`.
3. Require Linux or Windows Subsystem for Linux (WSL). Mutmut 3.6 exits on
   native Windows because it uses `fork`, Unix resource limits, and Unix
   signals. A Windows host may run a Linux container or WSL, but do not claim
   that native Windows is supported.
4. Find the existing mutation-test container and its `/work` mount. Ask for
   the container name when it cannot be discovered safely.
5. Do not push, edit pull requests, or update Jira unless the user explicitly
   authorizes that external operation.

## Run one mutation slice

A slice is one bounded mutation run against a selected source module and test
set.

1. **Choose one target module.** Keep the slice small enough to explain and
   count independently.
2. **Choose and validate the covering tests.** Read
   [references/covering-set.md](references/covering-set.md). Stop unless pytest
   succeeds and the coverage report shows that the selected tests execute the
   required percentage of the exact target file.
3. **Record the environment before editing files.** Run `slice-preflight.sh`
   with the slice ID, target module, container, and a slice-specific output
   directory. Stop when version-controlled source already contains edits or
   the container is missing.
4. **Back up the configuration.** Run `pyproject-mutmut.sh backup`, then set
   the reviewed `only_mutate` and `pytest_add_cli_args_test_selection` values.
   Do not change `mutate_only_covered_lines = false` until the rocisa
   unload/re-import crash documented in `pyproject.toml` is resolved.
5. **Run the selected tests on unchanged source.** They must pass before
   mutmut starts. A failure makes the slice invalid, so do not classify
   mutants from that run.
6. **Limit parallel mutation workers.** Use `mutmut run --max-children 32`
   unless the reviewed slice requires a lower value. Record the command,
   source-code version, container image, mutmut version, exit status, and
   result counts.
7. **Inspect every result that was not killed.** Use `mutmut results`,
   `mutmut show`, and `mutmut tests-for-mutant`. Keep survived, no-test,
   timeout, and tool-failure results separate.
8. **Add tests that distinguish the changed behavior.** For every proposed
   assertion, state which source change makes it fail. Do not add assertions
   only to increase coverage, silently skip tests, or assert behavior that the
   current source does not implement.
9. **Verify every claimed kill.** Rerun the named mutant with one worker. When
   a manifest is available, use `mutmut-verify.sh`. Count the mutant as killed
   only when the test passes on unchanged source, fails with pytest assertion
   status 1 on changed source, and the source file is restored. Collection,
   usage, internal, timeout, and interruption errors do not prove a kill.
10. **Review claims of identical behavior carefully.** Require a concrete
    explanation that covers every valid input. Do not add `pragma: no mutate`
    merely because writing a distinguishing test is difficult.
11. **Restore files after every outcome.** Run `pyproject-mutmut.sh restore`
    and `assert-clean`. Confirm that no temporary source change remains after
    success, failure, or interruption.
12. **Write the handoff.** Report the target, exact test selection,
    source-code version, environment, unchanged-source result, complete result
    counts, tests added, remaining risks, and evidence paths. Do not improve a
    score by hiding timeouts, tool failures, or other unresolved results.

## Guardrails

- Apply a mutant and restore source in one serial process.
- Never use the package `TOTAL` coverage row as proof for one target file.
- Never accept coverage from a failed pytest run.
- Never replace an unproven focused test set with the full unit suite without
  saying so.
- Always set `--max-children`; the host CPU count can differ between runs.
- Do not update every saved expected result at once. Follow the hipBLASLt
  snapshot rules and update only the reviewed tests.
- Preserve unrelated modified or untracked user files.
- Prefer a committed test selection over rediscovering it for every rerun.
  Validate it again after the source or test tree changes.

## Required outcome

End with one explicit state:

- **Certified:** the unchanged-source tests passed, the selected tests met the
  target-file coverage requirement, every result was counted, claimed kills
  were verified, identical-behavior claims were explained, and files were
  restored.
- **Deferred:** the selected tests did not meet the coverage requirement or
  the target file was missing from the report.
- **Inconclusive:** an environment, test collection, timeout, tool, or
  restoration failure prevents a trustworthy result.
- **Blocked:** a named external dependency or user decision is required.

Never call an incomplete or partly counted run certified.
