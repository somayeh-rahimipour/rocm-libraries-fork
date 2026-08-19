#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
set -euo pipefail

readonly ORIGINAL="26254111b82a6734053ac7d32ced5fff5e942295"
readonly PR17="27e1d83c21b1c8f5805e6018865ada8190e6615d"
readonly EXPECTED_SHA256="0d1b584e0267ab1c5cc8e821b3e420c669ee22aca7bbbd54ad96331c7a8f7cd8"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifact_dir="${repo_root}/tensilelite_packaging-stack-vs-original-by-commit"
manifest="${artifact_dir}/manifest.tsv"
source_patch="$(mktemp)"
reconstructed="$(mktemp)"
expected_fragment="$(mktemp)"
actual_fragment="$(mktemp)"
trap 'rm -f "${source_patch}" "${reconstructed}" "${expected_fragment}" "${actual_fragment}"' EXIT

excludes=(
  ':(exclude)projects/hipblaslt/tensilelite/PythonBuildGrillingDecisions.md'
  ':(exclude)projects/hipblaslt/tensilelite/PythonBuildImplementationPlan.md'
  ':(exclude)projects/hipblaslt/tensilelite/RocmVersionIdentityInvestigation.md'
  ':(exclude)projects/hipblaslt/tensilelite/RocmWheelRuntimeGrillingDecisions.md'
  ':(exclude)projects/hipblaslt/tensilelite/TheRockPythonBuildImplementationPlan.md'
  ':(exclude)projects/hipblaslt/tensilelite/TheRockRocmWheelPackageRolesResearch.md'
  ':(exclude)projects/hipblaslt/tensilelite/docs/PackagingDecisions.md'
  ':(exclude)projects/hipblaslt/tensilelite/docs/PackagingPlan.md'
  ':(exclude)projects/hipblaslt/tensilelite/docs/Public.md'
  ':(exclude)projects/hipblaslt/tensilelite/docs/TensileLiteNamingAudit.md'
)

git -C "${repo_root}" diff --no-ext-diff --full-index "${ORIGINAL}" "${PR17}" -- . "${excludes[@]}" > "${source_patch}"
actual_sha="$(sha256sum "${source_patch}" | awk '{print $1}')"
[[ "${actual_sha}" == "${EXPECTED_SHA256}" ]] || {
  echo "aggregate digest mismatch: expected ${EXPECTED_SHA256}, got ${actual_sha}" >&2
  exit 1
}

source_lines="$(wc -l < "${source_patch}")"
expected_start=1
mapped_headers=0
mapped_hunks=0
while IFS=$'\t' read -r source_start source_end kind path owner_type owner_commit owner_subject artifact artifact_start artifact_end basis; do
  [[ "${source_start}" == \#* || "${source_start}" == "source_start" ]] && continue
  [[ "${source_start}" -eq "${expected_start}" ]] || {
    echo "non-contiguous manifest source range at ${source_start} (expected ${expected_start})" >&2
    exit 1
  }
  sed -n "${source_start},${source_end}p" "${source_patch}" > "${expected_fragment}"
  sed -n "${artifact_start},${artifact_end}p" "${artifact_dir}/${artifact}" > "${actual_fragment}"
  cmp -s "${expected_fragment}" "${actual_fragment}" || {
    echo "fragment mismatch for ${path}:${source_start}-${source_end} -> ${artifact}:${artifact_start}-${artifact_end}" >&2
    exit 1
  }
  cat "${expected_fragment}" >> "${reconstructed}"
  expected_start=$((source_end + 1))
  [[ "${kind}" == "file-header" ]] && mapped_headers=$((mapped_headers + 1))
  [[ "${kind}" == "hunk" ]] && mapped_hunks=$((mapped_hunks + 1))
done < "${manifest}"

[[ "${expected_start}" -eq $((source_lines + 1)) ]] || {
  echo "manifest ended at line $((expected_start - 1)); aggregate has ${source_lines} lines" >&2
  exit 1
}
cmp -s "${source_patch}" "${reconstructed}" || {
  echo "reconstructed aggregate differs from regenerated source patch" >&2
  exit 1
}
[[ "${mapped_headers}" -eq "$(grep -c '^diff --git ' "${source_patch}")" ]] || {
  echo "file-header count mismatch" >&2
  exit 1
}
[[ "${mapped_hunks}" -eq "$(grep -c '^@@ ' "${source_patch}")" ]] || {
  echo "hunk count mismatch" >&2
  exit 1
}

echo "verified: ${source_lines} source lines, ${mapped_headers} file headers, and ${mapped_hunks} hunks partitioned exactly"
