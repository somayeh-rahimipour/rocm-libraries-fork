#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

fail() { echo "FAIL: $*" >&2; exit 1; }
pass() { echo "PASS: $*"; }

python="${PYTHON:-python3}"
abi_tag="$("$python" - <<'PY'
import sysconfig
print(sysconfig.get_config_var("EXT_SUFFIX"))
PY
)"
echo "=== interpreter: $("$python" -V), ext suffix: ${abi_tag} ==="

if command -v git >/dev/null 2>&1 && git -C "$here" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    have_git=1
    src_before="$(git -C "$here" status --porcelain -- src csrc)"
else
    have_git=0
    echo "SKIP: git unavailable; source-tree mutation diff will not run (the src/origami .so leak check below still runs)"
fi

build_args=()
if [ "${ORIGAMI_BUILD_FROM_SOURCE:-OFF}" = "ON" ]; then
    build_args+=(-C cmake.define.ORIGAMI_BUILD_FROM_SOURCE=ON)
fi

rm -rf dist
"$python" -m build --wheel --outdir dist ${build_args[@]+"${build_args[@]}"}
whl="$(ls dist/*.whl)"
[ -f "$whl" ] || fail "no wheel produced"
pass "built ${whl}"

listing="$("$python" -m zipfile -l "$whl")"
echo "$listing"

n_ext="$(echo "$listing" | awk '{print $1}' | grep -Fxc "origami/origami${abi_tag}" || true)"
[ "$n_ext" = "1" ] || fail "expected exactly one extension origami/origami${abi_tag}, found ${n_ext}"
pass "exactly one extension for this ABI"

echo "$listing" | grep -q "origami/__init__.py" || fail "__init__.py missing from wheel"
echo "$listing" | grep -q "origami/selector.py" || fail "selector.py missing from wheel"
pass "pure Python files recorded in wheel"

if echo "$listing" | grep -Eq "liborigami\.so"; then
    fail "liborigami is bundled in the wheel; it must be linked dynamically, not shipped"
fi
pass "no liborigami bundled"

if echo "$listing" | grep -q "bindings.cpp"; then
    fail "C++ source bindings.cpp leaked into the wheel"
fi
pass "no C++ source in wheel"

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT
"$python" -m zipfile -e "$whl" "$workdir"
ext_path="$(find "$workdir" -name "origami${abi_tag}")"
[ -f "$ext_path" ] || fail "could not extract extension from wheel"
readelf_bin=""
if command -v readelf >/dev/null 2>&1; then
    readelf_bin=readelf
elif command -v llvm-readelf >/dev/null 2>&1; then
    readelf_bin=llvm-readelf
else
    fail "neither readelf nor llvm-readelf found; cannot verify NEEDED liborigami.so.1"
fi
"$readelf_bin" -d "$ext_path" | grep -q "NEEDED.*liborigami.so.1" \
    || fail "extension does not record NEEDED liborigami.so.1"
pass "extension NEEDED liborigami.so.1"
if "$readelf_bin" -d "$ext_path" | grep -Eq "NEEDED.*liborigami\.so[^.]"; then
    fail "extension NEEDED an unversioned liborigami.so"
fi

if command -v auditwheel >/dev/null 2>&1; then
    auditwheel show "$whl" || true
fi

if [ "$have_git" = "1" ]; then
    src_after="$(git -C "$here" status --porcelain -- src csrc)"
    [ "$src_before" = "$src_after" ] || fail "build mutated the source tree:
$(diff <(echo "$src_before") <(echo "$src_after") || true)"
    pass "source tree unchanged by build (git)"
fi
[ -z "$(find src/origami -name '*.so' 2>/dev/null)" ] || fail "a compiled extension was left in src/origami"
pass "no compiled extension left in src/origami"

venv="$workdir/venv"
"$python" -m venv "$venv"
"$venv/bin/python" -m pip install --quiet --force-reinstall --no-deps "$whl"
ld_parts=()
[ -n "${ORIGAMI_RUNTIME_LIB_DIR:-}" ] && ld_parts+=("$ORIGAMI_RUNTIME_LIB_DIR")
[ -n "${CMAKE_PREFIX_PATH:-}" ] && ld_parts+=("${CMAKE_PREFIX_PATH}/lib")
[ -n "${LD_LIBRARY_PATH:-}" ] && ld_parts+=("$LD_LIBRARY_PATH")
if [ ${#ld_parts[@]} -gt 0 ]; then
    export LD_LIBRARY_PATH="$(IFS=:; echo "${ld_parts[*]}")"
fi
"$venv/bin/python" -c "import origami; print('imported origami', origami.__version__)" \
    || fail "import origami failed (is liborigami.so.1 on the loader path?)"
pass "import succeeds with runtime on loader path"

echo "=== all wheel-release assertions passed ==="
