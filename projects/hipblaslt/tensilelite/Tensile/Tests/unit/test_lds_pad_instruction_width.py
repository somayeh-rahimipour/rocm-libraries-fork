# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The unroll-major LDS pad must be measured in local-read INSTRUCTION widths.

An LDS op is serviced 32 DWORDs per cycle, so a local read moving W bytes per
lane covers 32*4/W lanes per pass, and lane n of a pass sits at byte n*S. Those
lanes land on distinct banks iff

    S = W * odd

i.e. the row stride is an ODD multiple of the per-lane instruction width. A pad
of an EVEN number of instruction widths cannot change that parity, so it cannot
fix a conflict -- which is what `optPad = LocalReadVectorWidth` produces once a
row takes more than one instruction (gfx11/12 WMMA, where LocalReadVectorWidth
== MIInputPerThread).
"""

import pytest

from Tensile.SolutionStructs.LdsPadding import (
    BYTES_PER_REGISTER,
    local_read_block_width,
    local_read_instruction_bytes,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Instruction selection must track memoryInstructions["LocalRead"]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "widthRegisters, blockWidth, why",
    [
        (1, 1, "b32"),
        (2, 2, "b64"),
        (3, 1, "no b96 local read: three b32"),
        (4, 4, "b128"),
        (6, 6, "b192, the widest local read in the table"),
        (8, 4, "b192 does not divide 8, so two b128 -- the gfx11/12 WMMA case"),
        (16, 4, "four b128"),
    ],
)
def test_block_width_matches_the_instruction_table(widthRegisters, blockWidth, why):
    assert local_read_block_width(widthRegisters) == blockWidth, why


def test_widest_entry_is_b192():
    """Pins the table. If a wider local read (e.g. b256, which already exists on
    the LocalWrite side) is added, this fails and the pad rule must be re-checked
    -- silently keeping 16B would re-introduce the conflict this guards."""
    assert max(local_read_block_width(w) for w in range(1, 33)) * BYTES_PER_REGISTER == 24


# ---------------------------------------------------------------------------
# The parity law
# ---------------------------------------------------------------------------

def _lanes_hit_distinct_banks(strideBytes, blockBytes, padBytes, instrBytes):
    banks, bankBytes = 32, 4
    lanesPerPass = banks * bankBytes // instrBytes
    seen = {}
    for lane in range(lanesPerPass):
        addr = lane * strideBytes
        if blockBytes:
            addr += (addr // blockBytes) * padBytes
        for reg in range(instrBytes // bankBytes):
            dword = addr // bankBytes + reg
            seen.setdefault(dword % banks, set()).add(dword)
    return max(len(v) for v in seen.values()) == 1


@pytest.mark.parametrize("multiple, clean", [(1, True), (2, False), (3, True), (4, False)])
def test_only_odd_instruction_multiples_are_conflict_free(multiple, clean):
    stride, instr = 128, 16          # gfx1151 repro: DepthU 32 * bf16 * VW 2
    assert _lanes_hit_distinct_banks(stride, stride, multiple * instr, instr) is clean


def test_legacy_pad_is_an_even_multiple_and_therefore_inert():
    """Pin the defect: lrvw=16 at bf16 is 32B = 2 instruction widths."""
    lrvw, bpe, stride = 16, 2, 32 * 2 * 2
    instr = local_read_instruction_bytes(lrvw, bpe)
    assert instr == 16                                   # two b128, not one 32B op
    assert (lrvw * bpe) // instr == 2                    # legacy pad = 2 widths, even
    assert not _lanes_hit_distinct_banks(stride, 128, lrvw * bpe, instr)
    clamped = min(lrvw, instr // bpe)
    assert clamped == 8                                  # one width
    assert _lanes_hit_distinct_banks(stride, 128, clamped * bpe, instr)


# ---------------------------------------------------------------------------
# The clamp is a no-op wherever a row is a single instruction (all of MFMA/CDNA)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "lrvw, bpe",
    [(1, 4), (2, 4), (4, 4),        # readRegs 1, 2, 4
     (2, 2), (4, 2), (8, 2),
     (4, 1), (8, 1), (16, 1)],
)
def test_clamp_is_a_noop_for_single_instruction_rows(lrvw, bpe):
    """readRegs <= 4 is everything MFMA/CDNA can reach -- the b192 reject in
    Solution.py caps non-WMMA there -- so CDNA layouts must not move."""
    assert int(lrvw * bpe // BYTES_PER_REGISTER) <= 4
    assert min(lrvw, local_read_instruction_bytes(lrvw, bpe) // bpe) == lrvw


@pytest.mark.parametrize("lrvw, bpe, expected", [(16, 2, 8), (32, 1, 16)])
def test_clamp_halves_the_wide_wmma_rows(lrvw, bpe, expected):
    """gfx11/12 WMMA: LocalReadVectorWidth == MIInputPerThread, readRegs == 8."""
    assert int(lrvw * bpe // BYTES_PER_REGISTER) == 8
    assert min(lrvw, local_read_instruction_bytes(lrvw, bpe) // bpe) == expected
