// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;

/// Epilogue store-sink pass (gfx1250).
///
/// Sinks each global `buffer_store` in the global-write epilogue as late as it
/// legally can within its basic block, so the subsequent InsertWaitAluPass emits
/// a graduated `s_wait_alu depctr_va_vdst(N)` (near-free) instead of a full
/// `va_vdst(0)` drain — letting the store overlap following VALU compute.
///
/// Movement only; the pass never writes wait counts or s_set_vgpr_msb. It must
/// run BEFORE InsertVgprMsbPass and InsertWaitAluPass so both regenerate for the
/// new order. Intended to run inside a ScopeAdaptor targeting the
/// "globalWriteEpilogue" region so it cannot move a store across the region
/// boundary.
///
/// Two position-dependent rules, each with a control constant:
///   - Bulk sink (`targetValu`): a store in the body sinks as late as legal to
///     hide its va_vdst behind following VALU.
///   - Tail guard (`tailGuard`): the last `tailGuard` stores in the block are
///     left in place (NOT sunk). The tail has little/no VALU runway to hide
///     behind and sits just before s_endpgm, so sinking there only pushes the
///     store's completion closer to kernel exit (exposed drain) for no overlap
///     gain. Skipping the sink is a no-op move — zero hazard risk.
///
/// A (bulk) store stops sinking at the first of:
///   - a later instruction that writes any of the store's data registers
///     (RAW/WAR: e.g. a next-batch buffer_load reusing the reg, or a v_cvt_pk),
///   - a later instruction that writes an SGPR the store reads (the SRD row
///     advance, s_add_u32 sgprSrd*),
///   - any side-effecting / boundary instruction (branch, label, other store,
///     s_nop wait-state, waitcnt),
///   - `targetValu` VALU instructions passed (the sink-distance knob).
///
/// `reverseSink`: process bulk stores bottom-up. In an interleaved
/// `V4 store V4 store` layout this lets an upper store sink into the slot the
/// lower store already vacated, crossing both VALU groups (va_vdst 4 -> 8)
/// instead of stranding on the not-yet-moved lower store. Front-to-back caps
/// each store at the next still-present store.
///
/// `crossLoadcnt`: allow a store to sink across a pure `s_wait_loadcnt`. On
/// gfx12 loadcnt/storecnt are separate counters; a buffer_store bumps STOREcnt
/// while s_wait_loadcnt tests LOADcnt, so crossing it does not perturb the wait.
/// This unblocks the beta!=0 read-modify-write epilogue, where each store is
/// followed by s_wait_loadcnt on the next row's C-load — without this the store
/// stalls at va_vdst(0) and gets no overlap. Still stops at s_wait_storecnt, the
/// combined *_dscnt waits, other stores, and any writesAny data hazard.
/// `msbGuard`: MSB-aware xcnt cost guard. Sink runs before InsertVgprMsbPass, so no
/// s_set_vgpr_msb exists yet; the guard predicts the flip boundaries via the shared
/// computeRequiredMsb() (the same one that pass materializes) and stops a store at
/// the last landing that does not straddle an msb flip. A straddled flip forces
/// InsertVgprMsbPass to emit s_wait_xcnt 0 (the store is in flight across a
/// non-replayable msb). Off = the unconditional sink (current uplifted behaviour).
/// On NonEdge the neighbours share one msb bank so the clean landing == the full
/// landing (no-op, wins preserved); on Edge it backs the store off before the first
/// straddled flip, removing the forced xcnt at the cost of some va_vdst overlap.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createEpilogueStoreSinkPass(unsigned targetValu = 10,
                                                                    unsigned tailGuard = 2,
                                                                    bool reverseSink = true,
                                                                    bool crossLoadcnt = true,
                                                                    bool msbGuard = false);

}  // namespace stinkytofu
