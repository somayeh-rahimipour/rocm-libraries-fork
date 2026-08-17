// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/EpilogueStoreSinkPass.hpp"

#include <iostream>
#include <set>
#include <utility>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/VgprMsbEncoding.hpp"

#define DEBUG_TYPE "EpilogueStoreSinkPass"

namespace {
using namespace stinkytofu;

// A single physical register unit: (type, index). b128 store data spans 4 units.
using RegUnit = std::pair<RegType, unsigned>;

// Expand every (isRegister) operand in \p regs into individual (type, idx) units.
static void collectUnits(const std::vector<StinkyRegister>& regs, std::set<RegUnit>& out) {
    for (const StinkyRegister& r : regs) {
        if (!r.isRegister()) continue;
        for (unsigned off = 0; off < r.reg.num; ++off) {
            out.insert({r.reg.type, r.reg.idx + off});
        }
    }
}

// Does \p inst write any unit in \p units? (WAW/WAR/RAW hazard against a set.)
static bool writesAny(const StinkyInstruction& inst, const std::set<RegUnit>& units) {
    for (const StinkyRegister& d : inst.getDestRegs()) {
        if (!d.isRegister()) continue;
        for (unsigned off = 0; off < d.reg.num; ++off) {
            if (units.count({d.reg.type, d.reg.idx + off})) return true;
        }
    }
    return false;
}

// Does this instruction bump the VA_VDST counter? Must match InsertWaitAluPass's
// producer classification (isVectorALU || isTranscendental || isMatrixInstruction)
// so our sink distance equals the va_vdst(N) the wait pass will emit.
static bool bumpsVaVdst(const StinkyInstruction& inst) {
    return isVectorALU(inst) || isTranscendental(inst) || isMatrixInstruction(inst);
}

// A pure `s_wait_loadcnt` (NOT storecnt, NOT the combined *_dscnt forms). On
// gfx12 the legacy vmcnt is split into separate loadcnt/storecnt: a buffer_store
// increments STOREcnt, while s_wait_loadcnt tests LOADcnt only. So a store may
// legally sink across s_wait_loadcnt — moving it does not change what that wait
// resolves (the store was never counted by loadcnt). This is the beta!=0 RMW
// epilogue case where the store is followed by s_wait_loadcnt on the next row's
// C-load; crossing it lets the store reach the following cvt/fmac VALU runway.
// We deliberately do NOT cross s_wait_storecnt (gates the store's own counter)
// or the combined *_dscnt variants (conservative).
static bool isPureLoadcntWait(const StinkyInstruction& inst) {
    return inst.getUnifiedOpcode() == GFX::s_wait_loadcnt;
}

// Predicted s_set_vgpr_msb value an instruction will require, or -1 if it carries
// no VGPR MSB (SALU, waits, side-effecting). Uses computeRequiredMsb — the SAME
// shared predictor InsertVgprMsbPass materializes from — so the flip boundaries we
// see here at sink time (before any s_set_vgpr_msb exists) are exactly the ones
// that pass will emit. This is how the guard is "MSB-aware" without the msb IR.
static int requiredMsbOf(const StinkyInstruction& inst) {
    auto [setVal, hasVgpr] = computeRequiredMsb(&inst);
    return hasVgpr ? setVal : -1;
}

// Sink one buffer_store within its block.
// storeIt points at the store; the [msb?/wait] preceding it are left in place
// (regenerated later by InsertVgprMsb / InsertWaitAlu).
//
// Returns the number of VALU ops the store was sunk past (0 = not moved).
//
// msbGuard: when true, stop sinking at the last landing where the store does NOT
// straddle an s_set_vgpr_msb flip. A store left in flight across a (non-replayable)
// msb forces InsertVgprMsbPass to emit s_wait_xcnt 0 (§5.2/§8.3): the fold rule
// only fires when the msb immediately precedes its own VMEM, not when a store sits
// between the msb and the VALU it configures. On NonEdge the store's neighbours
// share one msb bank, so the last clean landing == the full targetValu landing —
// the guard is a no-op and the +26%/+6% wins are preserved. On Edge every element
// flips the bank 3x, so the guard backs the store off before the first straddled
// flip, trading a little va_vdst overlap for removing a forced xcnt drain.
static unsigned sinkOneStore(BasicBlock& bb, BasicBlock::iterator storeIt, unsigned targetValu,
                             bool crossLoadcnt, bool msbGuard) {
    StinkyInstruction& store = getStinkyInst(storeIt);

    // The store's dependency footprint:
    //  - data + address regs it READS: nothing that writes them may be crossed
    //    (RAW producer is behind us; a later writer would be WAR — reg reuse,
    //     e.g. a next-batch buffer_load into the store's data regs).
    //  - it must not cross a writer of any reg it reads (covers SGPR SRD advance
    //    s_add_u32 sgprSrd*, which the store reads as an address base).
    std::set<RegUnit> readUnits;
    collectUnits(store.getSrcRegs(), readUnits);

    // MSB context in effect just before the store: the required-msb of the nearest
    // preceding computable instruction. A later inst whose required-msb differs is
    // a flip the store would straddle if it lands after that inst.
    int curMsb = -1;
    if (msbGuard) {
        for (BasicBlock::iterator b = storeIt; b != bb.begin();) {
            --b;
            auto* p = dyn_cast<StinkyInstruction>(b.getNodePtr());
            if (!p) break;
            int m = requiredMsbOf(*p);
            if (m >= 0) { curMsb = m; break; }
        }
    }

    BasicBlock::iterator it = std::next(storeIt);
    BasicBlock::iterator dest = storeIt;  // last legal insertion point (before `it`)
    BasicBlock::iterator cleanDest = storeIt;  // last msb-clean landing (guard on)
    unsigned valuPassed = 0;
    unsigned valuAtCleanDest = 0;  // VALU passed as of the last msb-clean landing

    while (it != bb.end() && valuPassed < targetValu) {
        IRBase* node = it.getNodePtr();
        auto* instPtr = dyn_cast<StinkyInstruction>(node);
        if (!instPtr) break;  // label / directive — hard boundary
        StinkyInstruction& cand = *instPtr;

        // Hard boundaries: other side-effecting insts (branch, another store,
        // waitcnt, barrier, the dwordx4 s_nop wait-state). Stop before them.
        // Exception: a pure s_wait_loadcnt gates LOADcnt, but this store bumps
        // STOREcnt (gfx12 split counters), so crossing it is safe and does not
        // perturb the wait. Cross it (still subject to the writesAny data check
        // below) to reach the beta!=0 next-row cvt/fmac VALU runway.
        if (hasSideEffect(cand) && !(crossLoadcnt && isPureLoadcntWait(cand))) break;

        // WAR/WAW: candidate writes a reg the store reads → cannot sink past it.
        if (writesAny(cand, readUnits)) break;

        // Legal to cross this instruction.
        if (bumpsVaVdst(cand)) ++valuPassed;
        ++it;
        dest = std::prev(it);

        // Track the furthest landing that does not straddle an msb flip. Landing
        // after `cand` is clean iff the MSB in effect has not changed since the
        // store's original position; the first differing required-msb marks the
        // flip the store must not cross.
        if (msbGuard) {
            int m = requiredMsbOf(cand);
            if (m >= 0 && m != curMsb) break;  // flip — stop before straddling it
            cleanDest = dest;
            valuAtCleanDest = valuPassed;
        }
    }

    if (msbGuard) {
        dest = cleanDest;
        valuPassed = valuAtCleanDest;
    }

    if (valuPassed == 0) return 0;  // no room / nothing to gain

    // Move the store to just AFTER `dest` (i.e. before std::next(dest)).
    BasicBlock::iterator insertPos = std::next(dest);
    bb.removeIR(&store);
    bb.insertIR(insertPos, &store);
    return valuPassed;
}

size_t sinkStoresInBlock(BasicBlock& bb, unsigned targetValu, unsigned tailGuard, bool reverseSink,
                         bool crossLoadcnt, bool msbGuard) {
    size_t moved = 0;
    // Snapshot store iterators first: moving one store must not disturb the walk.
    std::vector<BasicBlock::iterator> stores;
    for (auto it = bb.begin(); it != bb.end(); ++it) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst && isGlobalMemStore(*inst)) stores.push_back(it);
    }
    // Tail guard: leave the last `tailGuard` stores where codegen put them. They
    // have little runway and sit right before s_endpgm, so sinking only extends
    // the exposed drain. Not sinking them is a no-op move (no hazard risk).
    const size_t sinkCount = stores.size() > tailGuard ? stores.size() - tailGuard : 0;
    // Reverse (bottom-up) order lets an upper store sink into the slot a lower
    // store already vacated, so in an interleaved `V4 store V4 store` layout the
    // upper store crosses BOTH VALU groups (va_vdst 4 -> 8) instead of stranding
    // on the not-yet-moved lower store. Front-to-back would cap each store at the
    // next (still-present) store. A later store's move never invalidates an
    // earlier store's snapshot iterator (different node), so reverse is safe.
    for (size_t k = 0; k < sinkCount; ++k) {
        const size_t i = reverseSink ? (sinkCount - 1 - k) : k;
        if (sinkOneStore(bb, stores[i], targetValu, crossLoadcnt, msbGuard) > 0) ++moved;
    }
    return moved;
}

class EpilogueStoreSinkPass : public StinkyInstPass {
   public:
    static char ID;
    EpilogueStoreSinkPass(unsigned targetValu, unsigned tailGuard, bool reverseSink,
                          bool crossLoadcnt, bool msbGuard)
        : targetValu_(targetValu),
          tailGuard_(tailGuard),
          reverseSink_(reverseSink),
          crossLoadcnt_(crossLoadcnt),
          msbGuard_(msbGuard) {}

    const char* getName() const override {
        return "EpilogueStoreSinkPass";
    }

    PassID getPassID() const override {
        return &EpilogueStoreSinkPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        for (BasicBlock& bb : func) {
            if (!passCtx.shouldProcessBasicBlock(bb)) continue;
            const size_t moved = sinkStoresInBlock(bb, targetValu_, tailGuard_, reverseSink_,
                                                   crossLoadcnt_, msbGuard_);
            PASS_DEBUG(std::cerr << "[EpilogueStoreSinkPass] bb=\"" << bb.getLabel()
                                 << "\" sunk_stores=" << moved << " target=" << targetValu_
                                 << " tailGuard=" << tailGuard_ << " reverse=" << reverseSink_
                                 << " crossLoadcnt=" << crossLoadcnt_ << " msbGuard=" << msbGuard_
                                 << "\n");
        }
        return preserveCFGAnalyses();
    }

   private:
    unsigned targetValu_;
    unsigned tailGuard_;
    bool reverseSink_;
    bool crossLoadcnt_;
    bool msbGuard_;
};

char EpilogueStoreSinkPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createEpilogueStoreSinkPass(unsigned targetValu, unsigned tailGuard,
                                                  bool reverseSink, bool crossLoadcnt,
                                                  bool msbGuard) {
    return std::make_unique<EpilogueStoreSinkPass>(targetValu, tailGuard, reverseSink, crossLoadcnt,
                                                   msbGuard);
}
}  // namespace stinkytofu
