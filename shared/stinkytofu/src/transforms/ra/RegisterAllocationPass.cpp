/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#include "stinkytofu/transforms/ra/RegisterAllocationPass.hpp"

#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/ssa/SSALiveIntervals.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/AsmTargetRegisters.hpp"
#include "stinkytofu/support/LoopDetection.hpp"
#include "stinkytofu/support/OptimizationRemark.hpp"
#include "stinkytofu/transforms/ra/AllocationConstraints.hpp"
#include "stinkytofu/transforms/ra/AllocationVerifier.hpp"
#include "stinkytofu/transforms/ra/AllocatorRegistry.hpp"
#include "stinkytofu/transforms/ssa/SSADestruction.hpp"

#define DEBUG_TYPE "RegisterAllocationPass"

namespace stinkytofu {
namespace {

constexpr const char* kPassName = "RegisterAllocation";

const BasicBlock* findExcludedBlock(const Function& func, const PassContext& passCtx) {
    for (const BasicBlock& bb : func) {
        if (!passCtx.shouldProcessBasicBlock(bb)) return &bb;
    }
    return nullptr;
}

}  // namespace

Expected<AllocationResult> allocateRegisters(Function& function, RegisterAllocator& allocator,
                                             const RegisterAllocationOptions& options) {
    if (!function.hasAttachedSSA()) {
        return Expected<AllocationResult>::Error("@" + function.getName() +
                                                 ": no attached SSA; nothing to colour");
    }

    const AllocatorCapabilities caps = allocator.capabilities();
    if (caps.maySpill) {
        return Expected<AllocationResult>::Error("@" + function.getName() + ": allocator '" +
                                                 allocator.name() +
                                                 "' requires spilling, which is not implemented");
    }
    if (caps.mayRecolourMerges) {
        return Expected<AllocationResult>::Error(
            "@" + function.getName() + ": allocator '" + allocator.name() +
            "' may recolour merges, which needs copy insertion that is not implemented");
    }

    const SSALiveIntervals intervals = computeSSALiveIntervals(function);
    AsmTargetRegisters target = AsmTargetRegisters::forFunction(function);
    const AllocationConstraints constraints = AllocationConstraints::build(function, target);
    const std::vector<Loop> loops = detectLoops(function);
    const AllocationContext context{function,    intervals, target,
                                    constraints, loops,     options.allocateSgpr};

    Expected<AllocationResult> allocated = allocator.allocate(context);
    if (allocated.hasError()) return allocated;

    if (options.verify) {
        const AllocationVerificationResult checked =
            verifyAllocation(function, *allocated, context);
        if (!checked.ok()) {
            return Expected<AllocationResult>::Error(checked.toString());
        }
    }

    if (options.applyToOperands) {
        const SSADestructionResult destroyed = destroyAttachedSSA(function, *allocated);
        if (!destroyed.ok()) {
            return Expected<AllocationResult>::Error(destroyed.toString());
        }
    }

    return allocated;
}

class RegisterAllocationPassImpl : public Pass {
   public:
    static char ID;

    RegisterAllocationPassImpl(RegisterAllocationOptions options,
                               std::unique_ptr<RegisterAllocator> allocator)
        : options_(std::move(options)), allocator_(std::move(allocator)) {}

    const char* getName() const override {
        return "Register Allocation";
    }

    PassID getPassID() const override {
        return &RegisterAllocationPassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager&) override {
        if (const BasicBlock* excluded = findExcludedBlock(func, passCtx)) {
            missed(passCtx, "@" + func.getName() + ": basic-block filtering excludes ^" +
                                excluded->getLabel() +
                                "; register allocation needs the whole "
                                "function");
            return preserveCFGAnalyses();
        }

        if (!func.hasAttachedSSA()) {
            missed(passCtx, "@" + func.getName() + ": no attached SSA; nothing to colour");
            return preserveCFGAnalyses();
        }

        if (allocator_ == nullptr) {
            allocator_ = AllocatorRegistry::createAllocator(options_.allocator);
            if (allocator_ == nullptr) {
                missed(passCtx, "@" + func.getName() + ": allocator '" + options_.allocator +
                                    "' is not registered");
                return preserveCFGAnalyses();
            }
        }

        Expected<AllocationResult> result = allocateRegisters(func, *allocator_, options_);
        if (result.hasError()) {
            PASS_DEBUG(std::cerr << "RegisterAllocation: " << result.getError() << "\n");
            missed(passCtx, result.getError());
            return preserveCFGAnalyses();
        }

        const std::string summary = "@" + func.getName() + ": coloured " +
                                    std::to_string(result->valueCount()) + " value(s) with " +
                                    allocator_->name();
        if (options_.applyToOperands) {
            emitRemark(passCtx, {OptimizationRemark::Kind::Passed, kPassName, "AllocatedRegisters",
                                 summary});
        } else {
            emitRemark(passCtx, {OptimizationRemark::Kind::Analysis, kPassName, "ShadowColoring",
                                 summary + " (shadow, not applied)"});
        }
        return preserveCFGAnalyses();
    }

   private:
    static void missed(const PassContext& passCtx, const std::string& message) {
        emitRemark(passCtx, {OptimizationRemark::Kind::Missed, kPassName, "NotAllocated", message});
    }

    RegisterAllocationOptions options_;
    std::unique_ptr<RegisterAllocator> allocator_;
};

char RegisterAllocationPassImpl::ID = 0;

std::unique_ptr<Pass> createRegisterAllocationPass(RegisterAllocationOptions options,
                                                   std::unique_ptr<RegisterAllocator> allocator) {
    return std::make_unique<RegisterAllocationPassImpl>(std::move(options), std::move(allocator));
}

}  // namespace stinkytofu
