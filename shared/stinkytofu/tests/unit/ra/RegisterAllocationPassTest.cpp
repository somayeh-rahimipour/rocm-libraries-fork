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
#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "AllocationTestUtils.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/transforms/ra/LegacyIdentityAllocator.hpp"
#include "stinkytofu/transforms/ra/RegisterAllocationPass.hpp"
#include "stinkytofu/transforms/ssa/LiftAsmRegistersToSSAPass.hpp"
#include "stinkytofu/transforms/ssa/ReplayLegacyColoringPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

class RegisterAllocationPassTest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = std::make_unique<Function>("kernel");
        setFunctionArch(*func, kRaTestArch);
    }

    BasicBlock* block(const std::string& label) {
        return func->createBasicBlock(label);
    }

    RegisterAllocationOptions legacyApply() {
        RegisterAllocationOptions options;
        options.allocator = "legacy";
        options.applyToOperands = true;
        options.verify = true;
        return options;
    }

    std::unique_ptr<Function> func;
};

class RecolouringAllocator : public RegisterAllocator {
   public:
    const char* name() const override {
        return "recolour-merges";
    }
    AllocatorCapabilities capabilities() const override {
        AllocatorCapabilities caps;
        caps.mayRecolourMerges = true;
        return caps;
    }
    Expected<AllocationResult> allocate(const AllocationContext& context) override {
        allocated = true;
        return createLegacyColoring(context.function);
    }
    bool allocated = false;
};

class SpillingAllocator : public RegisterAllocator {
   public:
    const char* name() const override {
        return "spill";
    }
    AllocatorCapabilities capabilities() const override {
        AllocatorCapabilities caps;
        caps.maySpill = true;
        return caps;
    }
    Expected<AllocationResult> allocate(const AllocationContext& context) override {
        allocated = true;
        return createLegacyColoring(context.function);
    }
    bool allocated = false;
};

}  // namespace

TEST_F(RegisterAllocationPassTest, ApplyWithLegacyMatchesReplay) {
    BasicBlock* entry = block("entry");
    createDsReadB128InBlock(entry, kRaTestArch, 4, 0);
    createVAddInBlock(entry, kRaTestArch, 8, 4, 5);
    const std::string before = physicalIR(*func);

    ASSERT_TRUE(liftForAllocation(*func));
    LegacyIdentityAllocator allocator;
    Expected<AllocationResult> result = allocateRegisters(*func, allocator, legacyApply());

    ASSERT_TRUE(result.hasValue()) << (result.hasValue() ? "" : result.getError());
    EXPECT_EQ(physicalIR(*func), before);
    EXPECT_FALSE(func->hasAttachedSSA());
}

TEST_F(RegisterAllocationPassTest, PassApplyMatchesReplayLegacyColoringPass) {
    BasicBlock* entry = block("entry");
    createDsReadB128InBlock(entry, kRaTestArch, 4, 0);
    createVAddInBlock(entry, kRaTestArch, 8, 4, 5);
    const std::string before = physicalIR(*func);

    PassManager viaReplay;
    registerAllAnalyses(viaReplay.getAnalysisManager());
    viaReplay.setGemmTileConfig(func->getGemmTileConfig());
    viaReplay.addPass(createLiftAsmRegistersToSSAPass());
    viaReplay.addPass(createReplayLegacyColoringPass());
    viaReplay.run(*func);
    const std::string replayed = physicalIR(*func);

    func = std::make_unique<Function>("kernel");
    setFunctionArch(*func, kRaTestArch);
    entry = block("entry");
    createDsReadB128InBlock(entry, kRaTestArch, 4, 0);
    createVAddInBlock(entry, kRaTestArch, 8, 4, 5);

    PassManager viaAlloc;
    registerAllAnalyses(viaAlloc.getAnalysisManager());
    viaAlloc.setGemmTileConfig(func->getGemmTileConfig());
    viaAlloc.addPass(createLiftAsmRegistersToSSAPass());
    viaAlloc.addPass(createRegisterAllocationPass(legacyApply()));
    viaAlloc.run(*func);

    EXPECT_EQ(physicalIR(*func), before);
    EXPECT_EQ(physicalIR(*func), replayed);
    EXPECT_FALSE(func->hasAttachedSSA());
}

TEST_F(RegisterAllocationPassTest, ShadowLeavesAttachedSSAAndOperands) {
    BasicBlock* entry = block("entry");
    createVAddInBlock(entry, kRaTestArch, 2, 0, 1);
    const std::string before = physicalIR(*func);
    ASSERT_TRUE(liftForAllocation(*func));

    RegisterAllocationOptions options;
    options.allocator = "legacy";
    options.applyToOperands = false;
    LegacyIdentityAllocator allocator;
    Expected<AllocationResult> result = allocateRegisters(*func, allocator, options);

    ASSERT_TRUE(result.hasValue()) << (result.hasValue() ? "" : result.getError());
    EXPECT_TRUE(func->hasAttachedSSA());
    EXPECT_EQ(physicalIR(*func), before);
}

TEST_F(RegisterAllocationPassTest, RefusesAnAllocatorThatMayRecolourMerges) {
    createVAddInBlock(block("entry"), kRaTestArch, 2, 0, 1);
    const std::string before = physicalIR(*func);
    ASSERT_TRUE(liftForAllocation(*func));

    RecolouringAllocator allocator;
    Expected<AllocationResult> result = allocateRegisters(*func, allocator, legacyApply());

    EXPECT_TRUE(result.hasError());
    EXPECT_TRUE(contains(result.getError(), "copy insertion")) << result.getError();
    EXPECT_FALSE(allocator.allocated);
    EXPECT_TRUE(func->hasAttachedSSA());
    EXPECT_EQ(physicalIR(*func), before);
}

TEST_F(RegisterAllocationPassTest, RefusesAnAllocatorThatMaySpill) {
    createVAddInBlock(block("entry"), kRaTestArch, 2, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    SpillingAllocator allocator;
    Expected<AllocationResult> result = allocateRegisters(*func, allocator, legacyApply());

    EXPECT_TRUE(result.hasError());
    EXPECT_TRUE(contains(result.getError(), "spilling")) << result.getError();
    EXPECT_FALSE(allocator.allocated);
}

TEST_F(RegisterAllocationPassTest, PassReportsAMissingGraph) {
    createVAddInBlock(block("entry"), kRaTestArch, 2, 0, 1);

    PassContext passCtx;
    passCtx.setRemarksEnabled(true);
    AnalysisManager am;
    registerAllAnalyses(am);

    std::ostringstream captured;
    std::streambuf* previous = std::cerr.rdbuf(captured.rdbuf());
    createRegisterAllocationPass(legacyApply())->run(*func, passCtx, am);
    std::cerr.rdbuf(previous);

    const std::string text = captured.str();
    EXPECT_TRUE(contains(text, "missed: RegisterAllocation")) << text;
    EXPECT_TRUE(contains(text, "no attached SSA")) << text;
}

TEST_F(RegisterAllocationPassTest, PassReportsAnUnknownAllocator) {
    createVAddInBlock(block("entry"), kRaTestArch, 2, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    RegisterAllocationOptions options;
    options.allocator = "does-not-exist";
    options.applyToOperands = true;

    PassContext passCtx;
    passCtx.setRemarksEnabled(true);
    AnalysisManager am;
    registerAllAnalyses(am);

    std::ostringstream captured;
    std::streambuf* previous = std::cerr.rdbuf(captured.rdbuf());
    createRegisterAllocationPass(options)->run(*func, passCtx, am);
    std::cerr.rdbuf(previous);

    const std::string text = captured.str();
    EXPECT_TRUE(contains(text, "is not registered")) << text;
    EXPECT_TRUE(func->hasAttachedSSA());
}

TEST_F(RegisterAllocationPassTest, InjectedAllocatorIsUsed) {
    createVAddInBlock(block("entry"), kRaTestArch, 2, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));
    const std::string before = physicalIR(*func);

    auto injected = std::make_unique<LegacyIdentityAllocator>();
    PassContext passCtx;
    AnalysisManager am;
    registerAllAnalyses(am);
    createRegisterAllocationPass(legacyApply(), std::move(injected))->run(*func, passCtx, am);

    EXPECT_EQ(physicalIR(*func), before);
    EXPECT_FALSE(func->hasAttachedSSA());
}
