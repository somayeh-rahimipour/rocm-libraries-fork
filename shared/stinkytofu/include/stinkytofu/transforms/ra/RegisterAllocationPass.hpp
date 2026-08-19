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
#pragma once

// One pass for every colouring policy.
//
// The driver owns everything policy-independent: fetch analyses, build
// constraints, refuse a policy whose capabilities lowering cannot honour,
// call allocate, verify, then optionally apply. Injection at construction
// follows createStinkyWmmaVgprReorderPass.

#include <memory>
#include <string>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/analysis/ssa/SSAAllocation.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"
#include "stinkytofu/transforms/ra/RegisterAllocator.hpp"

namespace stinkytofu {

class Pass;

struct RegisterAllocationOptions {
    /// Registry name. "greedy" is the first non-identity policy; until it is
    /// registered, pass allocator=legacy.
    std::string allocator = "greedy";
    bool allocateSgpr = false;
    bool applyToOperands = false;  // false = shadow
    bool verify = true;
};

/// allocator == nullptr looks the policy up by name.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createRegisterAllocationPass(
    RegisterAllocationOptions options = {}, std::unique_ptr<RegisterAllocator> allocator = nullptr);

/// Driver without a PassManager, for tests.
STINKYTOFU_EXPORT Expected<AllocationResult> allocateRegisters(
    Function& function, RegisterAllocator& allocator,
    const RegisterAllocationOptions& options = {});

}  // namespace stinkytofu
