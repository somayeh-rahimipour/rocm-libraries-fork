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

// Name-to-factory map for colouring policies, mirroring BackendRegistry.
//
// registerAllAllocators() is the dead-strip guard: call it from main() so a
// static build keeps self-registering allocator TUs.

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/transforms/ra/RegisterAllocator.hpp"

namespace stinkytofu {

class STINKYTOFU_EXPORT AllocatorRegistry {
   public:
    using Factory = std::function<std::unique_ptr<RegisterAllocator>()>;

    /// Register \p factory under \p name. A second registration of the same name
    /// replaces the first, so registerAllAllocators() is idempotent.
    static void registerAllocator(std::string name, Factory factory);

    /// A new allocator for \p name, or nullptr when none is registered.
    static std::unique_ptr<RegisterAllocator> createAllocator(const std::string& name);

    /// Registered names, sorted. Adding an allocator later is one registration
    /// line; a conformance suite parameterized over this list picks it up.
    static std::vector<std::string> registeredAllocatorNames();

    /// Ensure allocator TUs are linked. Call from main() or module init.
    static void registerAllAllocators();

   private:
    AllocatorRegistry() = default;

    struct Registry;
    static Registry& getRegistry();
};

}  // namespace stinkytofu
