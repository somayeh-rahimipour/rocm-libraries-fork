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
#include "stinkytofu/transforms/ra/AllocatorRegistry.hpp"

#include <algorithm>
#include <unordered_map>
#include <utility>

namespace stinkytofu {

void anchorLegacyIdentityAllocator();

struct AllocatorRegistry::Registry {
    std::unordered_map<std::string, Factory> factories;
};

AllocatorRegistry::Registry& AllocatorRegistry::getRegistry() {
    static Registry registry;
    return registry;
}

void AllocatorRegistry::registerAllocator(std::string name, Factory factory) {
    getRegistry().factories[std::move(name)] = std::move(factory);
}

std::unique_ptr<RegisterAllocator> AllocatorRegistry::createAllocator(const std::string& name) {
    registerAllAllocators();
    auto& factories = getRegistry().factories;
    auto it = factories.find(name);
    if (it == factories.end() || !it->second) return nullptr;
    return it->second();
}

std::vector<std::string> AllocatorRegistry::registeredAllocatorNames() {
    registerAllAllocators();
    std::vector<std::string> names;
    names.reserve(getRegistry().factories.size());
    for (const auto& kv : getRegistry().factories) names.push_back(kv.first);
    std::sort(names.begin(), names.end());
    return names;
}

void AllocatorRegistry::registerAllAllocators() {
    anchorLegacyIdentityAllocator();
}

}  // namespace stinkytofu
