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
#include "stinkytofu/transforms/ra/AllocationVerifier.hpp"

#include <sstream>
#include <string>
#include <vector>

#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"
#include "stinkytofu/transforms/ra/PhysRegMatrix.hpp"

namespace stinkytofu {
namespace {

std::string valueName(SSAValueID id) {
    return "%" + std::to_string(id);
}

std::string joinIds(const std::vector<SSAValueID>& ids) {
    std::ostringstream out;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) out << ", ";
        out << valueName(ids[i]);
    }
    return out.str();
}

}  // namespace

std::string AllocationVerificationResult::toString() const {
    std::ostringstream out;
    for (size_t i = 0; i < errors.size(); ++i) {
        if (i > 0) out << '\n';
        out << errors[i];
    }
    return out.str();
}

AllocationVerificationResult verifyAllocation(const Function& function,
                                              const AllocationResult& result,
                                              const AllocationContext& context) {
    AllocationVerificationResult checked;
    auto error = [&](std::string message) { checked.errors.push_back(std::move(message)); };

    const std::string prefix = "@" + function.getName() + ": ";

    if (&context.function != &function) {
        error(prefix + "verification context describes a different function");
        return checked;
    }

    const uint64_t attachedShape = function.ssaArena().shape();
    if (attachedShape != kUnstampedShape && computeFunctionShape(function) != attachedShape) {
        error(prefix +
              "the function changed after it was lifted, so the attached SSA describes a "
              "different program");
    }
    if (result.shape() != kUnstampedShape && attachedShape != kUnstampedShape &&
        result.shape() != attachedShape) {
        error(prefix +
              "the allocation was computed against a different graph, so its SSA value IDs do "
              "not mean the same thing here");
    }

    PhysRegMatrix matrix(context.target);
    const AllocationConstraints& constraints = context.constraints;

    for (StinkySSAValue* value : function.ssaArena().values()) {
        if (value == nullptr) continue;
        const SSAValueID id = value->valueId();
        const std::string where = prefix + valueName(id);

        if (!result.isAssigned(id)) {
            error(where + " has no physical register");
            continue;
        }

        const RegKey physical = result.assignmentOf(id);
        if (physical.half != RegHalf::NONE) {
            error(where + " is assigned the sub-DWORD register " + regKeyToString(physical) +
                  ", which cannot be written back to a full-DWORD operand");
            continue;
        }

        const RegType expectedClass = constraints.classOf(id);
        if (expectedClass != RegType::UNKNOWN && physical.type != expectedClass) {
            error(where + " is " + regKeyToString(physical) + " but the value is class " +
                  regTypeToString(expectedClass));
            continue;
        }

        if (!context.target.isAllocatable(physical.type, physical.idx)) {
            if (context.target.isReserved(physical.type, physical.idx)) {
                error(where + " is assigned reserved " + regKeyToString(physical));
            } else {
                error(where + " is assigned " + regKeyToString(physical) +
                      ", which is not allocatable");
            }
            continue;
        }

        const LiveRange& range = context.intervals.rangeOf(id);
        if (!matrix.available(physical.type, physical.idx, range)) {
            std::vector<SSAValueID> conflicts;
            matrix.collectConflicts(physical.type, physical.idx, range, conflicts);
            std::ostringstream out;
            out << where << " overlaps";
            if (conflicts.empty()) {
                out << " another value";
            } else {
                out << ' ' << joinIds(conflicts);
            }
            out << " on " << regKeyToString(physical);
            error(out.str());
            continue;
        }
        matrix.bind(physical.type, physical.idx, id, range);
    }

    for (const TupleRun& run : constraints.tupleRuns()) {
        if (run.units.empty()) continue;
        const SSAValueID firstId = run.units.front();
        if (!result.isAssigned(firstId)) continue;
        const RegKey first = result.assignmentOf(firstId);
        for (size_t unit = 1; unit < run.units.size(); ++unit) {
            const SSAValueID id = run.units[unit];
            if (!result.isAssigned(id)) continue;
            const RegKey physical = result.assignmentOf(id);
            if (physical.type == first.type && physical.idx == first.idx + unit) continue;
            error(prefix + "tuple [" + joinIds(run.units) + "]: unit 0 is " +
                  regKeyToString(first) + " but unit " + std::to_string(unit) + " is " +
                  regKeyToString(physical) +
                  "; a range operand must be consecutive in operand "
                  "order");
            break;
        }
    }

    for (const AffinitySet& set : constraints.affinitySets()) {
        if (set.members.empty()) continue;
        const SSAValueID firstId = set.members.front();
        if (!result.isAssigned(firstId)) continue;
        const RegKey first = result.assignmentOf(firstId);
        for (size_t i = 1; i < set.members.size(); ++i) {
            const SSAValueID id = set.members[i];
            if (!result.isAssigned(id)) continue;
            const RegKey physical = result.assignmentOf(id);
            if (physical == first) continue;
            error(prefix + "affinity {" + joinIds(set.members) + "}: " + valueName(id) + " is " +
                  regKeyToString(physical) + " but " + valueName(firstId) + " is " +
                  regKeyToString(first) +
                  "; a merge and its incoming values must share one colour");
            break;
        }
    }

    return checked;
}

}  // namespace stinkytofu
