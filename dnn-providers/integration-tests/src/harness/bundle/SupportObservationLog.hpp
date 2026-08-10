// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <mutex>
#include <string>
#include <vector>

namespace hipdnn_integration_tests::bundle
{

struct SupportObservation
{
    std::string diagnosticPath; // "dir/Small.json" or "dir/sweep.json#caseId"
    std::string engineName;
    std::string arch;
    std::string platform;
    bool engineIsSupported = false; // resolved + in ranked list
};

// Process-wide log of resolved support observations. Populated during
// --write-support-claims runs and drained once after RUN_ALL_TESTS() to
// produce .support.json sidecars. Only resolved queries (OK or
// GRAPH_NOT_SUPPORTED) are recorded; unresolved queries are not
// observations of "unsupported" and must never null an existing claim.
class SupportObservationLog
{
public:
    static SupportObservationLog& get()
    {
        static SupportObservationLog s_instance;
        return s_instance;
    }

    SupportObservationLog(const SupportObservationLog&) = delete;
    SupportObservationLog& operator=(const SupportObservationLog&) = delete;
    SupportObservationLog(SupportObservationLog&&) = delete;
    SupportObservationLog& operator=(SupportObservationLog&&) = delete;

    void record(SupportObservation observation)
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        _observations.push_back(std::move(observation));
    }

    std::vector<SupportObservation> all() const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        return _observations;
    }

    bool empty() const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        return _observations.empty();
    }

    void reset()
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        _observations.clear();
    }

private:
    SupportObservationLog() = default;

    mutable std::mutex _mutex;
    std::vector<SupportObservation> _observations;
};

} // namespace hipdnn_integration_tests::bundle
