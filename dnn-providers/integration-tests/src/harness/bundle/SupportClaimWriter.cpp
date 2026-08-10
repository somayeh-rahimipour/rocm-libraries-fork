// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/SupportClaimWriter.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "harness/bundle/SupportClaims.hpp"

namespace hipdnn_integration_tests::bundle
{

namespace
{

// diagnostic path -> sidecar path, isSweep
// "dir/Small.json" → {"dir/Small.support.json", false}
// "dir/sweep.json#caseId" → {"dir/support.json", true}
struct SidecarTarget
{
    std::filesystem::path sidecarPath;
    bool isSweep = false;
};

SidecarTarget resolveSidecarTarget(const std::string& diagnosticPath)
{
    const auto hashPos = diagnosticPath.find('#');
    if(hashPos != std::string::npos)
    {
        const auto sweepJsonPath = std::filesystem::path(diagnosticPath.substr(0, hashPos));
        return {sweepJsonPath.parent_path() / "support.json", true};
    }
    return {supportJsonPath(std::filesystem::path(diagnosticPath)), false};
}

std::string extractCaseId(const std::string& diagnosticPath)
{
    const auto hashPos = diagnosticPath.find('#');
    if(hashPos == std::string::npos)
    {
        return {};
    }
    return diagnosticPath.substr(hashPos + 1);
}

void overlaySingleGraphCell(SupportClaims& existing,
                            const std::string& engineName,
                            const std::string& arch,
                            const std::string& platform,
                            bool engineIsSupported)
{
    if(engineIsSupported)
    {
        existing.claims[engineName][arch].insert(platform);
    }
    else
    {
        auto engineIt = existing.claims.find(engineName);
        if(engineIt == existing.claims.end())
        {
            return;
        }
        auto archIt = engineIt->second.find(arch);
        if(archIt == engineIt->second.end())
        {
            return;
        }
        archIt->second.erase(platform);
        if(archIt->second.empty())
        {
            engineIt->second.erase(archIt);
        }
        if(engineIt->second.empty())
        {
            existing.claims.erase(engineIt);
        }
    }
}

// Per-case support: engine -> caseId -> ArchPlatformMap
using FlatSweepMap = std::map<std::string, std::map<std::string, ArchPlatformMap>>;

FlatSweepMap flattenSweepClaims(const SweepSupportClaims& existing)
{
    FlatSweepMap flat;
    for(const auto& [engine, groups] : existing.claims)
    {
        for(const auto& group : groups)
        {
            for(const auto& caseId : group.cases)
            {
                flat[engine][caseId] = group.support;
            }
        }
    }
    return flat;
}

SweepSupportClaims regroupSweepClaims(const FlatSweepMap& flat, int version)
{
    SweepSupportClaims result;
    result.version = version;

    for(const auto& [engine, caseMap] : flat)
    {
        // Bucket cases by identical support footprint.
        // Use the canonical JSON string of the support map as the grouping key.
        std::map<std::string, std::vector<std::string>> footprintToCases;
        std::map<std::string, ArchPlatformMap> footprintToSupport;

        for(const auto& [caseId, supportMap] : caseMap)
        {
            if(supportMap.empty())
            {
                continue;
            }
            const auto key = archPlatformMapToJson(supportMap).dump();
            footprintToCases[key].push_back(caseId);
            footprintToSupport[key] = supportMap;
        }

        std::vector<SweepClaimGroup> groups;
        for(auto& [key, cases] : footprintToCases)
        {
            std::sort(cases.begin(), cases.end());
            SweepClaimGroup group;
            group.cases = std::move(cases);
            group.support = footprintToSupport.at(key);
            groups.push_back(std::move(group));
        }

        // Order groups by their first case id.
        std::sort(
            groups.begin(), groups.end(), [](const SweepClaimGroup& a, const SweepClaimGroup& b) {
                return a.cases.front() < b.cases.front();
            });

        if(!groups.empty())
        {
            result.claims[engine] = std::move(groups);
        }
    }

    return result;
}

bool writeIfChanged(const std::filesystem::path& filePath,
                    const std::string& newContent,
                    WriteSummary& summary)
{
    if(std::filesystem::exists(filePath))
    {
        std::ifstream existingFile(filePath);
        if(existingFile)
        {
            std::string existingContent((std::istreambuf_iterator<char>(existingFile)),
                                        std::istreambuf_iterator<char>());
            if(existingContent == newContent)
            {
                ++summary.filesUnchanged;
                return true;
            }
        }
    }

    std::ofstream outputFile(filePath);
    if(!outputFile)
    {
        summary.errors.push_back("could not open for writing: " + filePath.string());
        return false;
    }
    outputFile << newContent;
    if(!outputFile)
    {
        summary.errors.push_back("write failed: " + filePath.string());
        return false;
    }
    ++summary.filesWritten;
    return true;
}

} // namespace

WriteSummary writeObservedSupportClaims(const std::vector<SupportObservation>& observations)
{
    WriteSummary summary;

    // Group observations by sidecar file.
    struct PerFileObservation
    {
        std::string caseId; // empty for single-graph
        std::string engineName;
        std::string arch;
        std::string platform;
        bool engineIsSupported;
    };

    std::map<std::filesystem::path, std::vector<PerFileObservation>> observationsByFile;
    std::map<std::filesystem::path, bool> fileIsSweep;

    for(const auto& observation : observations)
    {
        const auto target = resolveSidecarTarget(observation.diagnosticPath);
        const auto caseId = extractCaseId(observation.diagnosticPath);

        observationsByFile[target.sidecarPath].push_back({caseId,
                                                          observation.engineName,
                                                          observation.arch,
                                                          observation.platform,
                                                          observation.engineIsSupported});
        fileIsSweep[target.sidecarPath] = target.isSweep;
    }

    for(const auto& [sidecarPath, fileObservations] : observationsByFile)
    {
        if(fileObservations.empty())
        {
            ++summary.targetsSkipped;
            continue;
        }

        const bool isSweep = fileIsSweep.at(sidecarPath);

        if(isSweep)
        {
            SweepSupportClaims existing;
            existing.version = 1;
            if(std::filesystem::exists(sidecarPath))
            {
                auto loaded = loadSweepSupportClaims(sidecarPath.parent_path());
                if(loaded.has_value())
                {
                    existing = std::move(*loaded);
                }
            }

            auto flat = flattenSweepClaims(existing);

            for(const auto& obs : fileObservations)
            {
                if(obs.engineIsSupported)
                {
                    flat[obs.engineName][obs.caseId][obs.arch].insert(obs.platform);
                }
                else
                {
                    auto engineIt = flat.find(obs.engineName);
                    if(engineIt == flat.end())
                    {
                        continue;
                    }
                    auto caseIt = engineIt->second.find(obs.caseId);
                    if(caseIt == engineIt->second.end())
                    {
                        continue;
                    }
                    auto archIt = caseIt->second.find(obs.arch);
                    if(archIt == caseIt->second.end())
                    {
                        continue;
                    }
                    archIt->second.erase(obs.platform);
                    if(archIt->second.empty())
                    {
                        caseIt->second.erase(archIt);
                    }
                    if(caseIt->second.empty())
                    {
                        engineIt->second.erase(caseIt);
                    }
                    if(engineIt->second.empty())
                    {
                        flat.erase(engineIt);
                    }
                }
            }

            const auto regrouped = regroupSweepClaims(flat, existing.version);
            const auto jsonContent = dumpCanonical(toJson(regrouped));
            writeIfChanged(sidecarPath, jsonContent, summary);
        }
        else
        {
            SupportClaims existing;
            existing.version = 1;
            if(std::filesystem::exists(sidecarPath))
            {
                std::ifstream existingFile(sidecarPath);
                if(existingFile)
                {
                    auto json
                        = nlohmann::json::parse(existingFile, nullptr, /*allow_exceptions=*/false);
                    if(!json.is_discarded())
                    {
                        existing = parseSupportClaimsJson(json, sidecarPath.string());
                    }
                }
            }

            for(const auto& obs : fileObservations)
            {
                overlaySingleGraphCell(
                    existing, obs.engineName, obs.arch, obs.platform, obs.engineIsSupported);
            }

            const auto jsonContent = dumpCanonical(toJson(existing));
            writeIfChanged(sidecarPath, jsonContent, summary);
        }
    }

    return summary;
}

} // namespace hipdnn_integration_tests::bundle
