// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <array>
#include <set>
#include <string>

#include <gtest/gtest.h>

#include "core/Container.hpp"
#include "core/Handle.hpp"
#include <hip_kernel_provider_common/HipDeviceUtils.hpp>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/PluginVersionConstants.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"
#endif

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::core;

/// Engines the provider exposes: one per discovered descriptor set, and nothing else.
///
/// Read from the inventory rather than hardcoded. A literal count goes wrong the moment
/// a pack ships, and it is the only thing standing between a dead-stripped pack table
/// and a green run.
static uint32_t expectedEngines()
{
#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
    return static_cast<uint32_t>(
        hip_kernel_provider::kernel_ingestor_engine::discoverDescriptorSets().size());
#else
    return 0;
#endif
}

/// Upper bound for the fixed-size buffers below. Generous rather than exact: a literal
/// that merely happens to fit today silently truncates the copy the moment an engine is
/// added, and a truncated copy fails as a count mismatch that names neither cause.
constexpr uint32_t MAX_EXPECTED_ENGINES = 64;

TEST(TestContainer, ConstructsSuccessfully)
{
    const Container container;
}

TEST(TestContainer, CopyEngineIdsReturnsExpectedEngineCount)
{
    uint32_t numEngines = 0;
    auto totalEngines = Container::copyEngineIds(nullptr, 0, numEngines);

    EXPECT_EQ(totalEngines, expectedEngines());
    EXPECT_EQ(numEngines, expectedEngines());
}

TEST(TestContainer, CopyEngineIdsWithBufferContainsEveryDescriptorEngine)
{
#ifndef HIPDNN_ENABLE_KERNEL_INGESTOR
    GTEST_SKIP();
#else
    std::array<int64_t, MAX_EXPECTED_ENGINES> engineIds = {};
    uint32_t numEngines = 0;
    auto totalEngines
        = Container::copyEngineIds(engineIds.data(), MAX_EXPECTED_ENGINES, numEngines);

    EXPECT_EQ(totalEngines, expectedEngines());
    EXPECT_EQ(numEngines, expectedEngines());

    // Advertising an id the constructor then fails to build is the failure this
    // guards: every id copied out must be one a discovered set claims.
    for(const auto& set : hip_kernel_provider::kernel_ingestor_engine::discoverDescriptorSets())
    {
        const auto id = hipdnn_data_sdk::utilities::engineNameToId(set.engine.name);
        EXPECT_NE(std::find(engineIds.begin(), engineIds.begin() + numEngines, id),
                  engineIds.begin() + numEngines)
            << set.engine.name;
    }
#endif
}

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
TEST(TestContainer, ExposesAnEngineForEveryDiscoveredDescriptorSet)
{
    using namespace hip_kernel_provider::kernel_ingestor_engine;

    // Names the ids rather than counting them. A count cannot tell a missing ingestor
    // engine from an extra native one, and it cannot see the failure this is really
    // guarding: the pack table being dropped from a binary that links the provider as a
    // static archive, which leaves the engine absent and every other assertion happy.
    const auto& sets = discoverDescriptorSets();

    // Named rather than counted: with an empty result the loop below is vacuous and
    // every count assertion in this file still passes, and a count cannot tell a missing
    // engine from a renamed one. Reachable, since a pack that fails symbol registration
    // is excluded from exactly this list.
    std::vector<std::string> names;
    names.reserve(sets.size());
    for(const auto& set : sets)
    {
        names.push_back(set.engine.name);
    }
    std::sort(names.begin(), names.end());
    EXPECT_EQ(names,
              (std::vector<std::string>{"hipkernel:AsmSdpaBackward",
                                        "hipkernel:AsmSdpaForward",
                                        "hipkernel:Batchnorm",
                                        "hipkernel:ConvFwd",
                                        "hipkernel:LayernormForward",
                                        "hipkernel:Pointwise",
                                        "hipkernel:RMSnorm",
                                        "hipkernel:Resample"}));

    Container container;
    const auto allEngineIds = container.getEngineManager().getAllEngineIds();

    for(const auto& set : sets)
    {
        const auto engineId = hipdnn_data_sdk::utilities::engineNameToId(set.engine.name);
        EXPECT_NE(std::find(allEngineIds.begin(), allEngineIds.end(), engineId), allEngineIds.end())
            << "no engine for descriptor set '" << set.engine.name << "'";
    }
}

TEST(TestContainer, EveryDescriptorEngineDeclaresTheSchemaItsGraphsRequire)
{
    using namespace hip_kernel_provider::kernel_ingestor_engine;

    // A UED that omits sdk_version reads as the 1.0.0 baseline, and GenericPlanBuilder
    // then declines any graph whose own floor is higher -- before a matcher runs, so the
    // engine simply never appears for that graph. An engine whose op takes a scalar
    // operand (epsilon, momentum, attention scale) can be handed that scalar as a runtime
    // pass-by-value tensor, which is a 1.2.0 feature (RFC 0016). The builders these
    // engines replaced had no floor at all and served those graphs.
    //
    // Asserted here because nothing else would: the E2E suites pass their scalars as
    // compile-time constants, so a silent regression to baseline stays green everywhere
    // except IntegrationGpuPassByValue.
    const hipdnn_data_sdk::utilities::Version passByValueFloor{
        hipdnn_plugin_sdk::K_PASS_BY_VALUE_MIN_API_VERSION};

    // Named rather than derived: whether an op takes a scalar operand is a property of
    // the operation, and the descriptors do not model it. Pointwise and PointwiseSub are
    // absent because their graphs carry no scalar to pass.
    const std::set<std::string> takesAScalarOperand{"hipkernel:LayernormForward",
                                                    "hipkernel:RMSnorm",
                                                    "hipkernel:Batchnorm",
                                                    "hipkernel:AsmSdpaForward",
                                                    "hipkernel:AsmSdpaBackward"};

    const auto& sets = discoverDescriptorSets();
    ASSERT_FALSE(sets.empty()) << "no descriptor sets discovered, so nothing was asserted";

    size_t checked = 0;
    for(const auto& set : sets)
    {
        if(takesAScalarOperand.count(set.engine.name) == 0)
        {
            continue;
        }
        ++checked;
        EXPECT_FALSE(set.engine.sdkVersion < passByValueFloor)
            << "engine '" << set.engine.name << "' declares graph schema "
            << set.engine.sdkVersion.str() << ", below the " << passByValueFloor.str()
            << " a runtime pass-by-value scalar requires";
    }

    // A renamed engine would otherwise silently drop out of the set above.
    EXPECT_EQ(checked, takesAScalarOperand.size())
        << "an engine named here was not discovered; the list and the descriptors disagree";
}
#endif

TEST(TestContainer, GetEngineManagerReturnsValidReference)
{
    Container container;
    auto& engineManager = container.getEngineManager();

    (void)engineManager;
}

TEST(TestContainer, GetApplicableEngineIdsSdpaGraph)
{
    SKIP_IF_NO_DEVICES();
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    Handle handle;
    auto deviceString = hip_kernel_provider_common::getDeviceString(handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }
    Container container;
    auto& engineManager = container.getEngineManager();

    const std::vector<int64_t> dims{4, 8, 256, 128};
    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    auto graph = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(dims,
                                                                     strides,
                                                                     dims,
                                                                     strides,
                                                                     dims,
                                                                     strides,
                                                                     dims,
                                                                     strides,
                                                                     DataType::BFLOAT16,
                                                                     DataType::FLOAT);
    auto graphBuffer = graph.Release();

    auto graphWrapper = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuffer.data(), graphBuffer.size());

    auto applicableEngines = engineManager.getApplicableEngineIds(handle, graphWrapper);

#ifdef HIPDNN_ENGINE_ASM_SDPA
    // The forward SDPA graph is served by the descriptor-backed engine now, so the
    // claim is about the id its UED name hashes to, not a compiled-in constant.
    ASSERT_EQ(applicableEngines.size(), 1);
    EXPECT_EQ(applicableEngines.front(),
              hipdnn_data_sdk::utilities::engineNameToId("hipkernel:AsmSdpaForward"));
#else
    EXPECT_TRUE(applicableEngines.empty());
#endif
}

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
TEST(TestContainer, GetApplicableEngineIdsPointwiseAddGraph)
{
    // Applicability is device-resolved: with no device, matchers decline.
    SKIP_IF_NO_DEVICES();

    using namespace hip_kernel_provider::kernel_ingestor_engine;
    using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

    Handle handle;
    Container container;
    auto& engineManager = container.getEngineManager();

    const auto graph = buildPointwiseGraph();
    const auto graphWrapper = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graph.GetBufferPointer(), graph.GetSize());

    auto applicableEngines = engineManager.getApplicableEngineIds(handle, graphWrapper);

    EXPECT_NE(std::find(applicableEngines.begin(),
                        applicableEngines.end(),
                        hipdnn_data_sdk::utilities::engineNameToId(POINTWISE_ADD.engineName)),
              applicableEngines.end());
}
#endif

TEST(TestContainer, GetAllEngineIds)
{
    Container container;
    auto& engineManager = container.getEngineManager();

    auto allEngines = engineManager.getAllEngineIds();

    ASSERT_EQ(allEngines.size(), expectedEngines());
}
