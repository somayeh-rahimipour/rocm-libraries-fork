// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "core/Handle.hpp"
#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include "engines/hip_mlops_engine/plans/RMSnorm/RMSnormApplicabilityChecks.hpp"
#include "engines/hip_mlops_engine/plans/RMSnorm/RMSnormBwdPlan.hpp"
#include "engines/hip_mlops_engine/plans/RMSnorm/RMSnormFwdPlan.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"

/**
 * @file RMSnormNative.cpp
 * @brief The RMSnorm engine's native half: forward and backward packs, matching,
 *        scoring, dispatch, and the one function that registers them.
 *
 * One engine, two packs -- the Pointwise add/mul shape, not the Pointwise/PointwiseSub
 * one. Forward and backward are discriminated by which node attributes union a graph
 * carries, exactly like add and mul are discriminated by an enum compare, and every
 * other fact about them is identical: both are single-node, fp32-compute,
 * runtime-compiled from the graph's own shape, with no kernel variant to key on. That
 * makes their KMD (empty) and UHD (single kernel, no ranking to do) the same
 * descriptor by construction, so one engine sharing them is the natural fit; splitting
 * into two engines would duplicate an identical schema and heuristic for no reason.
 * What differs -- which attributes union matches, which validator runs, which plan
 * class compiles and executes -- is exactly what each pack's own graph matcher and
 * dispatch handler capture, the same division PointwiseNative.cpp uses between its
 * shared applicability matcher and its per-pack operation matcher.
 *
 * Both packs delegate to the existing RMSnormFwdPlan and RMSnormBwdPlan: the
 * normalization dimension, outer/inner/stride extents, element types and launch
 * geometry are derived from the graph in those classes' compile(), and restating that
 * derivation here to fit the ingestor's shape would be a second implementation of the
 * same two kernels.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view FWD_GRAPH_MATCHER_SYMBOL = "hipkernel.rmsnorm_fwd.graph_match";
constexpr std::string_view BWD_GRAPH_MATCHER_SYMBOL = "hipkernel.rmsnorm_bwd.graph_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.rmsnorm.score";
constexpr std::string_view FWD_DISPATCH_SYMBOL = "hipkernel.rmsnorm_fwd.dispatch";
constexpr std::string_view BWD_DISPATCH_SYMBOL = "hipkernel.rmsnorm_bwd.dispatch";

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/**
 * @brief Graph-scoped applicability for the forward pack: one fp32-compute RMSnorm
 *        node over a tensor configuration the validator accepts.
 *
 * Binds no tokens: preparation rebuilds its parameters from the node, which is the one
 * place that derivation lives.
 */
bool rmsnormForwardGraphMatches(const MatchContext& context, BoundTokens& /*bound*/)
{
    // Execute-time override shapes can diverge from the dims the kernel is compiled
    // for, and those dims are baked into the launch (RFC 0008 §4.6).
    if(context.graph.getGraph().is_override_shape_enabled() || context.graph.nodeCount() != 1)
    {
        return false;
    }

    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::RMSNormAttributes
       || node.computeDataType() != data_objects::DataType::FLOAT)
    {
        return false;
    }

    // The validator is the existing engine's tensor-configuration check, reused rather
    // than restated. It reports a refusal by throwing, which a matcher must not do.
    try
    {
        rmsnorm::RMSnormValidator validator(context.graph.getTensorMap());
        validator.checkTensorConfigSupported(node.attributesAs<data_objects::RMSNormAttributes>());
    }
    catch(const std::exception& refusal)
    {
        HIPDNN_PLUGIN_LOG_INFO("rmsnorm forward declines this graph: " << refusal.what());
        return false;
    }
    return true;
}

/**
 * @brief Graph-scoped applicability for the backward pack: symmetric with the forward
 *        matcher, over the backward attributes union and validator.
 */
bool rmsnormBackwardGraphMatches(const MatchContext& context, BoundTokens& /*bound*/)
{
    if(context.graph.getGraph().is_override_shape_enabled() || context.graph.nodeCount() != 1)
    {
        return false;
    }

    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::RMSNormBackwardAttributes
       || node.computeDataType() != data_objects::DataType::FLOAT)
    {
        return false;
    }

    try
    {
        rmsnorm::RMSnormValidator validator(context.graph.getTensorMap());
        validator.checkBwdTensorConfigSupported(
            node.attributesAs<data_objects::RMSNormBackwardAttributes>());
    }
    catch(const std::exception& refusal)
    {
        HIPDNN_PLUGIN_LOG_INFO("rmsnorm backward declines this graph: " << refusal.what());
        return false;
    }
    return true;
}

/// One kernel per pack, so ranking has nothing to order. A heuristic is still
/// required, and both packs share it: neither has a choice to describe.
double rmsnormScore(const KernelDefinition& /*kernel*/, const MatchContext& /*context*/)
{
    return 0.0;
}

/**
 * @brief The full HIP properties of @p deviceId.
 *
 * The ingestor's own DeviceProperties is a deliberately narrow three-field view -- arch
 * name, warp size, CU count -- which is everything matching needs and less than a
 * kernel compile does. RMSnormFwdPlan and RMSnormBwdPlan's compile() take
 * hipDeviceProp_t, so the full record is queried here rather than widening what every
 * matcher is handed.
 *
 * @throws HipdnnPluginException if the device cannot be queried.
 */
hipDeviceProp_t fullDeviceProperties(DeviceId deviceId)
{
    hipDeviceProp_t properties{};
    if(deviceId == NO_DEVICE || hipGetDeviceProperties(&properties, deviceId) != hipSuccess)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "rmsnorm dispatch could not query properties for device " + std::to_string(deviceId));
    }
    return properties;
}

// ---------------------------------------------------------------------------
// Forward dispatch
// ---------------------------------------------------------------------------

/// Owns the compiled plan the existing engine would have built.
class PreparedRMSnormForward : public PreparedDispatch
{
public:
    explicit PreparedRMSnormForward(std::unique_ptr<rmsnorm::RMSnormFwdPlan> plan)
        : _plan(std::move(plan))
    {
    }

    const rmsnorm::RMSnormFwdPlan& plan() const
    {
        return *_plan;
    }

private:
    std::unique_ptr<rmsnorm::RMSnormFwdPlan> _plan;
};

/**
 * @brief The native dispatch behind the forward pack's UDD.
 *
 * A thin adapter onto RMSnormFwdPlan: the derivation of extents, element types and
 * launch geometry from the graph is that class's compile(), and duplicating it here to
 * fit the ingestor's shape would be two implementations of one kernel's launch.
 */
class RMSnormForwardDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    /// @param kernelCompiler Must outlive this handler; both are process-lifetime.
    explicit RMSnormForwardDispatchHandler(const compilation::IKernelCompiler& kernelCompiler)
        : _kernelCompiler(kernelCompiler)
    {
    }

    /// The kernel writes its outputs in place; RMSnormFwdPlan needs no workspace.
    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        return 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& /*bound*/,
                                              const KernelDefinition& /*kernel*/) const override
    {
        const auto& node = context.graph.getNodeWrapper(0);
        rmsnorm::RMSnormFwdParams params(node.attributesAs<data_objects::RMSNormAttributes>(),
                                         context.graph.getTensorMap());

        auto plan = std::make_unique<rmsnorm::RMSnormFwdPlan>(std::move(params));
        plan->compile(_kernelCompiler, fullDeviceProperties(context.deviceId));
        return std::make_unique<PreparedRMSnormForward>(std::move(plan));
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        dynamic_cast<const PreparedRMSnormForward&>(prepared).plan().execute(
            handle, deviceBuffers, numDeviceBuffers, workspace);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
};

/// This pack's dispatch handler. Process-lifetime: the registry holds a non-owning
/// pointer while a Container is created and destroyed per handle.
const RMSnormForwardDispatchHandler& rmsnormForwardDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const RMSnormForwardDispatchHandler s_dispatchHandler(s_kernelCompiler);
    return s_dispatchHandler;
}

// ---------------------------------------------------------------------------
// Backward dispatch
// ---------------------------------------------------------------------------

/// Owns the compiled plan the existing engine would have built.
class PreparedRMSnormBackward : public PreparedDispatch
{
public:
    explicit PreparedRMSnormBackward(std::unique_ptr<rmsnorm::RMSnormBwdPlan> plan)
        : _plan(std::move(plan))
    {
    }

    const rmsnorm::RMSnormBwdPlan& plan() const
    {
        return *_plan;
    }

private:
    std::unique_ptr<rmsnorm::RMSnormBwdPlan> _plan;
};

/**
 * @brief The native dispatch behind the backward pack's UDD.
 *
 * RMSnormBwdPlan compiles and launches two kernels -- BwdData and BwdWeightBias --
 * from one source file; the KDP names only the first as its kernel entry, because the
 * second has no variant of its own and is not a choice the catalog makes. Launching
 * both is this handler's job, exactly as RMSnormBwdPlan::execute() already does.
 */
class RMSnormBackwardDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    /// @param kernelCompiler Must outlive this handler; both are process-lifetime.
    explicit RMSnormBackwardDispatchHandler(const compilation::IKernelCompiler& kernelCompiler)
        : _kernelCompiler(kernelCompiler)
    {
    }

    /// Both kernels write their outputs in place; RMSnormBwdPlan needs no workspace.
    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        return 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& /*bound*/,
                                              const KernelDefinition& /*kernel*/) const override
    {
        const auto& node = context.graph.getNodeWrapper(0);
        rmsnorm::RMSnormBwdParams params(
            node.attributesAs<data_objects::RMSNormBackwardAttributes>(),
            context.graph.getTensorMap());

        auto plan = std::make_unique<rmsnorm::RMSnormBwdPlan>(std::move(params));
        plan->compile(_kernelCompiler, fullDeviceProperties(context.deviceId));
        return std::make_unique<PreparedRMSnormBackward>(std::move(plan));
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        dynamic_cast<const PreparedRMSnormBackward&>(prepared).plan().execute(
            handle, deviceBuffers, numDeviceBuffers, workspace);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
};

/// This pack's dispatch handler. Process-lifetime: the registry holds a non-owning
/// pointer while a Container is created and destroyed per handle.
const RMSnormBackwardDispatchHandler& rmsnormBackwardDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const RMSnormBackwardDispatchHandler s_dispatchHandler(s_kernelCompiler);
    return s_dispatchHandler;
}

} // namespace

void registerRMSnormSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(FWD_GRAPH_MATCHER_SYMBOL), &rmsnormForwardGraphMatches);
    scope.add(std::string(BWD_GRAPH_MATCHER_SYMBOL), &rmsnormBackwardGraphMatches);
    scope.add(std::string(SCORE_SYMBOL), &rmsnormScore);
    scope.add(std::string(FWD_DISPATCH_SYMBOL), &rmsnormForwardDispatchHandler());
    scope.add(std::string(BWD_DISPATCH_SYMBOL), &rmsnormBackwardDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
