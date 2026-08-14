// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <exception>
#include <memory>
#include <string>
#include <string_view>
#include <variant>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_inference_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_inference_attributes_variance_ext_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "core/Handle.hpp"
#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include "engines/hip_mlops_engine/plans/batchnorm/BatchnormApplicabilityChecks.hpp"
#include "engines/hip_mlops_engine/plans/batchnorm/BatchnormBwdPlan.hpp"
#include "engines/hip_mlops_engine/plans/batchnorm/BatchnormFwdInferencePlan.hpp"
#include "engines/hip_mlops_engine/plans/batchnorm/BatchnormFwdInferenceWithVariancePlan.hpp"
#include "engines/hip_mlops_engine/plans/batchnorm/BatchnormFwdTrainingPlan.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"

/**
 * @file BatchnormNative.cpp
 * @brief The batchnorm engine's native half, against eight packs replacing the two old
 *        builders' node-count-and-attribute-type fan-out.
 *
 * BatchnormPlanBuilder::buildPlan() and BatchnormFwdTrainingPlanBuilder::buildPlan()
 * together dispatched on seven shapes: fwd inference, fwd inference with variance, fwd
 * training, backward, and three fusions of those with a pointwise activation node (two
 * 2-node, one 3-node). One engine per shape would have meant seven engines competing
 * over the same graphs at plan-build time, so this stays what it always was
 * structurally: one engine,
 * seven variants, now expressed as one pack per variant plus a shared graph-shape
 * matcher every pack lists (RFC 0017 section 6's Pointwise add/mul shape, widened from
 * two packs to seven). The eighth pack, fwd training + activation, is
 * BatchnormFwdTrainingPlanBuilder's own second shape and folds in as a pack the same
 * way its solo-training sibling does.
 *
 * What a KMD field could not carry stays here, unchanged in substance from the old
 * builders:
 * - **Applicability.** Every tensor-configuration check is the existing
 *   BatchnormValidator's, invoked with the same attribute types the old builders used.
 *   The tensor-wiring checks that precede each fusion's validator call (does the
 *   activation chain from the right tensor, is virtuality where the fusion expects it)
 *   were private to BatchnormPlanBuilder.cpp and BatchnormFwdTrainingPlanBuilder.cpp,
 *   so no header exposed them for reuse; they are restated here rather than copied
 *   from an accessible source, but check the identical fields in the identical order.
 * - **Which plan.** Each pack's dispatch handler builds the one Params/Plan pair that
 *   variant always built; nothing about derivation moved.
 *
 * All eight share one KMD (a single `variant` string field), one UHD (nothing to rank
 * within a pack of one kernel), and one UDD (one dispatch handler parameterised by
 * variant, since prepare()/launch() differ only in which Params/Plan pair they build,
 * not in dispatch ABI).
 *
 * The symbol names below are restated rather than shared through a header, because a
 * descriptor file cannot export a constant to C++. The loader pre-flights every symbol
 * a descriptor names, so a typo costs the engine at load rather than at dispatch.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hip_kernel_provider::batchnorm::BatchnormBwdParams;
using hip_kernel_provider::batchnorm::BatchnormBwdPlan;
using hip_kernel_provider::batchnorm::BatchnormFwdInferenceParams;
using hip_kernel_provider::batchnorm::BatchnormFwdInferencePlan;
using hip_kernel_provider::batchnorm::BatchnormFwdInferenceWithVarianceParams;
using hip_kernel_provider::batchnorm::BatchnormFwdInferenceWithVariancePlan;
using hip_kernel_provider::batchnorm::BatchnormFwdTrainingParams;
using hip_kernel_provider::batchnorm::BatchnormFwdTrainingPlan;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_SHAPE_MATCHER_SYMBOL = "hipkernel.batchnorm.graph_shape_match";
constexpr std::string_view FWD_INFERENCE_MATCHER_SYMBOL = "hipkernel.batchnorm.fwd_inference_match";
constexpr std::string_view FWD_INFERENCE_WITH_VARIANCE_MATCHER_SYMBOL
    = "hipkernel.batchnorm.fwd_inference_with_variance_match";
constexpr std::string_view FWD_TRAINING_MATCHER_SYMBOL = "hipkernel.batchnorm.fwd_training_match";
constexpr std::string_view BWD_MATCHER_SYMBOL = "hipkernel.batchnorm.bwd_match";
constexpr std::string_view FWD_INFERENCE_ACTIVATION_MATCHER_SYMBOL
    = "hipkernel.batchnorm.fwd_inference_activation_match";
constexpr std::string_view FWD_INFERENCE_WITH_VARIANCE_ACTIVATION_MATCHER_SYMBOL
    = "hipkernel.batchnorm.fwd_inference_with_variance_activation_match";
constexpr std::string_view FWD_TRAINING_ACTIVATION_MATCHER_SYMBOL
    = "hipkernel.batchnorm.fwd_training_activation_match";
constexpr std::string_view BWD_ACTIVATION_MATCHER_SYMBOL
    = "hipkernel.batchnorm.bwd_activation_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.batchnorm.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.batchnorm.dispatch";

// The KMD's one field: which of the eight variants a matched kernel implements.
constexpr std::string_view VARIANT_FIELD = "variant";

/// One entry per pack; matches each pack's `variant` metadata value and its KDP name.
enum class Variant
{
    FWD_INFERENCE,
    FWD_INFERENCE_WITH_VARIANCE,
    FWD_TRAINING,
    BWD,
    FWD_INFERENCE_ACTIVATION,
    FWD_INFERENCE_WITH_VARIANCE_ACTIVATION,
    FWD_TRAINING_ACTIVATION,
    BWD_ACTIVATION,
};

std::string variantMetadataValue(Variant variant)
{
    switch(variant)
    {
    case Variant::FWD_INFERENCE:
        return "fwd_inference";
    case Variant::FWD_INFERENCE_WITH_VARIANCE:
        return "fwd_inference_with_variance";
    case Variant::FWD_TRAINING:
        return "fwd_training";
    case Variant::BWD:
        return "bwd";
    case Variant::FWD_INFERENCE_ACTIVATION:
        return "fwd_inference_activation";
    case Variant::FWD_INFERENCE_WITH_VARIANCE_ACTIVATION:
        return "fwd_inference_with_variance_activation";
    case Variant::FWD_TRAINING_ACTIVATION:
        return "fwd_training_activation";
    case Variant::BWD_ACTIVATION:
        return "bwd_activation";
    default:
        // Unreachable: every enumerator is handled above.
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "unhandled batchnorm variant");
    }
}

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/**
 * @brief Graph-scoped applicability, shared by every pack: no override shapes and
 *        every node computing in fp32.
 *
 * The whole of what the old builders' node-count switch checked before looking at
 * attribute types. Every pack lists this, so it runs once per (graph, device) and the
 * remaining seven packs reuse the memoized verdict; each pack's own matcher then checks
 * only node count and attribute types, exactly like Pointwise's shared single-element
 * check versus its per-pack operation check.
 */
bool batchnormGraphShapeMatches(const MatchContext& context, BoundTokens& /*bound*/)
{
    // Execute-time override shapes can diverge from the compile-time dims a plan bakes
    // into its kernel launch, so decline rather than risk a mismatch (RFC 0008 §4.6).
    if(context.graph.getGraph().is_override_shape_enabled())
    {
        return false;
    }

    if(context.graph.nodeCount() == 0)
    {
        return false;
    }

    for(const auto& node : context.graph.nodeWrappers())
    {
        if(node->computeDataType() != data_objects::DataType::FLOAT)
        {
            return false;
        }
    }
    return true;
}

/// True when node 0 is a pointwise-activation-fused two-node batchnorm inference
/// graph's first node type, and node 1 is a pointwise node -- the shape check that
/// precedes running either fusion's tensor-wiring and validator checks.
bool isTwoNodeInferenceFusionShape(const MatchContext& context,
                                   data_objects::NodeAttributes inferenceAttributesType)
{
    if(context.graph.nodeCount() != 2)
    {
        return false;
    }
    const auto& node0 = context.graph.getNodeWrapper(0);
    const auto& node1 = context.graph.getNodeWrapper(1);
    return node0.attributesType() == inferenceAttributesType
           && node1.attributesType() == data_objects::NodeAttributes::PointwiseAttributes;
}

/// Single-node fwd-inference discriminator: attribute type plus the existing
/// validator's tensor-configuration check.
bool batchnormFwdInferenceMatches(const MatchContext& context, BoundTokens& /*bound*/)
{
    if(context.graph.nodeCount() != 1)
    {
        return false;
    }
    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::BatchnormInferenceAttributes)
    {
        return false;
    }

    try
    {
        BatchnormValidator validator(context.graph.getTensorMap());
        validator.checkInferenceTensorConfigSupported(
            node.attributesAs<data_objects::BatchnormInferenceAttributes>());
    }
    catch(const std::exception& refusal)
    {
        HIPDNN_PLUGIN_LOG_INFO("batchnorm fwd inference declines this graph: " << refusal.what());
        return false;
    }
    return true;
}

/// Single-node fwd-inference-with-variance discriminator, mirroring the plain
/// fwd-inference matcher against the variance-ext attribute type.
bool batchnormFwdInferenceWithVarianceMatches(const MatchContext& context, BoundTokens& /*bound*/)
{
    if(context.graph.nodeCount() != 1)
    {
        return false;
    }
    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType()
       != data_objects::NodeAttributes::BatchnormInferenceAttributesVarianceExt)
    {
        return false;
    }

    try
    {
        BatchnormValidator validator(context.graph.getTensorMap());
        validator.checkInferenceVarianceExtTensorConfigSupported(
            node.attributesAs<data_objects::BatchnormInferenceAttributesVarianceExt>());
    }
    catch(const std::exception& refusal)
    {
        HIPDNN_PLUGIN_LOG_INFO(
            "batchnorm fwd inference with variance declines this graph: " << refusal.what());
        return false;
    }
    return true;
}

/// True when a pointwise node is a plain ReLU: RELU_FWD with no leaky slope. This is
/// BatchnormFwdTrainingPlanBuilder's own fusion-shape gate (distinct from the
/// inference fusions' activation-mode check, which lives on BatchnormValidator and
/// also accepts CLAMP), restated because it was private to that builder.
bool isSupportedFwdTrainingActivation(const data_objects::PointwiseAttributes& activation)
{
    if(activation.operation() != data_objects::PointwiseMode::RELU_FWD)
    {
        return false;
    }
    return !activation.relu_lower_clip_slope();
}

/**
 * @brief Tensor wiring for the two-node fwd-inference + activation fusion: the
 *        activation's in_0 must chain from the inference's y, and virtuality on both
 *        sides must match a fused intermediate (y and activation input virtual,
 *        everything else real).
 *
 * Restated from the old BatchnormPlanBuilder's private helper of the same shape,
 * which no header exposed for reuse; the tensor-configuration checks it precedes stay
 * on BatchnormValidator and are called, not restated.
 */
void checkFwdInferenceActivationTensorWiring(
    const data_objects::BatchnormInferenceAttributes& bnInfAttr,
    const data_objects::PointwiseAttributes& actAttr,
    const std::unordered_map<int64_t, const data_objects::TensorAttributes*>& tensorMap)
{
    if(actAttr.in_0_tensor_uid() != bnInfAttr.y_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation in_0 must be the batchnorm inference output tensor (y)");
    }

    const auto& bnInfTensorX
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.x_tensor_uid());
    const auto& bnInfTensorMean
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.mean_tensor_uid());
    const auto& bnInfTensorInvVar
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.inv_variance_tensor_uid());
    const auto& bnInfTensorScale
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.scale_tensor_uid());
    const auto& bnInfTensorBias
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.bias_tensor_uid());
    const auto& bnInfTensorY
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.y_tensor_uid());
    if(bnInfTensorX.virtual_() || bnInfTensorMean.virtual_() || bnInfTensorInvVar.virtual_()
       || bnInfTensorScale.virtual_() || bnInfTensorBias.virtual_() || !bnInfTensorY.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm inference input tensors must be non-virtual, output tensor must be virtual");
    }

    const auto& actTensorIn0
        = core::utils::findTensorAttributes(tensorMap, actAttr.in_0_tensor_uid());
    const auto& actTensorOut
        = core::utils::findTensorAttributes(tensorMap, actAttr.out_0_tensor_uid());
    if(!actTensorIn0.virtual_() || actTensorOut.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation input from batchnorm must be virtual, output must be non virtual");
    }
}

/// Same wiring as checkFwdInferenceActivationTensorWiring(), against the
/// variance-ext inference attributes' `variance` field in place of `inv_variance`.
void checkFwdInferenceWithVarianceActivationTensorWiring(
    const data_objects::BatchnormInferenceAttributesVarianceExt& bnInfAttr,
    const data_objects::PointwiseAttributes& actAttr,
    const std::unordered_map<int64_t, const data_objects::TensorAttributes*>& tensorMap)
{
    if(actAttr.in_0_tensor_uid() != bnInfAttr.y_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation in_0 must be the batchnorm inference output tensor (y)");
    }

    const auto& bnInfTensorX
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.x_tensor_uid());
    const auto& bnInfTensorMean
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.mean_tensor_uid());
    const auto& bnInfTensorVariance
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.variance_tensor_uid());
    const auto& bnInfTensorScale
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.scale_tensor_uid());
    const auto& bnInfTensorBias
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.bias_tensor_uid());
    const auto& bnInfTensorY
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.y_tensor_uid());
    if(bnInfTensorX.virtual_() || bnInfTensorMean.virtual_() || bnInfTensorVariance.virtual_()
       || bnInfTensorScale.virtual_() || bnInfTensorBias.virtual_() || !bnInfTensorY.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm inference input tensors must be non-virtual, output tensor must be virtual");
    }

    const auto& actTensorIn0
        = core::utils::findTensorAttributes(tensorMap, actAttr.in_0_tensor_uid());
    const auto& actTensorOut
        = core::utils::findTensorAttributes(tensorMap, actAttr.out_0_tensor_uid());
    if(!actTensorIn0.virtual_() || actTensorOut.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation input from batchnorm must be virtual, output must be non virtual");
    }
}

/**
 * @brief Tensor wiring for the three-node backward + activation fusion: activation
 *        backward chains from the inference output to the backward's dy, x and scale
 *        are shared with the inference node, and virtuality marks the two fused
 *        intermediates.
 *
 * Restated from the old BatchnormPlanBuilder's private helper of the same shape.
 */
void checkBwdActivationTensorWiring(
    const data_objects::BatchnormInferenceAttributes& bnInfAttr,
    const data_objects::PointwiseAttributes& actAttr,
    const data_objects::BatchnormBackwardAttributes& bnBwdAttr,
    const std::unordered_map<int64_t, const data_objects::TensorAttributes*>& tensorMap)
{
    const auto actIn1Uid = actAttr.in_1_tensor_uid();
    if(!actIn1Uid.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation backward requires in_1 tensor (forward activation input)");
    }
    if(actIn1Uid.value() != bnInfAttr.y_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation in_1 must be the batchnorm inference output tensor (y)");
    }
    if(actAttr.out_0_tensor_uid() != bnBwdAttr.dy_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward dy input must be the activation output tensor");
    }
    if(bnBwdAttr.x_tensor_uid() != bnInfAttr.x_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward must use the same X tensor as batchnorm inference");
    }
    if(bnBwdAttr.mean_tensor_uid().has_value()
       && bnBwdAttr.mean_tensor_uid().value() != bnInfAttr.mean_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward must use the same mean tensor as batchnorm inference");
    }
    if(bnBwdAttr.inv_variance_tensor_uid().has_value()
       && bnBwdAttr.inv_variance_tensor_uid().value() != bnInfAttr.inv_variance_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward must use the same inv_variance tensor as batchnorm inference");
    }
    if(bnBwdAttr.scale_tensor_uid() != bnInfAttr.scale_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward must use the same scale tensor as batchnorm inference");
    }

    const auto& bnInfTensorX
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.x_tensor_uid());
    const auto& bnInfTensorMean
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.mean_tensor_uid());
    const auto& bnInfTensorInvVar
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.inv_variance_tensor_uid());
    const auto& bnInfTensorScale
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.scale_tensor_uid());
    const auto& bnInfTensorBias
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.bias_tensor_uid());
    const auto& bnInfTensorY
        = core::utils::findTensorAttributes(tensorMap, bnInfAttr.y_tensor_uid());
    if(bnInfTensorX.virtual_() || bnInfTensorMean.virtual_() || bnInfTensorInvVar.virtual_()
       || bnInfTensorScale.virtual_() || bnInfTensorBias.virtual_() || !bnInfTensorY.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm inference input tensors must be non-virtual, output tensor must be virtual");
    }

    const auto& actTensorIn0
        = core::utils::findTensorAttributes(tensorMap, actAttr.in_0_tensor_uid());
    if(actTensorIn0.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "Activation in_0 (dy gradient) must be non-virtual");
    }
    const auto& actTensorIn1 = core::utils::findTensorAttributes(tensorMap, actIn1Uid.value());
    const auto& actTensorOut
        = core::utils::findTensorAttributes(tensorMap, actAttr.out_0_tensor_uid());
    if(!actTensorIn1.virtual_() || !actTensorOut.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation input from batchnorm must be virtual, output must be virtual");
    }

    const auto& bnBwdTensorDy
        = core::utils::findTensorAttributes(tensorMap, bnBwdAttr.dy_tensor_uid());
    const auto& bnBwdTensorDx
        = core::utils::findTensorAttributes(tensorMap, bnBwdAttr.dx_tensor_uid());
    const auto& bnBwdTensorDscale
        = core::utils::findTensorAttributes(tensorMap, bnBwdAttr.dscale_tensor_uid());
    const auto& bnBwdTensorDbias
        = core::utils::findTensorAttributes(tensorMap, bnBwdAttr.dbias_tensor_uid());
    if(!bnBwdTensorDy.virtual_() || bnBwdTensorDx.virtual_() || bnBwdTensorDscale.virtual_()
       || bnBwdTensorDbias.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward dy input must be virtual, output tensors must be non-virtual");
    }
}

/// Single-node fwd-training discriminator: a BatchnormAttributes node with no peer
/// statistics, validated by the existing training checker.
bool batchnormFwdTrainingMatches(const MatchContext& context, BoundTokens& /*bound*/)
{
    if(context.graph.nodeCount() != 1)
    {
        return false;
    }
    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::BatchnormAttributes)
    {
        return false;
    }

    try
    {
        BatchnormValidator validator(context.graph.getTensorMap());
        validator.checkFwdTrainingTensorConfigSupported(
            node.attributesAs<data_objects::BatchnormAttributes>());
    }
    catch(const std::exception& refusal)
    {
        HIPDNN_PLUGIN_LOG_INFO("batchnorm fwd training declines this graph: " << refusal.what());
        return false;
    }
    return true;
}

/// Single-node backward discriminator: a BatchnormBackwardAttributes node with no peer
/// statistics, validated by the existing backward checker.
bool batchnormBwdMatches(const MatchContext& context, BoundTokens& /*bound*/)
{
    if(context.graph.nodeCount() != 1)
    {
        return false;
    }
    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::BatchnormBackwardAttributes)
    {
        return false;
    }

    try
    {
        BatchnormValidator validator(context.graph.getTensorMap());
        validator.checkBwdTensorConfigSupported(
            node.attributesAs<data_objects::BatchnormBackwardAttributes>());
    }
    catch(const std::exception& refusal)
    {
        HIPDNN_PLUGIN_LOG_INFO("batchnorm backward declines this graph: " << refusal.what());
        return false;
    }
    return true;
}

/// Two-node fwd-inference + activation discriminator: the fusion's tensor wiring
/// (in_0 chains from the inference y, virtuality on both sides) plus the existing
/// fused-activation validator.
bool batchnormFwdInferenceActivationMatches(const MatchContext& context, BoundTokens& /*bound*/)
{
    if(!isTwoNodeInferenceFusionShape(context,
                                      data_objects::NodeAttributes::BatchnormInferenceAttributes))
    {
        return false;
    }

    const auto& bnInfAttr = context.graph.getNodeWrapper(0)
                                .attributesAs<data_objects::BatchnormInferenceAttributes>();
    const auto& actAttr
        = context.graph.getNodeWrapper(1).attributesAs<data_objects::PointwiseAttributes>();

    try
    {
        checkFwdInferenceActivationTensorWiring(bnInfAttr, actAttr, context.graph.getTensorMap());
        BatchnormValidator validator(context.graph.getTensorMap());
        validator.checkInferenceActivationTensorConfigSupported(bnInfAttr, actAttr);
    }
    catch(const std::exception& refusal)
    {
        HIPDNN_PLUGIN_LOG_INFO(
            "batchnorm fwd inference activation fusion declines this graph: " << refusal.what());
        return false;
    }
    return true;
}

/// Two-node fwd-inference-with-variance + activation discriminator, mirroring the
/// plain fusion matcher against the variance-ext attribute type.
bool batchnormFwdInferenceWithVarianceActivationMatches(const MatchContext& context,
                                                        BoundTokens& /*bound*/)
{
    if(!isTwoNodeInferenceFusionShape(
           context, data_objects::NodeAttributes::BatchnormInferenceAttributesVarianceExt))
    {
        return false;
    }

    const auto& bnInfAttr
        = context.graph.getNodeWrapper(0)
              .attributesAs<data_objects::BatchnormInferenceAttributesVarianceExt>();
    const auto& actAttr
        = context.graph.getNodeWrapper(1).attributesAs<data_objects::PointwiseAttributes>();

    try
    {
        checkFwdInferenceWithVarianceActivationTensorWiring(
            bnInfAttr, actAttr, context.graph.getTensorMap());
        BatchnormValidator validator(context.graph.getTensorMap());
        validator.checkInferenceVarianceExtActivationTensorConfigSupported(bnInfAttr, actAttr);
    }
    catch(const std::exception& refusal)
    {
        HIPDNN_PLUGIN_LOG_INFO(
            "batchnorm fwd inference with variance activation fusion declines this graph: "
            << refusal.what());
        return false;
    }
    return true;
}

/// Virtuality a running-statistics tensor must have if the graph names it: real,
/// never virtual, on every one of the four optional prev/next mean/variance tensors.
void checkRunningStatisticsTensorVirtuality(
    const data_objects::BatchnormAttributes& bnAttr,
    const std::unordered_map<int64_t, const data_objects::TensorAttributes*>& tensorMap)
{
    const auto checkNonVirtual = [&](const auto& uid, const char* label) {
        if(!uid.has_value())
        {
            return;
        }
        if(core::utils::findTensorAttributes(tensorMap, uid.value()).virtual_())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                           std::string("Batchnorm ") + label
                                                               + " tensor must be non-virtual");
        }
    };
    checkNonVirtual(bnAttr.prev_running_mean_tensor_uid(), "prev_running_mean");
    checkNonVirtual(bnAttr.prev_running_variance_tensor_uid(), "prev_running_variance");
    checkNonVirtual(bnAttr.next_running_mean_tensor_uid(), "next_running_mean");
    checkNonVirtual(bnAttr.next_running_variance_tensor_uid(), "next_running_variance");
}

/**
 * @brief Tensor wiring for the two-node fwd-training + activation fusion: IO tensors
 *        real on input and virtual on the fused output, optional saved/running
 *        statistics real when present, and the activation's virtual intermediate
 *        chaining from training's virtual y.
 *
 * Restated from BatchnormFwdTrainingPlanBuilder's private checkTensorVirtuality2Node(),
 * which no header exposed for reuse; the tensor-configuration checks that follow it
 * stay on BatchnormValidator and are called, not restated.
 */
void checkFwdTrainingActivationTensorWiring(
    const data_objects::BatchnormAttributes& bnAttr,
    const data_objects::PointwiseAttributes& actAttr,
    const std::unordered_map<int64_t, const data_objects::TensorAttributes*>& tensorMap)
{
    const auto& bnTensorX = core::utils::findTensorAttributes(tensorMap, bnAttr.x_tensor_uid());
    const auto& bnTensorScale
        = core::utils::findTensorAttributes(tensorMap, bnAttr.scale_tensor_uid());
    const auto& bnTensorBias
        = core::utils::findTensorAttributes(tensorMap, bnAttr.bias_tensor_uid());
    const auto& bnTensorY = core::utils::findTensorAttributes(tensorMap, bnAttr.y_tensor_uid());
    if(bnTensorX.virtual_() || bnTensorScale.virtual_() || bnTensorBias.virtual_()
       || !bnTensorY.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm training input tensors must be non-virtual, output tensor must be virtual");
    }

    if(bnAttr.mean_tensor_uid().has_value()
       && core::utils::findTensorAttributes(tensorMap, bnAttr.mean_tensor_uid().value()).virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Batchnorm mean tensor must be non-virtual");
    }
    if(bnAttr.inv_variance_tensor_uid().has_value()
       && core::utils::findTensorAttributes(tensorMap, bnAttr.inv_variance_tensor_uid().value())
              .virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "Batchnorm inv_variance tensor must be non-virtual");
    }
    checkRunningStatisticsTensorVirtuality(bnAttr, tensorMap);

    const auto& actTensorIn0
        = core::utils::findTensorAttributes(tensorMap, actAttr.in_0_tensor_uid());
    const auto& actTensorOut
        = core::utils::findTensorAttributes(tensorMap, actAttr.out_0_tensor_uid());
    if(!actTensorIn0.virtual_() || actTensorOut.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation input from batchnorm must be virtual, output must be non virtual");
    }
}

/// Two-node fwd-training + activation discriminator: BatchnormFwdTrainingPlanBuilder's
/// own fusion shape, checked with its own tensor-wiring and activation-mode rules
/// (RELU family, no leaky slope) via the existing validator.
bool batchnormFwdTrainingActivationMatches(const MatchContext& context, BoundTokens& /*bound*/)
{
    if(context.graph.nodeCount() != 2)
    {
        return false;
    }
    const auto& node0 = context.graph.getNodeWrapper(0);
    const auto& node1 = context.graph.getNodeWrapper(1);
    if(node0.attributesType() != data_objects::NodeAttributes::BatchnormAttributes
       || node1.attributesType() != data_objects::NodeAttributes::PointwiseAttributes)
    {
        return false;
    }

    const auto& bnAttr = node0.attributesAs<data_objects::BatchnormAttributes>();
    const auto& actAttr = node1.attributesAs<data_objects::PointwiseAttributes>();

    if(!isSupportedFwdTrainingActivation(actAttr)
       || actAttr.in_0_tensor_uid() != bnAttr.y_tensor_uid())
    {
        return false;
    }

    try
    {
        checkFwdTrainingActivationTensorWiring(bnAttr, actAttr, context.graph.getTensorMap());
        BatchnormValidator validator(context.graph.getTensorMap());
        validator.checkFwdTrainingActivationTensorConfigSupported(bnAttr, actAttr);
    }
    catch(const std::exception& refusal)
    {
        HIPDNN_PLUGIN_LOG_INFO(
            "batchnorm fwd training activation fusion declines this graph: " << refusal.what());
        return false;
    }
    return true;
}

/// Three-node backward + activation discriminator: fwd inference, activation
/// backward, then batchnorm backward, wired and validated exactly as the old
/// three-node fusion builder did.
bool batchnormBwdActivationMatches(const MatchContext& context, BoundTokens& /*bound*/)
{
    if(context.graph.nodeCount() != 3)
    {
        return false;
    }
    const auto& node0 = context.graph.getNodeWrapper(0);
    const auto& node1 = context.graph.getNodeWrapper(1);
    const auto& node2 = context.graph.getNodeWrapper(2);
    if(node0.attributesType() != data_objects::NodeAttributes::BatchnormInferenceAttributes
       || node1.attributesType() != data_objects::NodeAttributes::PointwiseAttributes
       || node2.attributesType() != data_objects::NodeAttributes::BatchnormBackwardAttributes)
    {
        return false;
    }

    const auto& bnInfAttr = node0.attributesAs<data_objects::BatchnormInferenceAttributes>();
    const auto& actAttr = node1.attributesAs<data_objects::PointwiseAttributes>();
    const auto& bnBwdAttr = node2.attributesAs<data_objects::BatchnormBackwardAttributes>();

    try
    {
        checkBwdActivationTensorWiring(bnInfAttr, actAttr, bnBwdAttr, context.graph.getTensorMap());
        BatchnormValidator validator(context.graph.getTensorMap());
        validator.checkBwdActivationTensorConfigSupported(bnInfAttr, actAttr, bnBwdAttr);
    }
    catch(const std::exception& refusal)
    {
        HIPDNN_PLUGIN_LOG_INFO(
            "batchnorm backward activation fusion declines this graph: " << refusal.what());
        return false;
    }
    return true;
}

/// One kernel per pack, so ranking has nothing to order. A heuristic is still required.
double batchnormScore(const KernelDefinition& /*kernel*/, const MatchContext& /*context*/)
{
    return 0.0;
}

/// The full HIP properties of @p deviceId, needed by every Plan's compile().
hipDeviceProp_t fullDeviceProperties(DeviceId deviceId)
{
    hipDeviceProp_t properties{};
    if(deviceId == NO_DEVICE || hipGetDeviceProperties(&properties, deviceId) != hipSuccess)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "batchnorm dispatch could not query properties for device " + std::to_string(deviceId));
    }
    return properties;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// Owns whichever of the four existing Plan classes this pack's variant built.
class PreparedBatchnorm : public PreparedDispatch
{
public:
    explicit PreparedBatchnorm(std::unique_ptr<BatchnormFwdInferencePlan> plan)
        : _plan(std::move(plan))
    {
    }
    explicit PreparedBatchnorm(std::unique_ptr<BatchnormFwdInferenceWithVariancePlan> plan)
        : _plan(std::move(plan))
    {
    }
    explicit PreparedBatchnorm(std::unique_ptr<BatchnormFwdTrainingPlan> plan)
        : _plan(std::move(plan))
    {
    }
    explicit PreparedBatchnorm(std::unique_ptr<BatchnormBwdPlan> plan)
        : _plan(std::move(plan))
    {
    }

    void execute(const Handle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace) const
    {
        std::visit(
            [&](const auto& plan) {
                plan->execute(handle, deviceBuffers, numDeviceBuffers, workspace);
            },
            _plan);
    }

private:
    std::variant<std::unique_ptr<BatchnormFwdInferencePlan>,
                 std::unique_ptr<BatchnormFwdInferenceWithVariancePlan>,
                 std::unique_ptr<BatchnormFwdTrainingPlan>,
                 std::unique_ptr<BatchnormBwdPlan>>
        _plan;
};

/**
 * @brief The native dispatch behind every batchnorm pack's UDD: builds and launches
 *        whichever Params/Plan pair the matched kernel's `variant` metadata names.
 *
 * One handler for all eight packs rather than eight: the dispatch ABI (workspace
 * query, prepare, launch) is identical across variants, and what differs -- which
 * Params constructor and Plan class to build -- is exactly what the `variant` field
 * exists to select. A pack listing its own matchers is what keeps a graph from ever
 * reaching this handler with a variant its shape does not support.
 */
class BatchnormDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    /// @param kernelCompiler Must outlive this handler; both are process-lifetime.
    explicit BatchnormDispatchHandler(const compilation::IKernelCompiler& kernelCompiler)
        : _kernelCompiler(kernelCompiler)
    {
    }

    /// Every batchnorm plan reports zero workspace (BatchnormPlanBuilder and
    /// BatchnormFwdTrainingPlanBuilder's getMaxWorkspaceSize() both return 0u).
    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        return 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& /*bound*/,
                                              const KernelDefinition& kernel) const override
    {
        const auto& variant = kernel.getStringMetadata(std::string(VARIANT_FIELD));
        const auto& graph = context.graph;
        const auto deviceProperties = fullDeviceProperties(context.deviceId);

        if(variant == variantMetadataValue(Variant::FWD_INFERENCE))
        {
            const auto& attr = graph.getNodeWrapper(0)
                                   .attributesAs<data_objects::BatchnormInferenceAttributes>();
            BatchnormFwdInferenceParams params(attr, graph.getTensorMap());
            auto plan = std::make_unique<BatchnormFwdInferencePlan>(std::move(params));
            plan->compile(_kernelCompiler, deviceProperties);
            return std::make_unique<PreparedBatchnorm>(std::move(plan));
        }
        if(variant == variantMetadataValue(Variant::FWD_INFERENCE_WITH_VARIANCE))
        {
            const auto& attr
                = graph.getNodeWrapper(0)
                      .attributesAs<data_objects::BatchnormInferenceAttributesVarianceExt>();
            BatchnormFwdInferenceWithVarianceParams params(attr, graph.getTensorMap());
            auto plan = std::make_unique<BatchnormFwdInferenceWithVariancePlan>(std::move(params));
            plan->compile(_kernelCompiler, deviceProperties);
            return std::make_unique<PreparedBatchnorm>(std::move(plan));
        }
        if(variant == variantMetadataValue(Variant::FWD_TRAINING))
        {
            const auto& attr
                = graph.getNodeWrapper(0).attributesAs<data_objects::BatchnormAttributes>();
            BatchnormFwdTrainingParams params(attr, graph.getTensorMap());
            auto plan = std::make_unique<BatchnormFwdTrainingPlan>(std::move(params));
            plan->compile(_kernelCompiler, deviceProperties);
            return std::make_unique<PreparedBatchnorm>(std::move(plan));
        }
        if(variant == variantMetadataValue(Variant::BWD))
        {
            const auto& attr
                = graph.getNodeWrapper(0).attributesAs<data_objects::BatchnormBackwardAttributes>();
            BatchnormBwdParams params(attr, graph.getTensorMap());
            auto plan = std::make_unique<BatchnormBwdPlan>(std::move(params));
            plan->compile(_kernelCompiler, deviceProperties);
            return std::make_unique<PreparedBatchnorm>(std::move(plan));
        }
        if(variant == variantMetadataValue(Variant::FWD_INFERENCE_ACTIVATION))
        {
            const auto& bnInfAttr = graph.getNodeWrapper(0)
                                        .attributesAs<data_objects::BatchnormInferenceAttributes>();
            const auto& actAttr
                = graph.getNodeWrapper(1).attributesAs<data_objects::PointwiseAttributes>();
            BatchnormFwdInferenceParams params(bnInfAttr, actAttr, graph.getTensorMap());
            auto plan = std::make_unique<BatchnormFwdInferencePlan>(std::move(params));
            plan->compile(_kernelCompiler, deviceProperties);
            return std::make_unique<PreparedBatchnorm>(std::move(plan));
        }
        if(variant == variantMetadataValue(Variant::FWD_INFERENCE_WITH_VARIANCE_ACTIVATION))
        {
            const auto& bnInfAttr
                = graph.getNodeWrapper(0)
                      .attributesAs<data_objects::BatchnormInferenceAttributesVarianceExt>();
            const auto& actAttr
                = graph.getNodeWrapper(1).attributesAs<data_objects::PointwiseAttributes>();
            BatchnormFwdInferenceWithVarianceParams params(
                bnInfAttr, actAttr, graph.getTensorMap());
            auto plan = std::make_unique<BatchnormFwdInferenceWithVariancePlan>(std::move(params));
            plan->compile(_kernelCompiler, deviceProperties);
            return std::make_unique<PreparedBatchnorm>(std::move(plan));
        }
        if(variant == variantMetadataValue(Variant::FWD_TRAINING_ACTIVATION))
        {
            const auto& bnAttr
                = graph.getNodeWrapper(0).attributesAs<data_objects::BatchnormAttributes>();
            const auto& actAttr
                = graph.getNodeWrapper(1).attributesAs<data_objects::PointwiseAttributes>();
            BatchnormFwdTrainingParams params(bnAttr, actAttr, graph.getTensorMap());
            auto plan = std::make_unique<BatchnormFwdTrainingPlan>(std::move(params));
            plan->compile(_kernelCompiler, deviceProperties);
            return std::make_unique<PreparedBatchnorm>(std::move(plan));
        }
        if(variant == variantMetadataValue(Variant::BWD_ACTIVATION))
        {
            const auto& bnInfAttr = graph.getNodeWrapper(0)
                                        .attributesAs<data_objects::BatchnormInferenceAttributes>();
            const auto& actAttr
                = graph.getNodeWrapper(1).attributesAs<data_objects::PointwiseAttributes>();
            const auto& bnBwdAttr
                = graph.getNodeWrapper(2).attributesAs<data_objects::BatchnormBackwardAttributes>();
            BatchnormBwdParams params(bnBwdAttr, actAttr, bnInfAttr, graph.getTensorMap());
            auto plan = std::make_unique<BatchnormBwdPlan>(std::move(params));
            plan->compile(_kernelCompiler, deviceProperties);
            return std::make_unique<PreparedBatchnorm>(std::move(plan));
        }

        // Unreachable via matching: each pack's kernel carries exactly the variant
        // value its own matcher gates on.
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "kernel '" + toString(kernel.kernelId) + "' declares unknown batchnorm variant '"
                + variant + "'");
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        dynamic_cast<const PreparedBatchnorm&>(prepared).execute(
            handle, deviceBuffers, numDeviceBuffers, workspace);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
};

/// This engine's dispatch handler. Process-lifetime: the registry holds a non-owning
/// pointer while a Container is created and destroyed per handle.
const BatchnormDispatchHandler& batchnormDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const BatchnormDispatchHandler s_dispatchHandler(s_kernelCompiler);
    return s_dispatchHandler;
}

} // namespace

void registerBatchnormSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_SHAPE_MATCHER_SYMBOL), &batchnormGraphShapeMatches);
    scope.add(std::string(FWD_INFERENCE_MATCHER_SYMBOL), &batchnormFwdInferenceMatches);
    scope.add(std::string(FWD_INFERENCE_WITH_VARIANCE_MATCHER_SYMBOL),
              &batchnormFwdInferenceWithVarianceMatches);
    scope.add(std::string(FWD_TRAINING_MATCHER_SYMBOL), &batchnormFwdTrainingMatches);
    scope.add(std::string(BWD_MATCHER_SYMBOL), &batchnormBwdMatches);
    scope.add(std::string(FWD_INFERENCE_ACTIVATION_MATCHER_SYMBOL),
              &batchnormFwdInferenceActivationMatches);
    scope.add(std::string(FWD_INFERENCE_WITH_VARIANCE_ACTIVATION_MATCHER_SYMBOL),
              &batchnormFwdInferenceWithVarianceActivationMatches);
    scope.add(std::string(FWD_TRAINING_ACTIVATION_MATCHER_SYMBOL),
              &batchnormFwdTrainingActivationMatches);
    scope.add(std::string(BWD_ACTIVATION_MATCHER_SYMBOL), &batchnormBwdActivationMatches);
    scope.add(std::string(SCORE_SYMBOL), &batchnormScore);
    scope.add(std::string(DISPATCH_SYMBOL), &batchnormDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
