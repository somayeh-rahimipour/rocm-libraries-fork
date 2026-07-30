// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Public pass-through wrapper for the MIOpen public/private library split. This
// translation unit is compiled into the public wrapper library libMIOpen.so.
// Each public C entry point declared in <miopen/miopen.h> has a matching
// extern "C" stub here. Every stub consults the runtime dispatch seam
// (miopen::wrapper::Dispatch, src/private/routing.hpp): when the call routes to
// hipDNN it is handed to the hipDNN forwarding path, otherwise it forwards to the
// corresponding _impl symbol in the private implementation library
// (libMIOpen_private.so). Dispatch reads MIOPEN_HIPDNN_FORWARDING and the
// compile-time forwarding set to make that decision per call. The private
// library's definitions are renamed to their _impl form at build time by
// force-including src/private/miopen_private_rename.h into every private source,
// so these stubs are the only definitions of the public miopenFoo names. This
// file is compiled WITHOUT that rename header, so it sees the public names from
// <miopen/miopen.h>.
//
// HAND-MAINTAINED. Add a stub here whenever a new MIOPEN_EXPORT function is
// added to miopen.h, and a matching `#define miopenNewFn miopenNewFn_impl` line
// to src/private/miopen_private_rename.h. The set of stubs must stay a superset
// of the public entry points implemented in libMIOpen_private.so.

#include <miopen/miopen.h>

#include "routing.hpp"

namespace {
// hipDNN forwarding path for entry points that route to Route::Hipdnn. The
// hipDNN call for a given entry point is implemented by replacing this call in
// that entry point's stub with the op-specific forwarding function. Until an
// entry point's hipDNN path is implemented it lands here and reports
// miopenStatusNotImplemented, so enabling forwarding for an op that has been
// added to the forwarding set without an implementation fails loudly rather than
// silently running the MIOpen implementation.
miopenStatus_t forward_to_hipdnn(const char* /*entryPoint*/) { return miopenStatusNotImplemented; }
} // namespace

extern "C" const char* miopenGetErrorString_impl(miopenStatus_t error);
extern "C" miopenStatus_t miopenGetVersion_impl(size_t* major, size_t* minor, size_t* patch);
extern "C" miopenStatus_t miopenCreate_impl(miopenHandle_t* handle);
extern "C" miopenStatus_t miopenCreateWithStream_impl(miopenHandle_t* handle,
                                                      miopenAcceleratorQueue_t stream);
extern "C" miopenStatus_t miopenDestroy_impl(miopenHandle_t handle);
extern "C" miopenStatus_t miopenSetStream_impl(miopenHandle_t handle,
                                               miopenAcceleratorQueue_t streamID);
extern "C" miopenStatus_t miopenGetStream_impl(miopenHandle_t handle,
                                               miopenAcceleratorQueue_t* streamID);
extern "C" miopenStatus_t miopenSetAllocator_impl(miopenHandle_t handle,
                                                  miopenAllocatorFunction allocator,
                                                  miopenDeallocatorFunction deallocator,
                                                  void* allocatorContext);
extern "C" miopenStatus_t miopenGetKernelTime_impl(miopenHandle_t handle, float* time);
extern "C" miopenStatus_t miopenEnableProfiling_impl(miopenHandle_t handle, bool enable);
extern "C" miopenStatus_t miopenCreateTensorDescriptor_impl(miopenTensorDescriptor_t* tensorDesc);
extern "C" miopenStatus_t miopenSet4dTensorDescriptor_impl(
    miopenTensorDescriptor_t tensorDesc, miopenDataType_t dataType, int n, int c, int h, int w);
extern "C" miopenStatus_t
miopenSetNdTensorDescriptorWithLayout_impl(miopenTensorDescriptor_t tensorDesc,
                                           miopenDataType_t dataType,
                                           miopenTensorLayout_t tensorLayout,
                                           const int* lens,
                                           int num_lens);
extern "C" miopenStatus_t miopenSet4dTensorDescriptorEx_impl(miopenTensorDescriptor_t tensorDesc,
                                                             miopenDataType_t dataType,
                                                             int n,
                                                             int c,
                                                             int h,
                                                             int w,
                                                             int nStride,
                                                             int cStride,
                                                             int hStride,
                                                             int wStride);
extern "C" miopenStatus_t miopenGet4dTensorDescriptor_impl(miopenTensorDescriptor_t tensorDesc,
                                                           miopenDataType_t* dataType,
                                                           int* n,
                                                           int* c,
                                                           int* h,
                                                           int* w,
                                                           int* nStride,
                                                           int* cStride,
                                                           int* hStride,
                                                           int* wStride);
extern "C" miopenStatus_t miopenSetTensorDescriptor_impl(miopenTensorDescriptor_t tensorDesc,
                                                         miopenDataType_t dataType,
                                                         int nbDims,
                                                         const int* dimsA,
                                                         const int* stridesA);
extern "C" miopenStatus_t miopenSetTensorDescriptorV2_impl(miopenTensorDescriptor_t tensorDesc,
                                                           miopenDataType_t dataType,
                                                           int nbDims,
                                                           const size_t* dimsA,
                                                           const size_t* stridesA);
extern "C" miopenStatus_t miopenSetTensorCastType_impl(miopenTensorDescriptor_t tensorDesc,
                                                       miopenDataType_t cast_type);
extern "C" miopenStatus_t miopenGetTensorDescriptorSize_impl(miopenTensorDescriptor_t tensorDesc,
                                                             int* size);
extern "C" miopenStatus_t miopenGetTensorDescriptor_impl(miopenTensorDescriptor_t tensorDesc,
                                                         miopenDataType_t* dataType,
                                                         int* dimsA,
                                                         int* stridesA);
extern "C" miopenStatus_t miopenDestroyTensorDescriptor_impl(miopenTensorDescriptor_t tensorDesc);
extern "C" miopenStatus_t
miopenCreateSeqTensorDescriptor_impl(miopenSeqTensorDescriptor_t* tensorDesc);
extern "C" miopenStatus_t
miopenDestroySeqTensorDescriptor_impl(miopenSeqTensorDescriptor_t tensorDesc);
extern "C" miopenStatus_t miopenOpTensor_impl(miopenHandle_t handle,
                                              miopenTensorOp_t tensorOp,
                                              const void* alpha1,
                                              miopenTensorDescriptor_t aDesc,
                                              const void* A,
                                              const void* alpha2,
                                              miopenTensorDescriptor_t bDesc,
                                              const void* B,
                                              const void* beta,
                                              miopenTensorDescriptor_t cDesc,
                                              void* C);
extern "C" miopenStatus_t miopenSetTensor_impl(miopenHandle_t handle,
                                               miopenTensorDescriptor_t yDesc,
                                               void* y,
                                               const void* alpha);
extern "C" miopenStatus_t miopenScaleTensor_impl(miopenHandle_t handle,
                                                 miopenTensorDescriptor_t yDesc,
                                                 void* y,
                                                 const void* alpha);
extern "C" miopenStatus_t miopenGetTensorNumBytes_impl(miopenTensorDescriptor_t tensorDesc,
                                                       size_t* numBytes);
extern "C" miopenStatus_t miopenTransformTensor_impl(miopenHandle_t handle,
                                                     const void* alpha,
                                                     miopenTensorDescriptor_t xDesc,
                                                     const void* x,
                                                     const void* beta,
                                                     miopenTensorDescriptor_t yDesc,
                                                     void* y);
extern "C" miopenStatus_t
miopenCreateConvolutionDescriptor_impl(miopenConvolutionDescriptor_t* convDesc);
extern "C" miopenStatus_t
miopenInitConvolutionDescriptor_impl(miopenConvolutionDescriptor_t convDesc,
                                     miopenConvolutionMode_t c_mode,
                                     int pad_h,
                                     int pad_w,
                                     int stride_h,
                                     int stride_w,
                                     int dilation_h,
                                     int dilation_w);
extern "C" miopenStatus_t
miopenInitConvolutionNdDescriptor_impl(miopenConvolutionDescriptor_t convDesc,
                                       int spatialDim,
                                       const int* padA,
                                       const int* strideA,
                                       const int* dilationA,
                                       miopenConvolutionMode_t c_mode);
extern "C" miopenStatus_t
miopenGetConvolutionSpatialDim_impl(miopenConvolutionDescriptor_t convDesc, int* spatialDim);
extern "C" miopenStatus_t
miopenGetConvolutionDescriptor_impl(miopenConvolutionDescriptor_t convDesc,
                                    miopenConvolutionMode_t* c_mode,
                                    int* pad_h,
                                    int* pad_w,
                                    int* stride_h,
                                    int* stride_w,
                                    int* dilation_h,
                                    int* dilation_w);
extern "C" miopenStatus_t
miopenGetConvolutionNdDescriptor_impl(miopenConvolutionDescriptor_t convDesc,
                                      int requestedSpatialDim,
                                      int* spatialDim,
                                      int* padA,
                                      int* strideA,
                                      int* dilationA,
                                      miopenConvolutionMode_t* c_mode);
extern "C" miopenStatus_t
miopenGetConvolutionGroupCount_impl(miopenConvolutionDescriptor_t convDesc, int* groupCount);
extern "C" miopenStatus_t
miopenSetConvolutionGroupCount_impl(miopenConvolutionDescriptor_t convDesc, int groupCount);
extern "C" miopenStatus_t miopenSetTransposeConvOutputPadding_impl(
    miopenConvolutionDescriptor_t convDesc, int adj_h, int adj_w);
extern "C" miopenStatus_t miopenSetTransposeConvNdOutputPadding_impl(
    miopenConvolutionDescriptor_t convDesc, int spatialDim, const int* adjA);
extern "C" miopenStatus_t
miopenGetConvolutionForwardOutputDim_impl(miopenConvolutionDescriptor_t convDesc,
                                          miopenTensorDescriptor_t inputTensorDesc,
                                          miopenTensorDescriptor_t filterDesc,
                                          int* n,
                                          int* c,
                                          int* h,
                                          int* w);
extern "C" miopenStatus_t
miopenGetConvolutionNdForwardOutputDim_impl(miopenConvolutionDescriptor_t convDesc,
                                            miopenTensorDescriptor_t inputTensorDesc,
                                            miopenTensorDescriptor_t filterDesc,
                                            int* nDim,
                                            int* outputTensorDimA);
extern "C" miopenStatus_t
miopenDestroyConvolutionDescriptor_impl(miopenConvolutionDescriptor_t convDesc);
extern "C" miopenStatus_t miopenSetConvolutionAttribute_impl(miopenConvolutionDescriptor_t convDesc,
                                                             miopenConvolutionAttrib_t attr,
                                                             int value);
extern "C" miopenStatus_t miopenGetConvolutionAttribute_impl(miopenConvolutionDescriptor_t convDesc,
                                                             miopenConvolutionAttrib_t attr,
                                                             int* value);
extern "C" miopenStatus_t miopenSetConvolutionFindMode_impl(miopenConvolutionDescriptor_t convDesc,
                                                            miopenConvolutionFindMode_t findMode);
extern "C" miopenStatus_t miopenGetConvolutionFindMode_impl(miopenConvolutionDescriptor_t convDesc,
                                                            miopenConvolutionFindMode_t* findMode);
extern "C" miopenStatus_t
miopenConvolutionForwardGetSolutionCount_impl(miopenHandle_t handle,
                                              miopenTensorDescriptor_t wDesc,
                                              miopenTensorDescriptor_t xDesc,
                                              miopenConvolutionDescriptor_t convDesc,
                                              miopenTensorDescriptor_t yDesc,
                                              size_t* solutionCount);
extern "C" miopenStatus_t
miopenConvolutionForwardGetSolution_impl(miopenHandle_t handle,
                                         miopenTensorDescriptor_t wDesc,
                                         miopenTensorDescriptor_t xDesc,
                                         miopenConvolutionDescriptor_t convDesc,
                                         miopenTensorDescriptor_t yDesc,
                                         size_t maxSolutionCount,
                                         size_t* solutionCount,
                                         miopenConvSolution_t* solutions);
extern "C" miopenStatus_t
miopenConvolutionForwardGetSolutionWorkspaceSize_impl(miopenHandle_t handle,
                                                      miopenTensorDescriptor_t wDesc,
                                                      miopenTensorDescriptor_t xDesc,
                                                      miopenConvolutionDescriptor_t convDesc,
                                                      miopenTensorDescriptor_t yDesc,
                                                      uint64_t solution_id,
                                                      size_t* workSpaceSize);
extern "C" miopenStatus_t
miopenConvolutionForwardCompileSolution_impl(miopenHandle_t handle,
                                             miopenTensorDescriptor_t wDesc,
                                             miopenTensorDescriptor_t xDesc,
                                             miopenConvolutionDescriptor_t convDesc,
                                             miopenTensorDescriptor_t yDesc,
                                             uint64_t solution_id);
extern "C" miopenStatus_t
miopenConvolutionForwardImmediate_impl(miopenHandle_t handle,
                                       miopenTensorDescriptor_t wDesc,
                                       const void* w,
                                       miopenTensorDescriptor_t xDesc,
                                       const void* x,
                                       miopenConvolutionDescriptor_t convDesc,
                                       miopenTensorDescriptor_t yDesc,
                                       void* y,
                                       void* workSpace,
                                       size_t workSpaceSize,
                                       uint64_t solution_id);
extern "C" miopenStatus_t
miopenConvolutionBackwardDataGetSolutionCount_impl(miopenHandle_t handle,
                                                   miopenTensorDescriptor_t dyDesc,
                                                   miopenTensorDescriptor_t wDesc,
                                                   miopenConvolutionDescriptor_t convDesc,
                                                   miopenTensorDescriptor_t dxDesc,
                                                   size_t* solutionCount);
extern "C" miopenStatus_t
miopenConvolutionBackwardDataGetSolution_impl(miopenHandle_t handle,
                                              miopenTensorDescriptor_t dyDesc,
                                              miopenTensorDescriptor_t wDesc,
                                              miopenConvolutionDescriptor_t convDesc,
                                              miopenTensorDescriptor_t dxDesc,
                                              size_t maxSolutionCount,
                                              size_t* solutionCount,
                                              miopenConvSolution_t* solutions);
extern "C" miopenStatus_t
miopenConvolutionBackwardDataGetSolutionWorkspaceSize_impl(miopenHandle_t handle,
                                                           miopenTensorDescriptor_t dyDesc,
                                                           miopenTensorDescriptor_t wDesc,
                                                           miopenConvolutionDescriptor_t convDesc,
                                                           miopenTensorDescriptor_t dxDesc,
                                                           uint64_t solution_id,
                                                           size_t* workSpaceSize);
extern "C" miopenStatus_t
miopenConvolutionBackwardDataCompileSolution_impl(miopenHandle_t handle,
                                                  miopenTensorDescriptor_t dyDesc,
                                                  miopenTensorDescriptor_t wDesc,
                                                  miopenConvolutionDescriptor_t convDesc,
                                                  miopenTensorDescriptor_t dxDesc,
                                                  uint64_t solution_id);
extern "C" miopenStatus_t
miopenConvolutionBackwardDataImmediate_impl(miopenHandle_t handle,
                                            miopenTensorDescriptor_t dyDesc,
                                            const void* dy,
                                            miopenTensorDescriptor_t wDesc,
                                            const void* w,
                                            miopenConvolutionDescriptor_t convDesc,
                                            miopenTensorDescriptor_t dxDesc,
                                            void* dx,
                                            void* workSpace,
                                            size_t workSpaceSize,
                                            uint64_t solution_id);
extern "C" miopenStatus_t
miopenConvolutionBackwardWeightsGetSolutionCount_impl(miopenHandle_t handle,
                                                      miopenTensorDescriptor_t dyDesc,
                                                      miopenTensorDescriptor_t xDesc,
                                                      miopenConvolutionDescriptor_t convDesc,
                                                      miopenTensorDescriptor_t dwDesc,
                                                      size_t* solutionCount);
extern "C" miopenStatus_t
miopenConvolutionBackwardWeightsGetSolution_impl(miopenHandle_t handle,
                                                 miopenTensorDescriptor_t dyDesc,
                                                 miopenTensorDescriptor_t xDesc,
                                                 miopenConvolutionDescriptor_t convDesc,
                                                 miopenTensorDescriptor_t dwDesc,
                                                 size_t maxSolutionCount,
                                                 size_t* solutionCount,
                                                 miopenConvSolution_t* solutions);
extern "C" miopenStatus_t miopenConvolutionBackwardWeightsGetSolutionWorkspaceSize_impl(
    miopenHandle_t handle,
    miopenTensorDescriptor_t dyDesc,
    miopenTensorDescriptor_t xDesc,
    miopenConvolutionDescriptor_t convDesc,
    miopenTensorDescriptor_t dwDesc,
    uint64_t solution_id,
    size_t* workSpaceSize);
extern "C" miopenStatus_t
miopenConvolutionBackwardWeightsCompileSolution_impl(miopenHandle_t handle,
                                                     miopenTensorDescriptor_t dyDesc,
                                                     miopenTensorDescriptor_t xDesc,
                                                     miopenConvolutionDescriptor_t convDesc,
                                                     miopenTensorDescriptor_t dwDesc,
                                                     uint64_t solution_id);
extern "C" miopenStatus_t
miopenConvolutionBackwardWeightsImmediate_impl(miopenHandle_t handle,
                                               miopenTensorDescriptor_t dyDesc,
                                               const void* dy,
                                               miopenTensorDescriptor_t xDesc,
                                               const void* x,
                                               miopenConvolutionDescriptor_t convDesc,
                                               miopenTensorDescriptor_t dwDesc,
                                               void* dw,
                                               void* workSpace,
                                               size_t workSpaceSize,
                                               uint64_t solution_id);
extern "C" miopenStatus_t
miopenConvolutionForwardGetWorkSpaceSize_impl(miopenHandle_t handle,
                                              miopenTensorDescriptor_t wDesc,
                                              miopenTensorDescriptor_t xDesc,
                                              miopenConvolutionDescriptor_t convDesc,
                                              miopenTensorDescriptor_t yDesc,
                                              size_t* workSpaceSize);
extern "C" miopenStatus_t
miopenFindConvolutionForwardAlgorithm_impl(miopenHandle_t handle,
                                           miopenTensorDescriptor_t xDesc,
                                           const void* x,
                                           miopenTensorDescriptor_t wDesc,
                                           const void* w,
                                           miopenConvolutionDescriptor_t convDesc,
                                           miopenTensorDescriptor_t yDesc,
                                           void* y,
                                           int requestAlgoCount,
                                           int* returnedAlgoCount,
                                           miopenConvAlgoPerf_t* perfResults,
                                           void* workSpace,
                                           size_t workSpaceSize,
                                           bool exhaustiveSearch);
extern "C" miopenStatus_t miopenConvolutionForward_impl(miopenHandle_t handle,
                                                        const void* alpha,
                                                        miopenTensorDescriptor_t xDesc,
                                                        const void* x,
                                                        miopenTensorDescriptor_t wDesc,
                                                        const void* w,
                                                        miopenConvolutionDescriptor_t convDesc,
                                                        miopenConvFwdAlgorithm_t algo,
                                                        const void* beta,
                                                        miopenTensorDescriptor_t yDesc,
                                                        void* y,
                                                        void* workSpace,
                                                        size_t workSpaceSize);
extern "C" miopenStatus_t miopenConvolutionForwardBias_impl(miopenHandle_t handle,
                                                            const void* alpha,
                                                            miopenTensorDescriptor_t bDesc,
                                                            const void* b,
                                                            const void* beta,
                                                            miopenTensorDescriptor_t yDesc,
                                                            void* y);
extern "C" miopenStatus_t
miopenConvolutionBackwardDataGetWorkSpaceSize_impl(miopenHandle_t handle,
                                                   miopenTensorDescriptor_t dyDesc,
                                                   miopenTensorDescriptor_t wDesc,
                                                   miopenConvolutionDescriptor_t convDesc,
                                                   miopenTensorDescriptor_t dxDesc,
                                                   size_t* workSpaceSize);
extern "C" miopenStatus_t
miopenFindConvolutionBackwardDataAlgorithm_impl(miopenHandle_t handle,
                                                miopenTensorDescriptor_t dyDesc,
                                                const void* dy,
                                                miopenTensorDescriptor_t wDesc,
                                                const void* w,
                                                miopenConvolutionDescriptor_t convDesc,
                                                miopenTensorDescriptor_t dxDesc,
                                                void* dx,
                                                int requestAlgoCount,
                                                int* returnedAlgoCount,
                                                miopenConvAlgoPerf_t* perfResults,
                                                void* workSpace,
                                                size_t workSpaceSize,
                                                bool exhaustiveSearch);
extern "C" miopenStatus_t miopenConvolutionBackwardData_impl(miopenHandle_t handle,
                                                             const void* alpha,
                                                             miopenTensorDescriptor_t dyDesc,
                                                             const void* dy,
                                                             miopenTensorDescriptor_t wDesc,
                                                             const void* w,
                                                             miopenConvolutionDescriptor_t convDesc,
                                                             miopenConvBwdDataAlgorithm_t algo,
                                                             const void* beta,
                                                             miopenTensorDescriptor_t dxDesc,
                                                             void* dx,
                                                             void* workSpace,
                                                             size_t workSpaceSize);
extern "C" miopenStatus_t
miopenConvolutionBackwardWeightsGetWorkSpaceSize_impl(miopenHandle_t handle,
                                                      miopenTensorDescriptor_t dyDesc,
                                                      miopenTensorDescriptor_t xDesc,
                                                      miopenConvolutionDescriptor_t convDesc,
                                                      miopenTensorDescriptor_t dwDesc,
                                                      size_t* workSpaceSize);
extern "C" miopenStatus_t
miopenFindConvolutionBackwardWeightsAlgorithm_impl(miopenHandle_t handle,
                                                   miopenTensorDescriptor_t dyDesc,
                                                   const void* dy,
                                                   miopenTensorDescriptor_t xDesc,
                                                   const void* x,
                                                   miopenConvolutionDescriptor_t convDesc,
                                                   miopenTensorDescriptor_t dwDesc,
                                                   void* dw,
                                                   int requestAlgoCount,
                                                   int* returnedAlgoCount,
                                                   miopenConvAlgoPerf_t* perfResults,
                                                   void* workSpace,
                                                   size_t workSpaceSize,
                                                   bool exhaustiveSearch);
extern "C" miopenStatus_t
miopenConvolutionBackwardWeights_impl(miopenHandle_t handle,
                                      const void* alpha,
                                      miopenTensorDescriptor_t dyDesc,
                                      const void* dy,
                                      miopenTensorDescriptor_t xDesc,
                                      const void* x,
                                      miopenConvolutionDescriptor_t convDesc,
                                      miopenConvBwdWeightsAlgorithm_t algo,
                                      const void* beta,
                                      miopenTensorDescriptor_t dwDesc,
                                      void* dw,
                                      void* workSpace,
                                      size_t workSpaceSize);
extern "C" miopenStatus_t miopenConvolutionBackwardBias_impl(miopenHandle_t handle,
                                                             const void* alpha,
                                                             miopenTensorDescriptor_t dyDesc,
                                                             const void* dy,
                                                             const void* beta,
                                                             miopenTensorDescriptor_t dbDesc,
                                                             void* db);
extern "C" miopenStatus_t miopenCreatePoolingDescriptor_impl(miopenPoolingDescriptor_t* poolDesc);
extern "C" miopenStatus_t miopenSetPoolingIndexType_impl(miopenPoolingDescriptor_t poolDesc,
                                                         miopenIndexType_t index_type);
extern "C" miopenStatus_t miopenGetPoolingIndexType_impl(miopenPoolingDescriptor_t poolDesc,
                                                         miopenIndexType_t* index_type);
extern "C" miopenStatus_t
miopenSetPoolingWorkSpaceIndexMode_impl(miopenPoolingDescriptor_t poolDesc,
                                        miopenPoolingWorkspaceIndexMode_t workspace_index);
extern "C" miopenStatus_t
miopenGetPoolingWorkSpaceIndexMode_impl(miopenPoolingDescriptor_t poolDesc,
                                        miopenPoolingWorkspaceIndexMode_t* workspace_index);
extern "C" miopenStatus_t miopenSet2dPoolingDescriptor_impl(miopenPoolingDescriptor_t poolDesc,
                                                            miopenPoolingMode_t mode,
                                                            int windowHeight,
                                                            int windowWidth,
                                                            int pad_h,
                                                            int pad_w,
                                                            int stride_h,
                                                            int stride_w);
extern "C" miopenStatus_t miopenGet2dPoolingDescriptor_impl(miopenPoolingDescriptor_t poolDesc,
                                                            miopenPoolingMode_t* mode,
                                                            int* windowHeight,
                                                            int* windowWidth,
                                                            int* pad_h,
                                                            int* pad_w,
                                                            int* stride_h,
                                                            int* stride_w);
extern "C" miopenStatus_t miopenGetPoolingForwardOutputDim_impl(miopenPoolingDescriptor_t poolDesc,
                                                                miopenTensorDescriptor_t tensorDesc,
                                                                int* n,
                                                                int* c,
                                                                int* h,
                                                                int* w);
extern "C" miopenStatus_t miopenSetNdPoolingDescriptor_impl(miopenPoolingDescriptor_t poolDesc,
                                                            miopenPoolingMode_t mode,
                                                            int nbDims,
                                                            const int* windowDimA,
                                                            const int* padA,
                                                            const int* stridesA);
extern "C" miopenStatus_t miopenGetNdPoolingDescriptor_impl(miopenPoolingDescriptor_t poolDesc,
                                                            int nbDimsRequested,
                                                            miopenPoolingMode_t* mode,
                                                            int* nbDims,
                                                            int* windowDimA,
                                                            int* padA,
                                                            int* stridesA);
extern "C" miopenStatus_t
miopenGetPoolingNdForwardOutputDim_impl(miopenPoolingDescriptor_t poolDesc,
                                        miopenTensorDescriptor_t tensorDesc,
                                        int dims,
                                        int* tensorDimArr);
extern "C" miopenStatus_t miopenPoolingGetWorkSpaceSize_impl(miopenTensorDescriptor_t yDesc,
                                                             size_t* workSpaceSize);
extern "C" miopenStatus_t miopenPoolingGetWorkSpaceSizeV2_impl(miopenPoolingDescriptor_t poolDesc,
                                                               miopenTensorDescriptor_t yDesc,
                                                               size_t* workSpaceSize);
extern "C" miopenStatus_t miopenPoolingForward_impl(miopenHandle_t handle,
                                                    miopenPoolingDescriptor_t poolDesc,
                                                    const void* alpha,
                                                    miopenTensorDescriptor_t xDesc,
                                                    const void* x,
                                                    const void* beta,
                                                    miopenTensorDescriptor_t yDesc,
                                                    void* y,
                                                    bool do_backward,
                                                    void* workSpace,
                                                    size_t workSpaceSize);
extern "C" miopenStatus_t miopenPoolingBackward_impl(miopenHandle_t handle,
                                                     miopenPoolingDescriptor_t poolDesc,
                                                     const void* alpha,
                                                     miopenTensorDescriptor_t yDesc,
                                                     const void* y,
                                                     miopenTensorDescriptor_t dyDesc,
                                                     const void* dy,
                                                     miopenTensorDescriptor_t xDesc,
                                                     const void* x,
                                                     const void* beta,
                                                     miopenTensorDescriptor_t dxDesc,
                                                     void* dx,
                                                     void* workSpace);
extern "C" miopenStatus_t miopenDestroyPoolingDescriptor_impl(miopenPoolingDescriptor_t poolDesc);
extern "C" miopenStatus_t miopenCreateLRNDescriptor_impl(miopenLRNDescriptor_t* lrnDesc);
extern "C" miopenStatus_t miopenSetLRNDescriptor_impl(miopenLRNDescriptor_t lrnDesc,
                                                      miopenLRNMode_t mode,
                                                      unsigned int lrnN,
                                                      double lrnAlpha,
                                                      double lrnBeta,
                                                      double lrnK);
extern "C" miopenStatus_t miopenGetLRNDescriptor_impl(miopenLRNDescriptor_t lrnDesc,
                                                      miopenLRNMode_t* mode,
                                                      unsigned int* lrnN,
                                                      double* lrnAlpha,
                                                      double* lrnBeta,
                                                      double* lrnK);
extern "C" miopenStatus_t miopenLRNGetWorkSpaceSize_impl(miopenTensorDescriptor_t yDesc,
                                                         size_t* workSpaceSize);
extern "C" miopenStatus_t miopenLRNForward_impl(miopenHandle_t handle,
                                                miopenLRNDescriptor_t lrnDesc,
                                                const void* alpha,
                                                miopenTensorDescriptor_t xDesc,
                                                const void* x,
                                                const void* beta,
                                                miopenTensorDescriptor_t yDesc,
                                                void* y,
                                                bool do_backward,
                                                void* workSpace);
extern "C" miopenStatus_t miopenLRNBackward_impl(miopenHandle_t handle,
                                                 miopenLRNDescriptor_t lrnDesc,
                                                 const void* alpha,
                                                 miopenTensorDescriptor_t yDesc,
                                                 const void* y,
                                                 miopenTensorDescriptor_t dyDesc,
                                                 const void* dy,
                                                 miopenTensorDescriptor_t xDesc,
                                                 const void* x,
                                                 const void* beta,
                                                 miopenTensorDescriptor_t dxDesc,
                                                 void* dx,
                                                 const void* workSpace);
extern "C" miopenStatus_t miopenDestroyLRNDescriptor_impl(miopenLRNDescriptor_t lrnDesc);
extern "C" miopenStatus_t miopenLayerNormForward_impl(miopenHandle_t handle,
                                                      miopenNormMode_t mode,
                                                      miopenTensorDescriptor_t xDesc,
                                                      const void* x,
                                                      miopenTensorDescriptor_t weightDesc,
                                                      const void* weight,
                                                      miopenTensorDescriptor_t biasDesc,
                                                      const void* bias,
                                                      float epsilon,
                                                      int32_t normalized_dim,
                                                      miopenTensorDescriptor_t yDesc,
                                                      void* y,
                                                      miopenTensorDescriptor_t meanDesc,
                                                      void* mean,
                                                      miopenTensorDescriptor_t rstdDesc,
                                                      void* rstd);
extern "C" miopenStatus_t
miopenGetLayerNormBackwardWorkspaceSize_impl(miopenHandle_t handle,
                                             miopenNormMode_t mode,
                                             miopenTensorDescriptor_t dyDesc,
                                             miopenTensorDescriptor_t xDesc,
                                             miopenTensorDescriptor_t weightDesc,
                                             miopenTensorDescriptor_t meanDesc,
                                             miopenTensorDescriptor_t rstdDesc,
                                             int32_t normalized_dim,
                                             miopenTensorDescriptor_t dxDesc,
                                             miopenTensorDescriptor_t dwDesc,
                                             miopenTensorDescriptor_t dbDesc,
                                             size_t* sizeInBytes);
extern "C" miopenStatus_t miopenLayerNormBackward_impl(miopenHandle_t handle,
                                                       miopenNormMode_t mode,
                                                       void* workspace,
                                                       size_t workspaceSizeInBytes,
                                                       miopenTensorDescriptor_t dyDesc,
                                                       const void* dy,
                                                       miopenTensorDescriptor_t xDesc,
                                                       const void* x,
                                                       miopenTensorDescriptor_t weightDesc,
                                                       const void* weight,
                                                       miopenTensorDescriptor_t meanDesc,
                                                       const void* mean,
                                                       miopenTensorDescriptor_t rstdDesc,
                                                       const void* rstd,
                                                       int32_t normalized_dim,
                                                       miopenTensorDescriptor_t dxDesc,
                                                       void* dx,
                                                       miopenTensorDescriptor_t dwDesc,
                                                       void* dw,
                                                       miopenTensorDescriptor_t dbDesc,
                                                       void* db);
extern "C" miopenStatus_t miopenCatForward_impl(miopenHandle_t handle,
                                                int32_t xCount,
                                                const miopenTensorDescriptor_t* xDescs,
                                                const void* const* xs,
                                                miopenTensorDescriptor_t yDesc,
                                                void* y,
                                                int32_t dim);
extern "C" miopenStatus_t
miopenDeriveBNTensorDescriptor_impl(miopenTensorDescriptor_t derivedBnDesc,
                                    miopenTensorDescriptor_t xDesc,
                                    miopenBatchNormMode_t bn_mode);
extern "C" miopenStatus_t
miopenBatchNormalizationForwardTraining_impl(miopenHandle_t handle,
                                             miopenBatchNormMode_t bn_mode,
                                             void* alpha,
                                             void* beta,
                                             miopenTensorDescriptor_t xDesc,
                                             const void* x,
                                             miopenTensorDescriptor_t yDesc,
                                             void* y,
                                             miopenTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                             void* bnScale,
                                             void* bnBias,
                                             double expAvgFactor,
                                             void* resultRunningMean,
                                             void* resultRunningVariance,
                                             double epsilon,
                                             void* resultSaveMean,
                                             void* resultSaveInvVariance);
extern "C" miopenStatus_t
miopenBatchNormalizationForwardTraining_V2_impl(miopenHandle_t handle,
                                                miopenBatchNormMode_t bn_mode,
                                                void* alpha,
                                                void* beta,
                                                miopenTensorDescriptor_t xDesc,
                                                const void* x,
                                                miopenTensorDescriptor_t yDesc,
                                                void* y,
                                                miopenTensorDescriptor_t scaleDesc,
                                                miopenTensorDescriptor_t biasVarDesc,
                                                miopenTensorDescriptor_t savedMeanDesc,
                                                miopenTensorDescriptor_t savedVarDesc,
                                                void* bnScale,
                                                void* bnBias,
                                                double expAvgFactor,
                                                void* resultRunningMean,
                                                void* resultRunningVariance,
                                                double epsilon,
                                                void* resultSaveMean,
                                                void* resultSaveInvVariance);
extern "C" miopenStatus_t
miopenBatchNormalizationForwardTraining_V3_impl(miopenHandle_t handle,
                                                miopenBatchNormMode_t bn_mode,
                                                void* alpha,
                                                void* beta,
                                                miopenTensorDescriptor_t xDesc,
                                                const void* x,
                                                miopenTensorDescriptor_t yDesc,
                                                void* y,
                                                miopenTensorDescriptor_t scaleDesc,
                                                miopenTensorDescriptor_t biasVarDesc,
                                                miopenTensorDescriptor_t savedMeanDesc,
                                                miopenTensorDescriptor_t savedVarDesc,
                                                void* bnScale,
                                                void* bnBias,
                                                double expAvgFactor,
                                                const void* prevResultRunningMean,
                                                const void* prevResultRunningVariance,
                                                void* nextResultRunningMean,
                                                void* nextResultRunningVariance,
                                                double epsilon,
                                                void* resultSaveMean,
                                                void* resultSaveInvVariance);
extern "C" miopenStatus_t
miopenBatchNormForwardTrainingActivation_impl(miopenHandle_t handle,
                                              miopenBatchNormMode_t bn_mode,
                                              void* alpha,
                                              void* beta,
                                              miopenTensorDescriptor_t xDesc,
                                              const void* x,
                                              miopenTensorDescriptor_t yDesc,
                                              void* y,
                                              miopenTensorDescriptor_t scaleDesc,
                                              miopenTensorDescriptor_t biasVarDesc,
                                              miopenTensorDescriptor_t savedMeanDesc,
                                              miopenTensorDescriptor_t savedVarDesc,
                                              void* bnScale,
                                              void* bnBias,
                                              double expAvgFactor,
                                              void* resultRunningMean,
                                              void* resultRunningVariance,
                                              double epsilon,
                                              void* resultSaveMean,
                                              void* resultSaveInvVariance,
                                              miopenActivationDescriptor_t activDesc);
extern "C" miopenStatus_t
miopenBatchNormForwardTrainingActivation_V2_impl(miopenHandle_t handle,
                                                 miopenBatchNormMode_t bn_mode,
                                                 void* alpha,
                                                 void* beta,
                                                 miopenTensorDescriptor_t xDesc,
                                                 const void* x,
                                                 miopenTensorDescriptor_t yDesc,
                                                 void* y,
                                                 miopenTensorDescriptor_t scaleDesc,
                                                 miopenTensorDescriptor_t biasVarDesc,
                                                 miopenTensorDescriptor_t savedMeanDesc,
                                                 miopenTensorDescriptor_t savedVarDesc,
                                                 void* bnScale,
                                                 void* bnBias,
                                                 double expAvgFactor,
                                                 const void* prevResultRunningMean,
                                                 const void* prevResultRunningVariance,
                                                 void* nextResultRunningMean,
                                                 void* nextResultRunningVariance,
                                                 double epsilon,
                                                 void* resultSaveMean,
                                                 void* resultSaveInvVariance,
                                                 miopenActivationDescriptor_t activDesc);
extern "C" miopenStatus_t
miopenBatchNormalizationForwardInference_impl(miopenHandle_t handle,
                                              miopenBatchNormMode_t bn_mode,
                                              void* alpha,
                                              void* beta,
                                              miopenTensorDescriptor_t xDesc,
                                              const void* x,
                                              miopenTensorDescriptor_t yDesc,
                                              void* y,
                                              miopenTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                              void* bnScale,
                                              void* bnBias,
                                              void* estimatedMean,
                                              void* estimatedVariance,
                                              double epsilon);
extern "C" miopenStatus_t
miopenBatchNormalizationForwardInference_V2_impl(miopenHandle_t handle,
                                                 miopenBatchNormMode_t bn_mode,
                                                 void* alpha,
                                                 void* beta,
                                                 miopenTensorDescriptor_t xDesc,
                                                 const void* x,
                                                 miopenTensorDescriptor_t yDesc,
                                                 void* y,
                                                 miopenTensorDescriptor_t scaleDesc,
                                                 miopenTensorDescriptor_t biasDesc,
                                                 miopenTensorDescriptor_t estMeanDesc,
                                                 miopenTensorDescriptor_t estVarianceDesc,
                                                 void* bnScale,
                                                 void* bnBias,
                                                 void* estimatedMean,
                                                 void* estimatedVariance,
                                                 double epsilon);
extern "C" miopenStatus_t miopenBatchNormalizationForwardInferenceInvVariance_impl(
    miopenHandle_t handle,
    miopenBatchNormMode_t bn_mode,
    void* alpha,
    void* beta,
    miopenTensorDescriptor_t xDesc,
    const void* x,
    miopenTensorDescriptor_t yDesc,
    void* y,
    miopenTensorDescriptor_t scaleDesc,
    miopenTensorDescriptor_t biasDesc,
    miopenTensorDescriptor_t estMeanDesc,
    miopenTensorDescriptor_t estInvVarianceDesc,
    void* bnScale,
    void* bnBias,
    void* estimatedMean,
    void* estimatedInvVariance);
extern "C" miopenStatus_t miopenBatchNormForwardInferenceActivationInvVariance_impl(
    miopenHandle_t handle,
    miopenBatchNormMode_t bn_mode,
    void* alpha,
    void* beta,
    miopenTensorDescriptor_t xDesc,
    const void* x,
    miopenTensorDescriptor_t yDesc,
    void* y,
    miopenTensorDescriptor_t scaleDesc,
    miopenTensorDescriptor_t biasDesc,
    miopenTensorDescriptor_t estMeanDesc,
    miopenTensorDescriptor_t estInvVarianceDesc,
    void* bnScale,
    void* bnBias,
    void* estimatedMean,
    void* estimatedInvVariance,
    miopenActivationDescriptor_t activDesc);
extern "C" miopenStatus_t
miopenBatchNormForwardInferenceActivation_impl(miopenHandle_t handle,
                                               miopenBatchNormMode_t bn_mode,
                                               void* alpha,
                                               void* beta,
                                               miopenTensorDescriptor_t xDesc,
                                               const void* x,
                                               miopenTensorDescriptor_t yDesc,
                                               void* y,
                                               miopenTensorDescriptor_t scaleDesc,
                                               miopenTensorDescriptor_t biasDesc,
                                               miopenTensorDescriptor_t estMeanDesc,
                                               miopenTensorDescriptor_t estVarianceDesc,
                                               void* bnScale,
                                               void* bnBias,
                                               void* estimatedMean,
                                               void* estimatedVariance,
                                               double epsilon,
                                               miopenActivationDescriptor_t activDesc);
extern "C" miopenStatus_t
miopenBatchNormalizationBackward_impl(miopenHandle_t handle,
                                      miopenBatchNormMode_t bn_mode,
                                      const void* alphaDataDiff,
                                      const void* betaDataDiff,
                                      const void* alphaParamDiff,
                                      const void* betaParamDiff,
                                      miopenTensorDescriptor_t xDesc,
                                      const void* x,
                                      miopenTensorDescriptor_t dyDesc,
                                      const void* dy,
                                      miopenTensorDescriptor_t dxDesc,
                                      void* dx,
                                      miopenTensorDescriptor_t bnScaleBiasDiffDesc,
                                      const void* bnScale,
                                      void* resultBnScaleDiff,
                                      void* resultBnBiasDiff,
                                      double epsilon,
                                      const void* savedMean,
                                      const void* savedInvVariance);
extern "C" miopenStatus_t
miopenBatchNormalizationBackward_V2_impl(miopenHandle_t handle,
                                         miopenBatchNormMode_t bn_mode,
                                         const void* alphaDataDiff,
                                         const void* betaDataDiff,
                                         const void* alphaParamDiff,
                                         const void* betaParamDiff,
                                         miopenTensorDescriptor_t xDesc,
                                         const void* x,
                                         miopenTensorDescriptor_t dyDesc,
                                         const void* dy,
                                         miopenTensorDescriptor_t dxDesc,
                                         void* dx,
                                         miopenTensorDescriptor_t scaleDesc,
                                         miopenTensorDescriptor_t biasDesc,
                                         miopenTensorDescriptor_t savedMeanDesc,
                                         miopenTensorDescriptor_t savedVarDesc,
                                         const void* bnScale,
                                         void* resultBnScaleDiff,
                                         void* resultBnBiasDiff,
                                         double epsilon,
                                         const void* savedMean,
                                         const void* savedInvVariance);
extern "C" miopenStatus_t
miopenBatchNormBackwardActivation_impl(miopenHandle_t handle,
                                       miopenBatchNormMode_t bn_mode,
                                       const void* alphaDataDiff,
                                       const void* betaDataDiff,
                                       const void* alphaParamDiff,
                                       const void* betaParamDiff,
                                       miopenTensorDescriptor_t xDesc,
                                       const void* x,
                                       miopenTensorDescriptor_t dyDesc,
                                       const void* dy,
                                       miopenTensorDescriptor_t dxDesc,
                                       void* dx,
                                       miopenTensorDescriptor_t scaleDesc,
                                       miopenTensorDescriptor_t biasDesc,
                                       miopenTensorDescriptor_t savedMeanDesc,
                                       miopenTensorDescriptor_t savedVarianceDesc,
                                       const void* bnScale,
                                       const void* bnBias,
                                       void* resultBnScaleDiff,
                                       void* resultBnBiasDiff,
                                       double epsilon,
                                       const void* savedMean,
                                       const void* savedInvVariance,
                                       miopenActivationDescriptor_t activDesc);
extern "C" miopenStatus_t
miopenCreateActivationDescriptor_impl(miopenActivationDescriptor_t* activDesc);
extern "C" miopenStatus_t miopenSetActivationDescriptor_impl(miopenActivationDescriptor_t activDesc,
                                                             miopenActivationMode_t mode,
                                                             double activAlpha,
                                                             double activBeta,
                                                             double activGamma);
extern "C" miopenStatus_t miopenGetActivationDescriptor_impl(miopenActivationDescriptor_t activDesc,
                                                             miopenActivationMode_t* mode,
                                                             double* activAlpha,
                                                             double* activBeta,
                                                             double* activGamma);
extern "C" miopenStatus_t miopenActivationForward_impl(miopenHandle_t handle,
                                                       miopenActivationDescriptor_t activDesc,
                                                       const void* alpha,
                                                       miopenTensorDescriptor_t xDesc,
                                                       const void* x,
                                                       const void* beta,
                                                       miopenTensorDescriptor_t yDesc,
                                                       void* y);
extern "C" miopenStatus_t miopenActivationBackward_impl(miopenHandle_t handle,
                                                        miopenActivationDescriptor_t activDesc,
                                                        const void* alpha,
                                                        miopenTensorDescriptor_t yDesc,
                                                        const void* y,
                                                        miopenTensorDescriptor_t dyDesc,
                                                        const void* dy,
                                                        miopenTensorDescriptor_t xDesc,
                                                        const void* x,
                                                        const void* beta,
                                                        miopenTensorDescriptor_t dxDesc,
                                                        void* dx);
extern "C" miopenStatus_t
miopenDestroyActivationDescriptor_impl(miopenActivationDescriptor_t activDesc);
extern "C" miopenStatus_t miopenGLUForward_impl(miopenHandle_t handle,
                                                miopenTensorDescriptor_t inputDesc,
                                                const void* input,
                                                miopenTensorDescriptor_t outputDesc,
                                                void* output,
                                                uint32_t dim);
extern "C" miopenStatus_t miopenGLUBackward_impl(miopenHandle_t handle,
                                                 miopenTensorDescriptor_t inputDesc,
                                                 const void* input,
                                                 miopenTensorDescriptor_t outputGradDesc,
                                                 const void* outputGrad,
                                                 miopenTensorDescriptor_t inputGradDesc,
                                                 void* inputGrad,
                                                 uint32_t dim);
extern "C" miopenStatus_t miopenSoftmaxForward_impl(miopenHandle_t handle,
                                                    const void* alpha,
                                                    miopenTensorDescriptor_t xDesc,
                                                    const void* x,
                                                    const void* beta,
                                                    miopenTensorDescriptor_t yDesc,
                                                    void* y);
extern "C" miopenStatus_t miopenSoftmaxBackward_impl(miopenHandle_t handle,
                                                     const void* alpha,
                                                     miopenTensorDescriptor_t yDesc,
                                                     const void* y,
                                                     miopenTensorDescriptor_t dyDesc,
                                                     const void* dy,
                                                     const void* beta,
                                                     miopenTensorDescriptor_t dxDesc,
                                                     void* dx);
extern "C" miopenStatus_t miopenSoftmaxForward_V2_impl(miopenHandle_t handle,
                                                       const void* alpha,
                                                       miopenTensorDescriptor_t xDesc,
                                                       const void* x,
                                                       const void* beta,
                                                       miopenTensorDescriptor_t yDesc,
                                                       void* y,
                                                       miopenSoftmaxAlgorithm_t algorithm,
                                                       miopenSoftmaxMode_t mode);
extern "C" miopenStatus_t miopenSoftmaxBackward_V2_impl(miopenHandle_t handle,
                                                        const void* alpha,
                                                        miopenTensorDescriptor_t yDesc,
                                                        const void* y,
                                                        miopenTensorDescriptor_t dyDesc,
                                                        const void* dy,
                                                        const void* beta,
                                                        miopenTensorDescriptor_t dxDesc,
                                                        void* dx,
                                                        miopenSoftmaxAlgorithm_t algorithm,
                                                        miopenSoftmaxMode_t mode);
extern "C" miopenStatus_t miopenCreateFusionPlan_impl(miopenFusionPlanDescriptor_t* fusePlanDesc,
                                                      miopenFusionDirection_t fuseDirection,
                                                      miopenTensorDescriptor_t inputDesc);
extern "C" miopenStatus_t miopenDestroyFusionPlan_impl(miopenFusionPlanDescriptor_t fusePlanDesc);
extern "C" miopenStatus_t miopenCompileFusionPlan_impl(miopenHandle_t handle,
                                                       miopenFusionPlanDescriptor_t fusePlanDesc);
extern "C" miopenStatus_t miopenFusionPlanGetOp_impl(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                     int op_idx,
                                                     miopenFusionOpDescriptor_t* op);
extern "C" miopenStatus_t
miopenFusionPlanGetWorkSpaceSize_impl(miopenHandle_t handle,
                                      miopenFusionPlanDescriptor_t fusePlanDesc,
                                      size_t* workSpaceSize,
                                      miopenConvFwdAlgorithm_t algo);
extern "C" miopenStatus_t
miopenFusionPlanConvolutionGetAlgo_impl(miopenFusionPlanDescriptor_t fusePlanDesc,
                                        int requestAlgoCount,
                                        int* returnedAlgoCount,
                                        miopenConvFwdAlgorithm_t* returnedAlgos);
extern "C" miopenStatus_t
miopenFusionPlanConvolutionSetAlgo_impl(miopenFusionPlanDescriptor_t fusePlanDesc,
                                        miopenConvFwdAlgorithm_t algo);
extern "C" miopenStatus_t miopenCreateOpConvForward_impl(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                         miopenFusionOpDescriptor_t* convOp,
                                                         miopenConvolutionDescriptor_t convDesc,
                                                         miopenTensorDescriptor_t wDesc);
extern "C" miopenStatus_t
miopenCreateOpActivationForward_impl(miopenFusionPlanDescriptor_t fusePlanDesc,
                                     miopenFusionOpDescriptor_t* activFwdOp,
                                     miopenActivationMode_t mode);
extern "C" miopenStatus_t
miopenCreateOpActivationBackward_impl(miopenFusionPlanDescriptor_t fusePlanDesc,
                                      miopenFusionOpDescriptor_t* activBwdOp,
                                      miopenActivationMode_t mode);
extern "C" miopenStatus_t miopenCreateOpBiasForward_impl(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                         miopenFusionOpDescriptor_t* biasOp,
                                                         miopenTensorDescriptor_t bDesc);
extern "C" miopenStatus_t
miopenCreateOpBatchNormInference_impl(miopenFusionPlanDescriptor_t fusePlanDesc,
                                      miopenFusionOpDescriptor_t* bnOp,
                                      miopenBatchNormMode_t bn_mode,
                                      miopenTensorDescriptor_t bnScaleBiasMeanVarDesc);
extern "C" miopenStatus_t
miopenCreateOpBatchNormForward_impl(miopenFusionPlanDescriptor_t fusePlanDesc,
                                    miopenFusionOpDescriptor_t* bnFwdOp,
                                    miopenBatchNormMode_t bn_mode,
                                    bool runningMeanVariance);
extern "C" miopenStatus_t
miopenCreateOpBatchNormBackward_impl(miopenFusionPlanDescriptor_t fusePlanDesc,
                                     miopenFusionOpDescriptor_t* bnBwdOp,
                                     miopenBatchNormMode_t bn_mode);
extern "C" miopenStatus_t miopenCreateOperatorArgs_impl(miopenOperatorArgs_t* args);
extern "C" miopenStatus_t miopenDestroyOperatorArgs_impl(miopenOperatorArgs_t args);
extern "C" miopenStatus_t miopenSetOpArgsConvForward_impl(miopenOperatorArgs_t args,
                                                          miopenFusionOpDescriptor_t convOp,
                                                          const void* alpha,
                                                          const void* beta,
                                                          const void* w);
extern "C" miopenStatus_t miopenSetOpArgsActivForward_impl(miopenOperatorArgs_t args,
                                                           miopenFusionOpDescriptor_t activFwdOp,
                                                           const void* alpha,
                                                           const void* beta,
                                                           double activAlpha,
                                                           double activBeta,
                                                           double activGamma);
extern "C" miopenStatus_t miopenSetOpArgsActivBackward_impl(miopenOperatorArgs_t args,
                                                            miopenFusionOpDescriptor_t activBwdOp,
                                                            const void* alpha,
                                                            const void* beta,
                                                            const void* y,
                                                            const void* reserved,
                                                            double activAlpha,
                                                            double activBeta,
                                                            double activGamma);
extern "C" miopenStatus_t miopenSetOpArgsBatchNormInference_impl(miopenOperatorArgs_t args,
                                                                 miopenFusionOpDescriptor_t bnOp,
                                                                 const void* alpha,
                                                                 const void* beta,
                                                                 const void* bnScale,
                                                                 const void* bnBias,
                                                                 const void* estimatedMean,
                                                                 const void* estimatedVariance,
                                                                 double epsilon);
extern "C" miopenStatus_t miopenSetOpArgsBatchNormForward_impl(miopenOperatorArgs_t args,
                                                               miopenFusionOpDescriptor_t bnOp,
                                                               const void* alpha,
                                                               const void* beta,
                                                               const void* bnScale,
                                                               const void* bnBias,
                                                               void* savedMean,
                                                               void* savedInvVariance,
                                                               void* runningMean,
                                                               void* runningVariance,
                                                               double expAvgFactor,
                                                               double epsilon);
extern "C" miopenStatus_t miopenSetOpArgsBatchNormBackward_impl(miopenOperatorArgs_t args,
                                                                miopenFusionOpDescriptor_t bnOp,
                                                                const void* alpha,
                                                                const void* beta,
                                                                const void* x,
                                                                const void* bnScale,
                                                                const void* bnBias,
                                                                void* resultBnScaleDiff,
                                                                void* resultBnBiasDiff,
                                                                const void* savedMean,
                                                                const void* savedInvVariance);
extern "C" miopenStatus_t miopenSetOpArgsBiasForward_impl(miopenOperatorArgs_t args,
                                                          miopenFusionOpDescriptor_t biasOp,
                                                          const void* alpha,
                                                          const void* beta,
                                                          const void* bias);
extern "C" miopenStatus_t miopenExecuteFusionPlan_impl(miopenHandle_t handle,
                                                       miopenFusionPlanDescriptor_t fusePlanDesc,
                                                       miopenTensorDescriptor_t inputDesc,
                                                       const void* input,
                                                       miopenTensorDescriptor_t outputDesc,
                                                       void* output,
                                                       miopenOperatorArgs_t args);
extern "C" miopenStatus_t miopenExecuteFusionPlan_v2_impl(miopenHandle_t handle,
                                                          miopenFusionPlanDescriptor_t fusePlanDesc,
                                                          miopenTensorDescriptor_t inputDesc,
                                                          const void* input,
                                                          miopenTensorDescriptor_t outputDesc,
                                                          void* output,
                                                          miopenOperatorArgs_t args,
                                                          void* workspace,
                                                          size_t workspaceSize);
extern "C" miopenStatus_t
miopenConvolutionBiasActivationForward_impl(miopenHandle_t handle,
                                            const void* alpha1,
                                            miopenTensorDescriptor_t xDesc,
                                            const void* x,
                                            miopenTensorDescriptor_t wDesc,
                                            const void* w,
                                            miopenConvolutionDescriptor_t convDesc,
                                            miopenConvFwdAlgorithm_t algo,
                                            void* workspace,
                                            size_t workspaceSizeInBytes,
                                            const void* alpha2,
                                            miopenTensorDescriptor_t zDesc,
                                            const void* z,
                                            miopenTensorDescriptor_t biasDesc,
                                            const void* bias,
                                            miopenActivationDescriptor_t activationDesc,
                                            miopenTensorDescriptor_t yDesc,
                                            void* y);
extern "C" miopenStatus_t miopenCreateRNNDescriptor_impl(miopenRNNDescriptor_t* rnnDesc);
extern "C" miopenStatus_t miopenGetRNNDescriptor_impl(miopenRNNDescriptor_t rnnDesc,
                                                      miopenRNNMode_t* rnnMode,
                                                      miopenRNNAlgo_t* algoMode,
                                                      miopenRNNInputMode_t* inputMode,
                                                      miopenRNNDirectionMode_t* dirMode,
                                                      miopenRNNBiasMode_t* biasMode,
                                                      int* hiddenSize,
                                                      int* layer);
extern "C" miopenStatus_t miopenGetRNNDescriptor_V2_impl(miopenRNNDescriptor_t rnnDesc,
                                                         int* hiddenSize,
                                                         int* layer,
                                                         miopenDropoutDescriptor_t* dropoutDesc,
                                                         miopenRNNInputMode_t* inputMode,
                                                         miopenRNNDirectionMode_t* dirMode,
                                                         miopenRNNMode_t* rnnMode,
                                                         miopenRNNBiasMode_t* biasMode,
                                                         miopenRNNAlgo_t* algoMode,
                                                         miopenDataType_t* dataType);
extern "C" miopenStatus_t miopenDestroyRNNDescriptor_impl(miopenRNNDescriptor_t rnnDesc);
extern "C" miopenStatus_t miopenSetRNNDescriptor_impl(miopenRNNDescriptor_t rnnDesc,
                                                      int hsize,
                                                      int nlayers,
                                                      miopenRNNInputMode_t inMode,
                                                      miopenRNNDirectionMode_t direction,
                                                      miopenRNNMode_t rnnMode,
                                                      miopenRNNBiasMode_t biasMode,
                                                      miopenRNNAlgo_t algo,
                                                      miopenDataType_t dataType);
extern "C" miopenStatus_t miopenSetRNNDescriptor_V2_impl(miopenRNNDescriptor_t rnnDesc,
                                                         int hsize,
                                                         int nlayers,
                                                         miopenDropoutDescriptor_t dropoutDesc,
                                                         miopenRNNInputMode_t inMode,
                                                         miopenRNNDirectionMode_t direction,
                                                         miopenRNNMode_t rnnMode,
                                                         miopenRNNBiasMode_t biasMode,
                                                         miopenRNNAlgo_t algo,
                                                         miopenDataType_t dataType);
extern "C" miopenStatus_t
miopenSetRNNDataSeqTensorDescriptor_impl(miopenSeqTensorDescriptor_t seqTensorDesc,
                                         miopenDataType_t dataType,
                                         miopenRNNBaseLayout_t layout,
                                         int maxSequenceLen,
                                         int batchSize,
                                         int vectorSize,
                                         const int* sequenceLenArray,
                                         void* paddingMarker);
extern "C" miopenStatus_t
miopenGetRNNDataSeqTensorDescriptor_impl(miopenSeqTensorDescriptor_t seqTensorDesc,
                                         miopenDataType_t* dataType,
                                         miopenRNNBaseLayout_t* layout,
                                         int* maxSequenceLen,
                                         int* batchSize,
                                         int* vectorSize,
                                         int sequenceLenArrayLimit,
                                         int* sequenceLenArray,
                                         void* paddingMarker);
extern "C" miopenStatus_t miopenGetRNNWorkspaceSize_impl(miopenHandle_t handle,
                                                         miopenRNNDescriptor_t rnnDesc,
                                                         int sequenceLen,
                                                         const miopenTensorDescriptor_t* xDesc,
                                                         size_t* numBytes);
extern "C" miopenStatus_t
miopenGetRNNTrainingReserveSize_impl(miopenHandle_t handle,
                                     miopenRNNDescriptor_t rnnDesc,
                                     int sequenceLen,
                                     const miopenTensorDescriptor_t* xDesc,
                                     size_t* numBytes);
extern "C" miopenStatus_t miopenGetRNNTempSpaceSizes_impl(miopenHandle_t handle,
                                                          miopenRNNDescriptor_t rnnDesc,
                                                          miopenSeqTensorDescriptor_t xDesc,
                                                          miopenRNNFWDMode_t fwdMode,
                                                          size_t* workSpaceSize,
                                                          size_t* reserveSpaceSize);
extern "C" miopenStatus_t miopenGetRNNParamsSize_impl(miopenHandle_t handle,
                                                      miopenRNNDescriptor_t rnnDesc,
                                                      miopenTensorDescriptor_t xDesc,
                                                      size_t* numBytes,
                                                      miopenDataType_t dtype);
extern "C" miopenStatus_t miopenGetRNNParamsDescriptor_impl(miopenHandle_t handle,
                                                            miopenRNNDescriptor_t rnnDesc,
                                                            miopenTensorDescriptor_t xDesc,
                                                            miopenTensorDescriptor_t wDesc,
                                                            miopenDataType_t dtype);
extern "C" miopenStatus_t miopenGetRNNInputTensorSize_impl(miopenHandle_t handle,
                                                           miopenRNNDescriptor_t rnnDesc,
                                                           int seqLen,
                                                           miopenTensorDescriptor_t* xDesc,
                                                           size_t* numBytes);
extern "C" miopenStatus_t miopenGetRNNHiddenTensorSize_impl(miopenHandle_t handle,
                                                            miopenRNNDescriptor_t rnnDesc,
                                                            int seqLen,
                                                            miopenTensorDescriptor_t* xDesc,
                                                            size_t* numBytes);
extern "C" miopenStatus_t miopenGetRNNLayerParamSize_impl(miopenHandle_t handle,
                                                          miopenRNNDescriptor_t rnnDesc,
                                                          int layer,
                                                          miopenTensorDescriptor_t xDesc,
                                                          int paramID,
                                                          size_t* numBytes);
extern "C" miopenStatus_t miopenGetRNNLayerBiasSize_impl(
    miopenHandle_t handle, miopenRNNDescriptor_t rnnDesc, int layer, int biasID, size_t* numBytes);
extern "C" miopenStatus_t miopenGetRNNLayerParam_impl(miopenHandle_t handle,
                                                      miopenRNNDescriptor_t rnnDesc,
                                                      int layer,
                                                      miopenTensorDescriptor_t xDesc,
                                                      miopenTensorDescriptor_t wDesc,
                                                      const void* w,
                                                      int paramID,
                                                      miopenTensorDescriptor_t paramDesc,
                                                      void* layerParam);
extern "C" miopenStatus_t miopenGetRNNLayerBias_impl(miopenHandle_t handle,
                                                     miopenRNNDescriptor_t rnnDesc,
                                                     int layer,
                                                     miopenTensorDescriptor_t xDesc,
                                                     miopenTensorDescriptor_t wDesc,
                                                     const void* w,
                                                     int biasID,
                                                     miopenTensorDescriptor_t biasDesc,
                                                     void* layerBias);
extern "C" miopenStatus_t miopenGetRNNLayerParamOffset_impl(miopenRNNDescriptor_t rnnDesc,
                                                            int layer,
                                                            miopenTensorDescriptor_t xDesc,
                                                            int paramID,
                                                            miopenTensorDescriptor_t paramDesc,
                                                            size_t* layerParamOffset);
extern "C" miopenStatus_t miopenGetRNNLayerBiasOffset_impl(miopenRNNDescriptor_t rnnDesc,
                                                           int layer,
                                                           miopenTensorDescriptor_t xDesc,
                                                           int biasID,
                                                           miopenTensorDescriptor_t biasDesc,
                                                           size_t* layerBiasOffset);
extern "C" miopenStatus_t miopenSetRNNLayerParam_impl(miopenHandle_t handle,
                                                      miopenRNNDescriptor_t rnnDesc,
                                                      int layer,
                                                      miopenTensorDescriptor_t xDesc,
                                                      miopenTensorDescriptor_t wDesc,
                                                      void* w,
                                                      int paramID,
                                                      miopenTensorDescriptor_t paramDesc,
                                                      const void* layerParam);
extern "C" miopenStatus_t miopenSetRNNLayerBias_impl(miopenHandle_t handle,
                                                     miopenRNNDescriptor_t rnnDesc,
                                                     int layer,
                                                     miopenTensorDescriptor_t xDesc,
                                                     miopenTensorDescriptor_t wDesc,
                                                     void* w,
                                                     int biasID,
                                                     miopenTensorDescriptor_t biasDesc,
                                                     const void* layerBias);
extern "C" miopenStatus_t miopenSetRNNPaddingMode_impl(miopenRNNDescriptor_t rnnDesc,
                                                       miopenRNNPaddingMode_t paddingMode);
extern "C" miopenStatus_t miopenGetRNNPaddingMode_impl(miopenRNNDescriptor_t rnnDesc,
                                                       miopenRNNPaddingMode_t* paddingMode);
extern "C" miopenStatus_t miopenRNNForward_impl(miopenHandle_t handle,
                                                miopenRNNDescriptor_t rnnDesc,
                                                miopenRNNFWDMode_t fwdMode,
                                                miopenSeqTensorDescriptor_t xDesc,
                                                const void* x,
                                                miopenTensorDescriptor_t hDesc,
                                                const void* hx,
                                                void* hy,
                                                miopenTensorDescriptor_t cDesc,
                                                const void* cx,
                                                void* cy,
                                                miopenSeqTensorDescriptor_t yDesc,
                                                void* y,
                                                const void* w,
                                                size_t weightSpaceSize,
                                                void* workSpace,
                                                size_t workSpaceNumBytes,
                                                void* reserveSpace,
                                                size_t reserveSpaceNumBytes);
extern "C" miopenStatus_t miopenRNNBackwardSeqData_impl(miopenHandle_t handle,
                                                        miopenRNNDescriptor_t rnnDesc,
                                                        miopenSeqTensorDescriptor_t yDesc,
                                                        const void* y,
                                                        const void* dy,
                                                        miopenTensorDescriptor_t hDesc,
                                                        const void* hx,
                                                        const void* dhy,
                                                        void* dhx,
                                                        miopenTensorDescriptor_t cDesc,
                                                        const void* cx,
                                                        const void* dcy,
                                                        void* dcx,
                                                        miopenSeqTensorDescriptor_t xDesc,
                                                        void* dx,
                                                        const void* w,
                                                        size_t weightSpaceSize,
                                                        void* workSpace,
                                                        size_t workSpaceNumBytes,
                                                        void* reserveSpace,
                                                        size_t reserveSpaceNumBytes);
extern "C" miopenStatus_t miopenRNNBackwardWeightsSeqTensor_impl(miopenHandle_t handle,
                                                                 miopenRNNDescriptor_t rnnDesc,
                                                                 miopenSeqTensorDescriptor_t xDesc,
                                                                 const void* x,
                                                                 miopenTensorDescriptor_t hDesc,
                                                                 const void* hx,
                                                                 miopenSeqTensorDescriptor_t yDesc,
                                                                 const void* y,
                                                                 void* dw,
                                                                 size_t weightSpaceSize,
                                                                 void* workSpace,
                                                                 size_t workSpaceNumBytes,
                                                                 const void* reserveSpace,
                                                                 size_t reserveSpaceNumBytes);
extern "C" miopenStatus_t miopenRNNForwardTraining_impl(miopenHandle_t handle,
                                                        miopenRNNDescriptor_t rnnDesc,
                                                        int sequenceLen,
                                                        const miopenTensorDescriptor_t* xDesc,
                                                        const void* x,
                                                        miopenTensorDescriptor_t hxDesc,
                                                        const void* hx,
                                                        miopenTensorDescriptor_t cxDesc,
                                                        const void* cx,
                                                        miopenTensorDescriptor_t wDesc,
                                                        const void* w,
                                                        const miopenTensorDescriptor_t* yDesc,
                                                        void* y,
                                                        miopenTensorDescriptor_t hyDesc,
                                                        void* hy,
                                                        miopenTensorDescriptor_t cyDesc,
                                                        void* cy,
                                                        void* workSpace,
                                                        size_t workSpaceNumBytes,
                                                        void* reserveSpace,
                                                        size_t reserveSpaceNumBytes);
extern "C" miopenStatus_t miopenRNNBackwardData_impl(miopenHandle_t handle,
                                                     miopenRNNDescriptor_t rnnDesc,
                                                     int sequenceLen,
                                                     const miopenTensorDescriptor_t* yDesc,
                                                     const void* y,
                                                     const miopenTensorDescriptor_t* dyDesc,
                                                     const void* dy,
                                                     miopenTensorDescriptor_t dhyDesc,
                                                     const void* dhy,
                                                     miopenTensorDescriptor_t dcyDesc,
                                                     const void* dcy,
                                                     miopenTensorDescriptor_t wDesc,
                                                     const void* w,
                                                     miopenTensorDescriptor_t hxDesc,
                                                     const void* hx,
                                                     miopenTensorDescriptor_t cxDesc,
                                                     const void* cx,
                                                     const miopenTensorDescriptor_t* dxDesc,
                                                     void* dx,
                                                     miopenTensorDescriptor_t dhxDesc,
                                                     void* dhx,
                                                     miopenTensorDescriptor_t dcxDesc,
                                                     void* dcx,
                                                     void* workSpace,
                                                     size_t workSpaceNumBytes,
                                                     void* reserveSpace,
                                                     size_t reserveSpaceNumBytes);
extern "C" miopenStatus_t miopenRNNBackwardWeights_impl(miopenHandle_t handle,
                                                        miopenRNNDescriptor_t rnnDesc,
                                                        int sequenceLen,
                                                        const miopenTensorDescriptor_t* xDesc,
                                                        const void* x,
                                                        miopenTensorDescriptor_t hxDesc,
                                                        const void* hx,
                                                        const miopenTensorDescriptor_t* yDesc,
                                                        const void* y,
                                                        miopenTensorDescriptor_t dwDesc,
                                                        void* dw,
                                                        void* workSpace,
                                                        size_t workSpaceNumBytes,
                                                        const void* reserveSpace,
                                                        size_t reserveSpaceNumBytes);
extern "C" miopenStatus_t miopenRNNForwardInference_impl(miopenHandle_t handle,
                                                         miopenRNNDescriptor_t rnnDesc,
                                                         int sequenceLen,
                                                         const miopenTensorDescriptor_t* xDesc,
                                                         const void* x,
                                                         miopenTensorDescriptor_t hxDesc,
                                                         const void* hx,
                                                         miopenTensorDescriptor_t cxDesc,
                                                         const void* cx,
                                                         miopenTensorDescriptor_t wDesc,
                                                         const void* w,
                                                         const miopenTensorDescriptor_t* yDesc,
                                                         void* y,
                                                         miopenTensorDescriptor_t hyDesc,
                                                         void* hy,
                                                         miopenTensorDescriptor_t cyDesc,
                                                         void* cy,
                                                         void* workSpace,
                                                         size_t workSpaceNumBytes);
extern "C" miopenStatus_t
miopenCreateCTCLossDescriptor_impl(miopenCTCLossDescriptor_t* ctcLossDesc);
extern "C" miopenStatus_t miopenGetCTCLossDescriptor_impl(miopenCTCLossDescriptor_t ctcLossDesc,
                                                          miopenDataType_t* dataType,
                                                          int* blank_label_id,
                                                          bool* apply_softmax_layer);
extern "C" miopenStatus_t
miopenDestroyCTCLossDescriptor_impl(miopenCTCLossDescriptor_t ctcLossDesc);
extern "C" miopenStatus_t miopenSetCTCLossDescriptor_impl(miopenCTCLossDescriptor_t ctcLossDesc,
                                                          miopenDataType_t dataType,
                                                          int blank_label_id,
                                                          bool apply_softmax_layer);
extern "C" miopenStatus_t miopenGetCTCLossWorkspaceSize_impl(miopenHandle_t handle,
                                                             miopenTensorDescriptor_t probsDesc,
                                                             miopenTensorDescriptor_t gradientsDesc,
                                                             const int* labels,
                                                             const int* labelLengths,
                                                             const int* inputLengths,
                                                             miopenCTCLossAlgo_t algo,
                                                             miopenCTCLossDescriptor_t ctcLossDesc,
                                                             size_t* workSpaceSize);
extern "C" miopenStatus_t miopenCTCLoss_impl(miopenHandle_t handle,
                                             miopenTensorDescriptor_t probsDesc,
                                             const void* probs,
                                             const int* labels,
                                             const int* labelLengths,
                                             const int* inputLengths,
                                             void* losses,
                                             miopenTensorDescriptor_t gradientsDesc,
                                             void* gradients,
                                             miopenCTCLossAlgo_t algo,
                                             miopenCTCLossDescriptor_t ctcLossDesc,
                                             void* workSpace,
                                             size_t workSpaceSize);
extern "C" miopenStatus_t
miopenCreateDropoutDescriptor_impl(miopenDropoutDescriptor_t* dropoutDesc);
extern "C" miopenStatus_t
miopenDestroyDropoutDescriptor_impl(miopenDropoutDescriptor_t dropoutDesc);
extern "C" miopenStatus_t miopenDropoutGetReserveSpaceSize_impl(miopenTensorDescriptor_t xDesc,
                                                                size_t* reserveSpaceSizeInBytes);
extern "C" miopenStatus_t miopenDropoutGetStatesSize_impl(miopenHandle_t handle,
                                                          size_t* stateSizeInBytes);
extern "C" miopenStatus_t miopenGetDropoutDescriptor_impl(miopenDropoutDescriptor_t dropoutDesc,
                                                          miopenHandle_t handle,
                                                          float* dropout,
                                                          void** states,
                                                          unsigned long long* seed,
                                                          bool* use_mask,
                                                          bool* state_evo,
                                                          miopenRNGType_t* rng_mode);
extern "C" miopenStatus_t miopenRestoreDropoutDescriptor_impl(miopenDropoutDescriptor_t dropoutDesc,
                                                              miopenHandle_t handle,
                                                              float dropout,
                                                              void* states,
                                                              size_t stateSizeInBytes,
                                                              unsigned long long seed,
                                                              bool use_mask,
                                                              bool state_evo,
                                                              miopenRNGType_t rng_mode);
extern "C" miopenStatus_t miopenSetDropoutDescriptor_impl(miopenDropoutDescriptor_t dropoutDesc,
                                                          miopenHandle_t handle,
                                                          float dropout,
                                                          void* states,
                                                          size_t stateSizeInBytes,
                                                          unsigned long long seed,
                                                          bool use_mask,
                                                          bool state_evo,
                                                          miopenRNGType_t rng_mode);
extern "C" miopenStatus_t miopenDropoutForward_impl(miopenHandle_t handle,
                                                    miopenDropoutDescriptor_t dropoutDesc,
                                                    miopenTensorDescriptor_t noise_shape,
                                                    miopenTensorDescriptor_t xDesc,
                                                    const void* x,
                                                    miopenTensorDescriptor_t yDesc,
                                                    void* y,
                                                    void* reserveSpace,
                                                    size_t reserveSpaceSizeInBytes);
extern "C" miopenStatus_t miopenDropoutBackward_impl(miopenHandle_t handle,
                                                     miopenDropoutDescriptor_t dropoutDesc,
                                                     miopenTensorDescriptor_t noise_shape,
                                                     miopenTensorDescriptor_t dyDesc,
                                                     const void* dy,
                                                     miopenTensorDescriptor_t dxDesc,
                                                     void* dx,
                                                     void* reserveSpace,
                                                     size_t reserveSpaceSizeInBytes);
extern "C" miopenStatus_t
miopenCreateReduceTensorDescriptor_impl(miopenReduceTensorDescriptor_t* reduceTensorDesc);
extern "C" miopenStatus_t
miopenDestroyReduceTensorDescriptor_impl(miopenReduceTensorDescriptor_t reduceTensorDesc);
extern "C" miopenStatus_t
miopenSetReduceTensorDescriptor_impl(miopenReduceTensorDescriptor_t reduceTensorDesc,
                                     miopenReduceTensorOp_t reduceTensorOp,
                                     miopenDataType_t reduceTensorCompType,
                                     miopenNanPropagation_t reduceTensorNanOpt,
                                     miopenReduceTensorIndices_t reduceTensorIndices,
                                     miopenIndicesType_t reduceTensorIndicesType);
extern "C" miopenStatus_t
miopenGetReduceTensorDescriptor_impl(miopenReduceTensorDescriptor_t reduceTensorDesc,
                                     miopenReduceTensorOp_t* reduceTensorOp,
                                     miopenDataType_t* reduceTensorCompType,
                                     miopenNanPropagation_t* reduceTensorNanOpt,
                                     miopenReduceTensorIndices_t* reduceTensorIndices,
                                     miopenIndicesType_t* reduceTensorIndicesType);
extern "C" miopenStatus_t
miopenGetReductionIndicesSize_impl(miopenHandle_t handle,
                                   miopenReduceTensorDescriptor_t reduceTensorDesc,
                                   miopenTensorDescriptor_t aDesc,
                                   miopenTensorDescriptor_t cDesc,
                                   size_t* sizeInBytes);
extern "C" miopenStatus_t
miopenGetReductionWorkspaceSize_impl(miopenHandle_t handle,
                                     miopenReduceTensorDescriptor_t reduceTensorDesc,
                                     miopenTensorDescriptor_t aDesc,
                                     miopenTensorDescriptor_t cDesc,
                                     size_t* sizeInBytes);
extern "C" miopenStatus_t miopenReduceTensor_impl(miopenHandle_t handle,
                                                  miopenReduceTensorDescriptor_t reduceTensorDesc,
                                                  void* indices,
                                                  size_t indicesSizeInBytes,
                                                  void* workspace,
                                                  size_t workspaceSizeInBytes,
                                                  const void* alpha,
                                                  miopenTensorDescriptor_t aDesc,
                                                  const void* A,
                                                  const void* beta,
                                                  miopenTensorDescriptor_t cDesc,
                                                  void* C);
extern "C" miopenStatus_t miopenCreateConvProblem_impl(miopenProblem_t* problem,
                                                       miopenConvolutionDescriptor_t operatorDesc,
                                                       miopenProblemDirection_t direction);
extern "C" miopenStatus_t miopenCreateMhaProblem_impl(miopenProblem_t* problem,
                                                      miopenMhaDescriptor_t operatorDesc,
                                                      miopenProblemDirection_t direction);
extern "C" miopenStatus_t miopenCreateMhaDescriptor_impl(miopenMhaDescriptor_t* mhaDesc);
extern "C" miopenStatus_t miopenSetMhaDescriptor_impl(miopenMhaDescriptor_t mhaDesc, float scale);
extern "C" miopenStatus_t miopenGetMhaDescriptor_impl(miopenMhaDescriptor_t mhaDesc, float* scale);
extern "C" miopenStatus_t
miopenCreateSoftmaxDescriptor_impl(miopenSoftmaxDescriptor_t* softmaxDesc);
extern "C" miopenStatus_t miopenSetSoftmaxDescriptor_impl(miopenSoftmaxDescriptor_t softmaxDesc,
                                                          float alpha,
                                                          float beta,
                                                          miopenSoftmaxAlgorithm_t algorithm,
                                                          miopenSoftmaxMode_t mode);
extern "C" miopenStatus_t miopenGetSoftmaxDescriptor_impl(miopenSoftmaxDescriptor_t softmaxDesc,
                                                          float* alpha,
                                                          float* beta,
                                                          miopenSoftmaxAlgorithm_t* algorithm,
                                                          miopenSoftmaxMode_t* mode);
extern "C" miopenStatus_t miopenDestroyProblem_impl(miopenProblem_t problem);
extern "C" miopenStatus_t miopenSetProblemTensorDescriptor_impl(
    miopenProblem_t problem, miopenTensorArgumentId_t id, miopenTensorDescriptor_t descriptor);
extern "C" miopenStatus_t miopenCreateFindOptions_impl(miopenFindOptions_t* options);
extern "C" miopenStatus_t miopenDestroyFindOptions_impl(miopenFindOptions_t options);
extern "C" miopenStatus_t miopenSetFindOptionTuning_impl(miopenFindOptions_t options, int value);
extern "C" miopenStatus_t miopenSetFindOptionResultsOrder_impl(miopenFindOptions_t options,
                                                               miopenFindResultsOrder_t value);
extern "C" miopenStatus_t miopenSetFindOptionWorkspaceLimit_impl(miopenFindOptions_t options,
                                                                 size_t value);
extern "C" miopenStatus_t miopenSetFindOptionPreallocatedWorkspace_impl(miopenFindOptions_t options,
                                                                        void* buffer,
                                                                        size_t size);
extern "C" miopenStatus_t miopenSetFindOptionPreallocatedTensor_impl(miopenFindOptions_t options,
                                                                     miopenTensorArgumentId_t id,
                                                                     void* buffer);
extern "C" miopenStatus_t miopenSetFindOptionAttachBinaries_impl(miopenFindOptions_t options,
                                                                 unsigned attach);
extern "C" miopenStatus_t miopenFindSolutions_impl(miopenHandle_t handle,
                                                   miopenProblem_t problem,
                                                   miopenFindOptions_t options,
                                                   miopenSolution_t* solutions,
                                                   size_t* numSolutions,
                                                   size_t maxSolutions);
extern "C" miopenStatus_t miopenRunSolution_impl(miopenHandle_t handle,
                                                 miopenSolution_t solution,
                                                 size_t nInputs,
                                                 const miopenTensorArgument_t* tensors,
                                                 void* workspace,
                                                 size_t workspaceSize);
extern "C" miopenStatus_t miopenDestroySolution_impl(miopenSolution_t solution);
extern "C" miopenStatus_t
miopenLoadSolution_impl(miopenSolution_t* solution, const char* data, size_t size);
extern "C" miopenStatus_t miopenSaveSolution_impl(miopenSolution_t solution, char* data);
extern "C" miopenStatus_t miopenGetSolutionSize_impl(miopenSolution_t solution, size_t* size);
extern "C" miopenStatus_t miopenGetSolutionWorkspaceSize_impl(miopenSolution_t solution,
                                                              size_t* workspaceSize);
extern "C" miopenStatus_t miopenGetSolutionTime_impl(miopenSolution_t solution, float* time);
extern "C" miopenStatus_t miopenGetSolutionSolverId_impl(miopenSolution_t solution,
                                                         uint64_t* solverId);
extern "C" miopenStatus_t miopenGetSolverIdConvAlgorithm_impl(uint64_t solverId,
                                                              miopenConvAlgorithm_t* result);
extern "C" miopenStatus_t
miopenCreateActivationProblem_impl(miopenProblem_t* problem,
                                   miopenActivationDescriptor_t operatorDesc,
                                   miopenProblemDirection_t direction);
extern "C" miopenStatus_t miopenCreateBatchnormProblem_impl(miopenProblem_t* problem,
                                                            miopenBatchNormMode_t mode,
                                                            bool runningMeanVariance,
                                                            miopenProblemDirection_t direction);
extern "C" miopenStatus_t miopenFuseProblems_impl(miopenProblem_t problem1,
                                                  miopenProblem_t problem2);
extern "C" miopenStatus_t miopenCreateBiasProblem_impl(miopenProblem_t* problem,
                                                       miopenProblemDirection_t direction);
extern "C" miopenStatus_t miopenCreateSoftmaxProblem_impl(miopenProblem_t* problem,
                                                          miopenSoftmaxDescriptor_t operatorDesc,
                                                          miopenProblemDirection_t direction);
extern "C" miopenStatus_t
miopenGetReduceCalculationWorkspaceSize_impl(miopenHandle_t handle,
                                             miopenTensorDescriptor_t xDesc,
                                             int32_t dim,
                                             miopenReduceCalculationOp_t reduceCalculationOp,
                                             miopenTensorDescriptor_t reduceDesc,
                                             size_t* sizeInBytes);
extern "C" miopenStatus_t
miopenReduceCalculationForward_impl(miopenHandle_t handle,
                                    miopenReduceCalculationNanPropagation_t nanPropagation,
                                    void* workspace,
                                    size_t workspaceSizeInBytes,
                                    miopenTensorDescriptor_t xDesc,
                                    const void* x,
                                    int32_t dim,
                                    miopenReduceCalculationOp_t reduceCalculationOp,
                                    miopenTensorDescriptor_t reduceDesc,
                                    void* y);
extern "C" miopenStatus_t miopenReduceExtremeForward_impl(miopenHandle_t handle,
                                                          miopenTensorDescriptor_t xDesc,
                                                          const void* x,
                                                          int32_t dim,
                                                          miopenReduceExtremeOp_t reduceExtremeOp,
                                                          miopenTensorDescriptor_t yDesc,
                                                          void* y,
                                                          miopenTensorDescriptor_t indiceDesc,
                                                          void* indice);
extern "C" miopenStatus_t miopenGroupNormForward_impl(miopenHandle_t handle,
                                                      miopenNormMode_t mode,
                                                      miopenTensorDescriptor_t xDesc,
                                                      const void* x,
                                                      miopenTensorDescriptor_t weightDesc,
                                                      const void* weight,
                                                      miopenTensorDescriptor_t biasDesc,
                                                      const void* bias,
                                                      uint64_t num_groups,
                                                      float epsilon,
                                                      miopenTensorDescriptor_t yDesc,
                                                      void* y,
                                                      miopenTensorDescriptor_t meanDesc,
                                                      void* mean,
                                                      miopenTensorDescriptor_t rstdDesc,
                                                      void* rstd);
extern "C" miopenStatus_t miopenAddLayerNormForward_impl(miopenHandle_t handle,
                                                         miopenNormMode_t mode,
                                                         miopenTensorDescriptor_t xDesc,
                                                         const void* x,
                                                         miopenTensorDescriptor_t x2Desc,
                                                         const void* x2,
                                                         miopenTensorDescriptor_t weightDesc,
                                                         const void* weight,
                                                         miopenTensorDescriptor_t biasDesc,
                                                         const void* bias,
                                                         float epsilon,
                                                         int32_t normalized_dim,
                                                         miopenTensorDescriptor_t yDesc,
                                                         void* y,
                                                         miopenTensorDescriptor_t meanDesc,
                                                         void* mean,
                                                         miopenTensorDescriptor_t rstdDesc,
                                                         void* rstd);
extern "C" miopenStatus_t miopenT5LayerNormForward_impl(miopenHandle_t handle,
                                                        miopenNormMode_t mode,
                                                        miopenTensorDescriptor_t xDesc,
                                                        const void* x,
                                                        miopenTensorDescriptor_t weightDesc,
                                                        const void* weight,
                                                        float epsilon,
                                                        miopenTensorDescriptor_t yDesc,
                                                        void* y,
                                                        miopenTensorDescriptor_t rstdDesc,
                                                        void* rstd);
extern "C" miopenStatus_t
miopenGetT5LayerNormBackwardWorkspaceSize_impl(miopenHandle_t handle,
                                               miopenNormMode_t mode,
                                               miopenTensorDescriptor_t dyDesc,
                                               miopenTensorDescriptor_t xDesc,
                                               miopenTensorDescriptor_t weightDesc,
                                               miopenTensorDescriptor_t rstdDesc,
                                               miopenTensorDescriptor_t dxDesc,
                                               miopenTensorDescriptor_t dwDesc,
                                               size_t* sizeInBytes);
extern "C" miopenStatus_t miopenT5LayerNormBackward_impl(miopenHandle_t handle,
                                                         miopenNormMode_t mode,
                                                         void* workspace,
                                                         size_t workspaceSizeInBytes,
                                                         miopenTensorDescriptor_t dyDesc,
                                                         const void* dy,
                                                         miopenTensorDescriptor_t xDesc,
                                                         const void* x,
                                                         miopenTensorDescriptor_t weightDesc,
                                                         const void* weight,
                                                         miopenTensorDescriptor_t rstdDesc,
                                                         const void* rstd,
                                                         miopenTensorDescriptor_t dxDesc,
                                                         void* dx,
                                                         miopenTensorDescriptor_t dwDesc,
                                                         void* dw);
extern "C" miopenStatus_t miopenFusedAdam_impl(miopenHandle_t handle,
                                               miopenTensorDescriptor_t paramDesc,
                                               void* param,
                                               miopenTensorDescriptor_t gradDesc,
                                               const void* grad,
                                               miopenTensorDescriptor_t expAvgDesc,
                                               void* expAvg,
                                               miopenTensorDescriptor_t expAvgSqDesc,
                                               void* expAvgSq,
                                               miopenTensorDescriptor_t maxExpAvgSqDesc,
                                               void* maxExpAvgSq,
                                               miopenTensorDescriptor_t stateStepDesc,
                                               void* stateStep,
                                               unsigned int state_step,
                                               float lr,
                                               float beta1,
                                               float beta2,
                                               float weight_decay,
                                               float eps,
                                               bool amsgrad,
                                               bool maximize,
                                               bool adamw,
                                               miopenTensorDescriptor_t gradScaleDesc,
                                               const void* gradScale,
                                               miopenTensorDescriptor_t foundInfDesc,
                                               const void* foundInf);
extern "C" miopenStatus_t
miopenFusedAdamWithOutput_impl(miopenHandle_t handle,
                               miopenTensorDescriptor_t paramInDesc,
                               void* paramIn,
                               miopenTensorDescriptor_t paramOutDesc,
                               void* paramOut,
                               miopenTensorDescriptor_t paramOutFloat16Desc,
                               void* paramOutFloat16,
                               miopenTensorDescriptor_t gradInDesc,
                               const void* gradIn,
                               miopenTensorDescriptor_t expAvgInDesc,
                               void* expAvgIn,
                               miopenTensorDescriptor_t expAvgOutDesc,
                               void* expAvgOut,
                               miopenTensorDescriptor_t expAvgSqInDesc,
                               void* expAvgSqIn,
                               miopenTensorDescriptor_t expAvgSqOutDesc,
                               void* expAvgSqOut,
                               miopenTensorDescriptor_t maxExpAvgSqInDesc,
                               void* maxExpAvgSqIn,
                               miopenTensorDescriptor_t maxExpAvgSqOutDesc,
                               void* maxExpAvgSqOut,
                               miopenTensorDescriptor_t stateStepInDesc,
                               void* stateStepIn,
                               miopenTensorDescriptor_t stateStepOutDesc,
                               void* stateStepOut,
                               unsigned int state_step,
                               float lr,
                               float beta1,
                               float beta2,
                               float weight_decay,
                               float eps,
                               bool amsgrad,
                               bool maximize,
                               bool adamw,
                               miopenTensorDescriptor_t gradScaleDesc,
                               const void* gradScale,
                               miopenTensorDescriptor_t foundInfDesc,
                               const void* foundInf);
extern "C" miopenStatus_t miopenTransformersAdamW_impl(miopenHandle_t handle,
                                                       miopenTensorDescriptor_t paramDesc,
                                                       void* param,
                                                       miopenTensorDescriptor_t gradDesc,
                                                       const void* grad,
                                                       miopenTensorDescriptor_t expAvgDesc,
                                                       void* expAvg,
                                                       miopenTensorDescriptor_t expAvgSqDesc,
                                                       void* expAvgSq,
                                                       miopenTensorDescriptor_t stateStepDesc,
                                                       void* stateStep,
                                                       unsigned int state_step,
                                                       float lr,
                                                       float beta1,
                                                       float beta2,
                                                       float weight_decay,
                                                       float eps,
                                                       bool correct_bias,
                                                       miopenTensorDescriptor_t gradScaleDesc,
                                                       const void* gradScale,
                                                       miopenTensorDescriptor_t foundInfDesc,
                                                       const void* foundInf);
extern "C" miopenStatus_t
miopenTransformersAdamWWithOutput_impl(miopenHandle_t handle,
                                       miopenTensorDescriptor_t paramInDesc,
                                       void* paramIn,
                                       miopenTensorDescriptor_t paramOutDesc,
                                       void* paramOut,
                                       miopenTensorDescriptor_t paramOutFloat16Desc,
                                       void* paramOutFloat16,
                                       miopenTensorDescriptor_t gradInDesc,
                                       const void* gradIn,
                                       miopenTensorDescriptor_t expAvgInDesc,
                                       void* expAvgIn,
                                       miopenTensorDescriptor_t expAvgOutDesc,
                                       void* expAvgOut,
                                       miopenTensorDescriptor_t expAvgSqInDesc,
                                       void* expAvgSqIn,
                                       miopenTensorDescriptor_t expAvgSqOutDesc,
                                       void* expAvgSqOut,
                                       miopenTensorDescriptor_t stateStepInDesc,
                                       void* stateStepIn,
                                       miopenTensorDescriptor_t stateStepOutDesc,
                                       void* stateStepOut,
                                       unsigned int state_step,
                                       float lr,
                                       float beta1,
                                       float beta2,
                                       float weight_decay,
                                       float eps,
                                       float step_size,
                                       bool correct_bias,
                                       miopenTensorDescriptor_t gradScaleDesc,
                                       const void* gradScale,
                                       miopenTensorDescriptor_t foundInfDesc,
                                       const void* foundInf);
extern "C" miopenStatus_t
miopenGetGetitemWorkspaceSize_impl(miopenHandle_t handle,
                                   uint32_t indexCount,
                                   const miopenTensorDescriptor_t* indexDescs,
                                   size_t* sizeInBytes);
extern "C" miopenStatus_t miopenGetitemBackward_impl(miopenHandle_t handle,
                                                     void* workspace,
                                                     size_t workspaceSizeInBytes,
                                                     miopenTensorDescriptor_t dyDesc,
                                                     const void* dy,
                                                     uint32_t indexCount,
                                                     const miopenTensorDescriptor_t* indexDescs,
                                                     const void* const* indexs,
                                                     miopenTensorDescriptor_t dxDesc,
                                                     void* dx,
                                                     miopenTensorDescriptor_t errorDesc,
                                                     void* error,
                                                     uint32_t dimCount,
                                                     const int32_t* dims,
                                                     uint32_t sliceCount,
                                                     const int32_t* slices,
                                                     uint32_t offset);
extern "C" miopenStatus_t miopenRoPEForward_impl(miopenHandle_t handle,
                                                 miopenTensorDescriptor_t xDesc,
                                                 const void* x,
                                                 miopenTensorDescriptor_t cosDesc,
                                                 const void* cos,
                                                 miopenTensorDescriptor_t sinDesc,
                                                 const void* sin,
                                                 miopenTensorDescriptor_t yDesc,
                                                 void* y);
extern "C" miopenStatus_t miopenRoPEBackward_impl(miopenHandle_t handle,
                                                  miopenTensorDescriptor_t dyDesc,
                                                  const void* dy,
                                                  miopenTensorDescriptor_t cosDesc,
                                                  const void* cos,
                                                  miopenTensorDescriptor_t sinDesc,
                                                  const void* sin,
                                                  miopenTensorDescriptor_t dxDesc,
                                                  void* dx);
extern "C" miopenStatus_t miopenKthvalueForward_impl(miopenHandle_t handle,
                                                     miopenTensorDescriptor_t inputDesc,
                                                     const void* input,
                                                     miopenTensorDescriptor_t outputDesc,
                                                     void* output,
                                                     miopenTensorDescriptor_t indicesDesc,
                                                     size_t* indices,
                                                     size_t k,
                                                     int32_t dim  = -1,
                                                     bool keepDim = false);
extern "C" miopenStatus_t
miopenGetPReLUBackwardWorkspaceSize_impl(miopenHandle_t handle,
                                         miopenTensorDescriptor_t inputDesc,
                                         miopenTensorDescriptor_t weightDesc,
                                         size_t* sizeInBytes);
extern "C" miopenStatus_t miopenPReLUBackward_impl(miopenHandle_t handle,
                                                   void* workspace,
                                                   size_t workspaceSizeInBytes,
                                                   miopenTensorDescriptor_t inputDesc,
                                                   const void* input,
                                                   miopenTensorDescriptor_t weightDesc,
                                                   const void* weight,
                                                   miopenTensorDescriptor_t doutputDesc,
                                                   const void* doutput,
                                                   miopenTensorDescriptor_t dinputDesc,
                                                   void* dinput,
                                                   miopenTensorDescriptor_t dweightDesc,
                                                   void* dweight);
extern "C" miopenStatus_t
miopenGetSoftMarginLossForwardWorkspaceSize_impl(miopenHandle_t handle,
                                                 miopenTensorDescriptor_t inputDesc,
                                                 miopenTensorDescriptor_t targetDesc,
                                                 miopenTensorDescriptor_t outputDesc,
                                                 miopenLossReductionMode_t reduction,
                                                 size_t* sizeInBytes);
extern "C" miopenStatus_t miopenSoftMarginLossForward_impl(miopenHandle_t handle,
                                                           miopenTensorDescriptor_t inputDesc,
                                                           const void* input,
                                                           miopenTensorDescriptor_t targetDesc,
                                                           const void* target,
                                                           miopenTensorDescriptor_t outputDesc,
                                                           void* output,
                                                           miopenLossReductionMode_t reduction,
                                                           void* workspace             = nullptr,
                                                           size_t workspaceSizeInBytes = 0);
extern "C" miopenStatus_t miopenSoftMarginLossBackward_impl(miopenHandle_t handle,
                                                            miopenTensorDescriptor_t inputDesc,
                                                            const void* input,
                                                            miopenTensorDescriptor_t targetDesc,
                                                            const void* target,
                                                            miopenTensorDescriptor_t doutputDesc,
                                                            const void* doutput,
                                                            miopenTensorDescriptor_t dinputDesc,
                                                            void* dinput,
                                                            miopenLossReductionMode_t reduction);
extern "C" miopenStatus_t
miopenGetMultiMarginLossForwardWorkspaceSize_impl(miopenHandle_t handle,
                                                  miopenTensorDescriptor_t inputDesc,
                                                  miopenTensorDescriptor_t targetDesc,
                                                  miopenTensorDescriptor_t weightDesc,
                                                  miopenTensorDescriptor_t outputDesc,
                                                  long p,
                                                  float margin,
                                                  miopenLossReductionMode_t reduction,
                                                  size_t* sizeInBytes);
extern "C" miopenStatus_t miopenMultiMarginLossForward_impl(miopenHandle_t handle,
                                                            miopenTensorDescriptor_t inputDesc,
                                                            const void* input,
                                                            miopenTensorDescriptor_t targetDesc,
                                                            const void* target,
                                                            miopenTensorDescriptor_t weightDesc,
                                                            const void* weight,
                                                            miopenTensorDescriptor_t outputDesc,
                                                            void* output,
                                                            long p,
                                                            float margin,
                                                            miopenLossReductionMode_t reduction,
                                                            void* workspace,
                                                            size_t workspaceSizeInBytes);
extern "C" miopenStatus_t miopenSetTuningPolicy_impl(miopenHandle_t handle,
                                                     miopenTuningPolicy_t newValue);
extern "C" miopenStatus_t miopenGetTuningPolicy_impl(miopenHandle_t handle,
                                                     miopenTuningPolicy_t* value);

extern "C" const char* miopenGetErrorString(miopenStatus_t error)
{
    // miopenGetErrorString returns a string, not a status, and is never in the
    // forwarding set; consult the dispatcher only so the configuration banner is
    // emitted if this is the first wrapped call in the process.
    (void)miopen::wrapper::Dispatch("miopenGetErrorString");
    return miopenGetErrorString_impl(error);
}

extern "C" miopenStatus_t miopenGetVersion(size_t* major, size_t* minor, size_t* patch)
{
    if(miopen::wrapper::Dispatch("miopenGetVersion") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetVersion");
    return miopenGetVersion_impl(major, minor, patch);
}

// clang-format off
// Keep this stub multi-line: investigation_q4_stub_count CTest counts `{` on
// column 0 to enforce stub/header parity. See tools/wrapper/check_stub_count.cmake.
extern "C" miopenStatus_t miopenCreate(miopenHandle_t* handle)
{
    if(miopen::wrapper::Dispatch("miopenCreate") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenCreate");
    return miopenCreate_impl(handle);
}
// clang-format on

extern "C" miopenStatus_t miopenCreateWithStream(miopenHandle_t* handle,
                                                 miopenAcceleratorQueue_t stream)
{
    if(miopen::wrapper::Dispatch("miopenCreateWithStream") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenCreateWithStream");
    return miopenCreateWithStream_impl(handle, stream);
}

extern "C" miopenStatus_t miopenDestroy(miopenHandle_t handle)
{
    if(miopen::wrapper::Dispatch("miopenDestroy") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenDestroy");
    return miopenDestroy_impl(handle);
}

extern "C" miopenStatus_t miopenSetStream(miopenHandle_t handle, miopenAcceleratorQueue_t streamID)
{
    if(miopen::wrapper::Dispatch("miopenSetStream") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetStream");
    return miopenSetStream_impl(handle, streamID);
}

extern "C" miopenStatus_t miopenGetStream(miopenHandle_t handle, miopenAcceleratorQueue_t* streamID)
{
    if(miopen::wrapper::Dispatch("miopenGetStream") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetStream");
    return miopenGetStream_impl(handle, streamID);
}

extern "C" miopenStatus_t miopenSetAllocator(miopenHandle_t handle,
                                             miopenAllocatorFunction allocator,
                                             miopenDeallocatorFunction deallocator,
                                             void* allocatorContext)
{
    if(miopen::wrapper::Dispatch("miopenSetAllocator") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetAllocator");
    return miopenSetAllocator_impl(handle, allocator, deallocator, allocatorContext);
}

extern "C" miopenStatus_t miopenGetKernelTime(miopenHandle_t handle, float* time)
{
    if(miopen::wrapper::Dispatch("miopenGetKernelTime") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetKernelTime");
    return miopenGetKernelTime_impl(handle, time);
}

extern "C" miopenStatus_t miopenEnableProfiling(miopenHandle_t handle, bool enable)
{
    if(miopen::wrapper::Dispatch("miopenEnableProfiling") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenEnableProfiling");
    return miopenEnableProfiling_impl(handle, enable);
}

extern "C" miopenStatus_t miopenCreateTensorDescriptor(miopenTensorDescriptor_t* tensorDesc)
{
    if(miopen::wrapper::Dispatch("miopenCreateTensorDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenCreateTensorDescriptor");
    return miopenCreateTensorDescriptor_impl(tensorDesc);
}

extern "C" miopenStatus_t miopenSet4dTensorDescriptor(
    miopenTensorDescriptor_t tensorDesc, miopenDataType_t dataType, int n, int c, int h, int w)
{
    if(miopen::wrapper::Dispatch("miopenSet4dTensorDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSet4dTensorDescriptor");
    return miopenSet4dTensorDescriptor_impl(tensorDesc, dataType, n, c, h, w);
}

extern "C" miopenStatus_t miopenSetNdTensorDescriptorWithLayout(miopenTensorDescriptor_t tensorDesc,
                                                                miopenDataType_t dataType,
                                                                miopenTensorLayout_t tensorLayout,
                                                                const int* lens,
                                                                int num_lens)
{
    if(miopen::wrapper::Dispatch("miopenSetNdTensorDescriptorWithLayout") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetNdTensorDescriptorWithLayout");
    return miopenSetNdTensorDescriptorWithLayout_impl(
        tensorDesc, dataType, tensorLayout, lens, num_lens);
}

extern "C" miopenStatus_t miopenSet4dTensorDescriptorEx(miopenTensorDescriptor_t tensorDesc,
                                                        miopenDataType_t dataType,
                                                        int n,
                                                        int c,
                                                        int h,
                                                        int w,
                                                        int nStride,
                                                        int cStride,
                                                        int hStride,
                                                        int wStride)
{
    if(miopen::wrapper::Dispatch("miopenSet4dTensorDescriptorEx") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSet4dTensorDescriptorEx");
    return miopenSet4dTensorDescriptorEx_impl(
        tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride);
}

extern "C" miopenStatus_t miopenGet4dTensorDescriptor(miopenTensorDescriptor_t tensorDesc,
                                                      miopenDataType_t* dataType,
                                                      int* n,
                                                      int* c,
                                                      int* h,
                                                      int* w,
                                                      int* nStride,
                                                      int* cStride,
                                                      int* hStride,
                                                      int* wStride)
{
    if(miopen::wrapper::Dispatch("miopenGet4dTensorDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGet4dTensorDescriptor");
    return miopenGet4dTensorDescriptor_impl(
        tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride);
}

extern "C" miopenStatus_t miopenSetTensorDescriptor(miopenTensorDescriptor_t tensorDesc,
                                                    miopenDataType_t dataType,
                                                    int nbDims,
                                                    const int* dimsA,
                                                    const int* stridesA)
{
    if(miopen::wrapper::Dispatch("miopenSetTensorDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetTensorDescriptor");
    return miopenSetTensorDescriptor_impl(tensorDesc, dataType, nbDims, dimsA, stridesA);
}

extern "C" miopenStatus_t miopenSetTensorDescriptorV2(miopenTensorDescriptor_t tensorDesc,
                                                      miopenDataType_t dataType,
                                                      int nbDims,
                                                      const size_t* dimsA,
                                                      const size_t* stridesA)
{
    if(miopen::wrapper::Dispatch("miopenSetTensorDescriptorV2") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetTensorDescriptorV2");
    return miopenSetTensorDescriptorV2_impl(tensorDesc, dataType, nbDims, dimsA, stridesA);
}

extern "C" miopenStatus_t miopenSetTensorCastType(miopenTensorDescriptor_t tensorDesc,
                                                  miopenDataType_t cast_type)
{
    if(miopen::wrapper::Dispatch("miopenSetTensorCastType") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetTensorCastType");
    return miopenSetTensorCastType_impl(tensorDesc, cast_type);
}

extern "C" miopenStatus_t miopenGetTensorDescriptorSize(miopenTensorDescriptor_t tensorDesc,
                                                        int* size)
{
    if(miopen::wrapper::Dispatch("miopenGetTensorDescriptorSize") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetTensorDescriptorSize");
    return miopenGetTensorDescriptorSize_impl(tensorDesc, size);
}

extern "C" miopenStatus_t miopenGetTensorDescriptor(miopenTensorDescriptor_t tensorDesc,
                                                    miopenDataType_t* dataType,
                                                    int* dimsA,
                                                    int* stridesA)
{
    if(miopen::wrapper::Dispatch("miopenGetTensorDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetTensorDescriptor");
    return miopenGetTensorDescriptor_impl(tensorDesc, dataType, dimsA, stridesA);
}

extern "C" miopenStatus_t miopenDestroyTensorDescriptor(miopenTensorDescriptor_t tensorDesc)
{
    if(miopen::wrapper::Dispatch("miopenDestroyTensorDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenDestroyTensorDescriptor");
    return miopenDestroyTensorDescriptor_impl(tensorDesc);
}

extern "C" miopenStatus_t miopenCreateSeqTensorDescriptor(miopenSeqTensorDescriptor_t* tensorDesc)
{
    if(miopen::wrapper::Dispatch("miopenCreateSeqTensorDescriptor") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenCreateSeqTensorDescriptor");
    return miopenCreateSeqTensorDescriptor_impl(tensorDesc);
}

extern "C" miopenStatus_t miopenDestroySeqTensorDescriptor(miopenSeqTensorDescriptor_t tensorDesc)
{
    if(miopen::wrapper::Dispatch("miopenDestroySeqTensorDescriptor") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenDestroySeqTensorDescriptor");
    return miopenDestroySeqTensorDescriptor_impl(tensorDesc);
}

extern "C" miopenStatus_t miopenOpTensor(miopenHandle_t handle,
                                         miopenTensorOp_t tensorOp,
                                         const void* alpha1,
                                         const miopenTensorDescriptor_t aDesc,
                                         const void* A,
                                         const void* alpha2,
                                         const miopenTensorDescriptor_t bDesc,
                                         const void* B,
                                         const void* beta,
                                         const miopenTensorDescriptor_t cDesc,
                                         void* C)
{
    if(miopen::wrapper::Dispatch("miopenOpTensor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenOpTensor");
    return miopenOpTensor_impl(
        handle, tensorOp, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc, C);
}

extern "C" miopenStatus_t miopenSetTensor(miopenHandle_t handle,
                                          const miopenTensorDescriptor_t yDesc,
                                          void* y,
                                          const void* alpha)
{
    if(miopen::wrapper::Dispatch("miopenSetTensor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetTensor");
    return miopenSetTensor_impl(handle, yDesc, y, alpha);
}

extern "C" miopenStatus_t miopenScaleTensor(miopenHandle_t handle,
                                            const miopenTensorDescriptor_t yDesc,
                                            void* y,
                                            const void* alpha)
{
    if(miopen::wrapper::Dispatch("miopenScaleTensor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenScaleTensor");
    return miopenScaleTensor_impl(handle, yDesc, y, alpha);
}

extern "C" miopenStatus_t miopenGetTensorNumBytes(miopenTensorDescriptor_t tensorDesc,
                                                  size_t* numBytes)
{
    if(miopen::wrapper::Dispatch("miopenGetTensorNumBytes") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetTensorNumBytes");
    return miopenGetTensorNumBytes_impl(tensorDesc, numBytes);
}

extern "C" miopenStatus_t miopenTransformTensor(miopenHandle_t handle,
                                                const void* alpha,
                                                const miopenTensorDescriptor_t xDesc,
                                                const void* x,
                                                const void* beta,
                                                const miopenTensorDescriptor_t yDesc,
                                                void* y)
{
    if(miopen::wrapper::Dispatch("miopenTransformTensor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenTransformTensor");
    return miopenTransformTensor_impl(handle, alpha, xDesc, x, beta, yDesc, y);
}

extern "C" miopenStatus_t miopenCreateConvolutionDescriptor(miopenConvolutionDescriptor_t* convDesc)
{
    if(miopen::wrapper::Dispatch("miopenCreateConvolutionDescriptor") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenCreateConvolutionDescriptor");
    return miopenCreateConvolutionDescriptor_impl(convDesc);
}

extern "C" miopenStatus_t miopenInitConvolutionDescriptor(miopenConvolutionDescriptor_t convDesc,
                                                          miopenConvolutionMode_t c_mode,
                                                          int pad_h,
                                                          int pad_w,
                                                          int stride_h,
                                                          int stride_w,
                                                          int dilation_h,
                                                          int dilation_w)
{
    if(miopen::wrapper::Dispatch("miopenInitConvolutionDescriptor") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenInitConvolutionDescriptor");
    return miopenInitConvolutionDescriptor_impl(
        convDesc, c_mode, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
}

extern "C" miopenStatus_t miopenInitConvolutionNdDescriptor(miopenConvolutionDescriptor_t convDesc,
                                                            int spatialDim,
                                                            const int* padA,
                                                            const int* strideA,
                                                            const int* dilationA,
                                                            miopenConvolutionMode_t c_mode)
{
    if(miopen::wrapper::Dispatch("miopenInitConvolutionNdDescriptor") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenInitConvolutionNdDescriptor");
    return miopenInitConvolutionNdDescriptor_impl(
        convDesc, spatialDim, padA, strideA, dilationA, c_mode);
}

extern "C" miopenStatus_t miopenGetConvolutionSpatialDim(miopenConvolutionDescriptor_t convDesc,
                                                         int* spatialDim)
{
    if(miopen::wrapper::Dispatch("miopenGetConvolutionSpatialDim") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetConvolutionSpatialDim");
    return miopenGetConvolutionSpatialDim_impl(convDesc, spatialDim);
}

extern "C" miopenStatus_t miopenGetConvolutionDescriptor(miopenConvolutionDescriptor_t convDesc,
                                                         miopenConvolutionMode_t* c_mode,
                                                         int* pad_h,
                                                         int* pad_w,
                                                         int* stride_h,
                                                         int* stride_w,
                                                         int* dilation_h,
                                                         int* dilation_w)
{
    if(miopen::wrapper::Dispatch("miopenGetConvolutionDescriptor") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetConvolutionDescriptor");
    return miopenGetConvolutionDescriptor_impl(
        convDesc, c_mode, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
}

extern "C" miopenStatus_t miopenGetConvolutionNdDescriptor(miopenConvolutionDescriptor_t convDesc,
                                                           int requestedSpatialDim,
                                                           int* spatialDim,
                                                           int* padA,
                                                           int* strideA,
                                                           int* dilationA,
                                                           miopenConvolutionMode_t* c_mode)
{
    if(miopen::wrapper::Dispatch("miopenGetConvolutionNdDescriptor") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetConvolutionNdDescriptor");
    return miopenGetConvolutionNdDescriptor_impl(
        convDesc, requestedSpatialDim, spatialDim, padA, strideA, dilationA, c_mode);
}

extern "C" miopenStatus_t miopenGetConvolutionGroupCount(miopenConvolutionDescriptor_t convDesc,
                                                         int* groupCount)
{
    if(miopen::wrapper::Dispatch("miopenGetConvolutionGroupCount") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetConvolutionGroupCount");
    return miopenGetConvolutionGroupCount_impl(convDesc, groupCount);
}

extern "C" miopenStatus_t miopenSetConvolutionGroupCount(miopenConvolutionDescriptor_t convDesc,
                                                         int groupCount)
{
    if(miopen::wrapper::Dispatch("miopenSetConvolutionGroupCount") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetConvolutionGroupCount");
    return miopenSetConvolutionGroupCount_impl(convDesc, groupCount);
}

extern "C" miopenStatus_t
miopenSetTransposeConvOutputPadding(miopenConvolutionDescriptor_t convDesc, int adj_h, int adj_w)
{
    if(miopen::wrapper::Dispatch("miopenSetTransposeConvOutputPadding") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetTransposeConvOutputPadding");
    return miopenSetTransposeConvOutputPadding_impl(convDesc, adj_h, adj_w);
}

extern "C" miopenStatus_t miopenSetTransposeConvNdOutputPadding(
    miopenConvolutionDescriptor_t convDesc, int spatialDim, const int* adjA)
{
    if(miopen::wrapper::Dispatch("miopenSetTransposeConvNdOutputPadding") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetTransposeConvNdOutputPadding");
    return miopenSetTransposeConvNdOutputPadding_impl(convDesc, spatialDim, adjA);
}

extern "C" miopenStatus_t
miopenGetConvolutionForwardOutputDim(miopenConvolutionDescriptor_t convDesc,
                                     const miopenTensorDescriptor_t inputTensorDesc,
                                     const miopenTensorDescriptor_t filterDesc,
                                     int* n,
                                     int* c,
                                     int* h,
                                     int* w)
{
    if(miopen::wrapper::Dispatch("miopenGetConvolutionForwardOutputDim") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetConvolutionForwardOutputDim");
    return miopenGetConvolutionForwardOutputDim_impl(
        convDesc, inputTensorDesc, filterDesc, n, c, h, w);
}

extern "C" miopenStatus_t
miopenGetConvolutionNdForwardOutputDim(miopenConvolutionDescriptor_t convDesc,
                                       const miopenTensorDescriptor_t inputTensorDesc,
                                       const miopenTensorDescriptor_t filterDesc,
                                       int* nDim,
                                       int* outputTensorDimA)
{
    if(miopen::wrapper::Dispatch("miopenGetConvolutionNdForwardOutputDim") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetConvolutionNdForwardOutputDim");
    return miopenGetConvolutionNdForwardOutputDim_impl(
        convDesc, inputTensorDesc, filterDesc, nDim, outputTensorDimA);
}

extern "C" miopenStatus_t miopenDestroyConvolutionDescriptor(miopenConvolutionDescriptor_t convDesc)
{
    if(miopen::wrapper::Dispatch("miopenDestroyConvolutionDescriptor") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenDestroyConvolutionDescriptor");
    return miopenDestroyConvolutionDescriptor_impl(convDesc);
}

extern "C" miopenStatus_t miopenSetConvolutionAttribute(miopenConvolutionDescriptor_t convDesc,
                                                        const miopenConvolutionAttrib_t attr,
                                                        int value)
{
    if(miopen::wrapper::Dispatch("miopenSetConvolutionAttribute") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetConvolutionAttribute");
    return miopenSetConvolutionAttribute_impl(convDesc, attr, value);
}

extern "C" miopenStatus_t miopenGetConvolutionAttribute(miopenConvolutionDescriptor_t convDesc,
                                                        const miopenConvolutionAttrib_t attr,
                                                        int* value)
{
    if(miopen::wrapper::Dispatch("miopenGetConvolutionAttribute") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetConvolutionAttribute");
    return miopenGetConvolutionAttribute_impl(convDesc, attr, value);
}

extern "C" miopenStatus_t miopenSetConvolutionFindMode(miopenConvolutionDescriptor_t convDesc,
                                                       miopenConvolutionFindMode_t findMode)
{
    if(miopen::wrapper::Dispatch("miopenSetConvolutionFindMode") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetConvolutionFindMode");
    return miopenSetConvolutionFindMode_impl(convDesc, findMode);
}

extern "C" miopenStatus_t miopenGetConvolutionFindMode(const miopenConvolutionDescriptor_t convDesc,
                                                       miopenConvolutionFindMode_t* findMode)
{
    if(miopen::wrapper::Dispatch("miopenGetConvolutionFindMode") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetConvolutionFindMode");
    return miopenGetConvolutionFindMode_impl(convDesc, findMode);
}

extern "C" miopenStatus_t
miopenConvolutionForwardGetSolutionCount(miopenHandle_t handle,
                                         const miopenTensorDescriptor_t wDesc,
                                         const miopenTensorDescriptor_t xDesc,
                                         const miopenConvolutionDescriptor_t convDesc,
                                         const miopenTensorDescriptor_t yDesc,
                                         size_t* solutionCount)
{
    if(miopen::wrapper::Dispatch("miopenConvolutionForwardGetSolutionCount") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenConvolutionForwardGetSolutionCount");
    return miopenConvolutionForwardGetSolutionCount_impl(
        handle, wDesc, xDesc, convDesc, yDesc, solutionCount);
}

extern "C" miopenStatus_t
miopenConvolutionForwardGetSolution(miopenHandle_t handle,
                                    const miopenTensorDescriptor_t wDesc,
                                    const miopenTensorDescriptor_t xDesc,
                                    const miopenConvolutionDescriptor_t convDesc,
                                    const miopenTensorDescriptor_t yDesc,
                                    const size_t maxSolutionCount,
                                    size_t* solutionCount,
                                    miopenConvSolution_t* solutions)
{
    if(miopen::wrapper::Dispatch("miopenConvolutionForwardGetSolution") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenConvolutionForwardGetSolution");
    return miopenConvolutionForwardGetSolution_impl(
        handle, wDesc, xDesc, convDesc, yDesc, maxSolutionCount, solutionCount, solutions);
}

extern "C" miopenStatus_t
miopenConvolutionForwardGetSolutionWorkspaceSize(miopenHandle_t handle,
                                                 const miopenTensorDescriptor_t wDesc,
                                                 const miopenTensorDescriptor_t xDesc,
                                                 const miopenConvolutionDescriptor_t convDesc,
                                                 const miopenTensorDescriptor_t yDesc,
                                                 const uint64_t solution_id,
                                                 size_t* workSpaceSize)
{
    if(miopen::wrapper::Dispatch("miopenConvolutionForwardGetSolutionWorkspaceSize") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenConvolutionForwardGetSolutionWorkspaceSize");
    return miopenConvolutionForwardGetSolutionWorkspaceSize_impl(
        handle, wDesc, xDesc, convDesc, yDesc, solution_id, workSpaceSize);
}

extern "C" miopenStatus_t
miopenConvolutionForwardCompileSolution(miopenHandle_t handle,
                                        const miopenTensorDescriptor_t wDesc,
                                        const miopenTensorDescriptor_t xDesc,
                                        const miopenConvolutionDescriptor_t convDesc,
                                        const miopenTensorDescriptor_t yDesc,
                                        const uint64_t solution_id)
{
    if(miopen::wrapper::Dispatch("miopenConvolutionForwardCompileSolution") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenConvolutionForwardCompileSolution");
    return miopenConvolutionForwardCompileSolution_impl(
        handle, wDesc, xDesc, convDesc, yDesc, solution_id);
}

extern "C" miopenStatus_t
miopenConvolutionForwardImmediate(miopenHandle_t handle,
                                  const miopenTensorDescriptor_t wDesc,
                                  const void* w,
                                  const miopenTensorDescriptor_t xDesc,
                                  const void* x,
                                  const miopenConvolutionDescriptor_t convDesc,
                                  const miopenTensorDescriptor_t yDesc,
                                  void* y,
                                  void* workSpace,
                                  size_t workSpaceSize,
                                  const uint64_t solution_id)
{
    if(miopen::wrapper::Dispatch("miopenConvolutionForwardImmediate") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenConvolutionForwardImmediate");
    return miopenConvolutionForwardImmediate_impl(
        handle, wDesc, w, xDesc, x, convDesc, yDesc, y, workSpace, workSpaceSize, solution_id);
}

extern "C" miopenStatus_t
miopenConvolutionBackwardDataGetSolutionCount(miopenHandle_t handle,
                                              const miopenTensorDescriptor_t dyDesc,
                                              const miopenTensorDescriptor_t wDesc,
                                              const miopenConvolutionDescriptor_t convDesc,
                                              const miopenTensorDescriptor_t dxDesc,
                                              size_t* solutionCount)
{
    if(miopen::wrapper::Dispatch("miopenConvolutionBackwardDataGetSolutionCount") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenConvolutionBackwardDataGetSolutionCount");
    return miopenConvolutionBackwardDataGetSolutionCount_impl(
        handle, dyDesc, wDesc, convDesc, dxDesc, solutionCount);
}

extern "C" miopenStatus_t
miopenConvolutionBackwardDataGetSolution(miopenHandle_t handle,
                                         const miopenTensorDescriptor_t dyDesc,
                                         const miopenTensorDescriptor_t wDesc,
                                         const miopenConvolutionDescriptor_t convDesc,
                                         const miopenTensorDescriptor_t dxDesc,
                                         const size_t maxSolutionCount,
                                         size_t* solutionCount,
                                         miopenConvSolution_t* solutions)
{
    if(miopen::wrapper::Dispatch("miopenConvolutionBackwardDataGetSolution") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenConvolutionBackwardDataGetSolution");
    return miopenConvolutionBackwardDataGetSolution_impl(
        handle, dyDesc, wDesc, convDesc, dxDesc, maxSolutionCount, solutionCount, solutions);
}

extern "C" miopenStatus_t
miopenConvolutionBackwardDataGetSolutionWorkspaceSize(miopenHandle_t handle,
                                                      const miopenTensorDescriptor_t dyDesc,
                                                      const miopenTensorDescriptor_t wDesc,
                                                      const miopenConvolutionDescriptor_t convDesc,
                                                      const miopenTensorDescriptor_t dxDesc,
                                                      const uint64_t solution_id,
                                                      size_t* workSpaceSize)
{
    if(miopen::wrapper::Dispatch("miopenConvolutionBackwardDataGetSolutionWorkspaceSize") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenConvolutionBackwardDataGetSolutionWorkspaceSize");
    return miopenConvolutionBackwardDataGetSolutionWorkspaceSize_impl(
        handle, dyDesc, wDesc, convDesc, dxDesc, solution_id, workSpaceSize);
}

extern "C" miopenStatus_t
miopenConvolutionBackwardDataCompileSolution(miopenHandle_t handle,
                                             const miopenTensorDescriptor_t dyDesc,
                                             const miopenTensorDescriptor_t wDesc,
                                             const miopenConvolutionDescriptor_t convDesc,
                                             const miopenTensorDescriptor_t dxDesc,
                                             const uint64_t solution_id)
{
    if(miopen::wrapper::Dispatch("miopenConvolutionBackwardDataCompileSolution") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenConvolutionBackwardDataCompileSolution");
    return miopenConvolutionBackwardDataCompileSolution_impl(
        handle, dyDesc, wDesc, convDesc, dxDesc, solution_id);
}

extern "C" miopenStatus_t
miopenConvolutionBackwardDataImmediate(miopenHandle_t handle,
                                       const miopenTensorDescriptor_t dyDesc,
                                       const void* dy,
                                       const miopenTensorDescriptor_t wDesc,
                                       const void* w,
                                       const miopenConvolutionDescriptor_t convDesc,
                                       const miopenTensorDescriptor_t dxDesc,
                                       void* dx,
                                       void* workSpace,
                                       size_t workSpaceSize,
                                       const uint64_t solution_id)
{
    if(miopen::wrapper::Dispatch("miopenConvolutionBackwardDataImmediate") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenConvolutionBackwardDataImmediate");
    return miopenConvolutionBackwardDataImmediate_impl(
        handle, dyDesc, dy, wDesc, w, convDesc, dxDesc, dx, workSpace, workSpaceSize, solution_id);
}

extern "C" miopenStatus_t
miopenConvolutionBackwardWeightsGetSolutionCount(miopenHandle_t handle,
                                                 const miopenTensorDescriptor_t dyDesc,
                                                 const miopenTensorDescriptor_t xDesc,
                                                 const miopenConvolutionDescriptor_t convDesc,
                                                 const miopenTensorDescriptor_t dwDesc,
                                                 size_t* solutionCount)
{
    if(miopen::wrapper::Dispatch("miopenConvolutionBackwardWeightsGetSolutionCount") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenConvolutionBackwardWeightsGetSolutionCount");
    return miopenConvolutionBackwardWeightsGetSolutionCount_impl(
        handle, dyDesc, xDesc, convDesc, dwDesc, solutionCount);
}

extern "C" miopenStatus_t
miopenConvolutionBackwardWeightsGetSolution(miopenHandle_t handle,
                                            const miopenTensorDescriptor_t dyDesc,
                                            const miopenTensorDescriptor_t xDesc,
                                            const miopenConvolutionDescriptor_t convDesc,
                                            const miopenTensorDescriptor_t dwDesc,
                                            const size_t maxSolutionCount,
                                            size_t* solutionCount,
                                            miopenConvSolution_t* solutions)
{
    if(miopen::wrapper::Dispatch("miopenConvolutionBackwardWeightsGetSolution") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenConvolutionBackwardWeightsGetSolution");
    return miopenConvolutionBackwardWeightsGetSolution_impl(
        handle, dyDesc, xDesc, convDesc, dwDesc, maxSolutionCount, solutionCount, solutions);
}

extern "C" miopenStatus_t miopenConvolutionBackwardWeightsGetSolutionWorkspaceSize(
    miopenHandle_t handle,
    const miopenTensorDescriptor_t dyDesc,
    const miopenTensorDescriptor_t xDesc,
    const miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t dwDesc,
    const uint64_t solution_id,
    size_t* workSpaceSize)
{
    if(miopen::wrapper::Dispatch("miopenConvolutionBackwardWeightsGetSolutionWorkspaceSize") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenConvolutionBackwardWeightsGetSolutionWorkspaceSize");
    return miopenConvolutionBackwardWeightsGetSolutionWorkspaceSize_impl(
        handle, dyDesc, xDesc, convDesc, dwDesc, solution_id, workSpaceSize);
}

extern "C" miopenStatus_t
miopenConvolutionBackwardWeightsCompileSolution(miopenHandle_t handle,
                                                const miopenTensorDescriptor_t dyDesc,
                                                const miopenTensorDescriptor_t xDesc,
                                                const miopenConvolutionDescriptor_t convDesc,
                                                const miopenTensorDescriptor_t dwDesc,
                                                const uint64_t solution_id)
{
    if(miopen::wrapper::Dispatch("miopenConvolutionBackwardWeightsCompileSolution") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenConvolutionBackwardWeightsCompileSolution");
    return miopenConvolutionBackwardWeightsCompileSolution_impl(
        handle, dyDesc, xDesc, convDesc, dwDesc, solution_id);
}

extern "C" miopenStatus_t
miopenConvolutionBackwardWeightsImmediate(miopenHandle_t handle,
                                          const miopenTensorDescriptor_t dyDesc,
                                          const void* dy,
                                          const miopenTensorDescriptor_t xDesc,
                                          const void* x,
                                          const miopenConvolutionDescriptor_t convDesc,
                                          const miopenTensorDescriptor_t dwDesc,
                                          void* dw,
                                          void* workSpace,
                                          size_t workSpaceSize,
                                          const uint64_t solution_id)
{
    if(miopen::wrapper::Dispatch("miopenConvolutionBackwardWeightsImmediate") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenConvolutionBackwardWeightsImmediate");
    return miopenConvolutionBackwardWeightsImmediate_impl(
        handle, dyDesc, dy, xDesc, x, convDesc, dwDesc, dw, workSpace, workSpaceSize, solution_id);
}

extern "C" miopenStatus_t
miopenConvolutionForwardGetWorkSpaceSize(miopenHandle_t handle,
                                         const miopenTensorDescriptor_t wDesc,
                                         const miopenTensorDescriptor_t xDesc,
                                         const miopenConvolutionDescriptor_t convDesc,
                                         const miopenTensorDescriptor_t yDesc,
                                         size_t* workSpaceSize)
{
    if(miopen::wrapper::Dispatch("miopenConvolutionForwardGetWorkSpaceSize") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenConvolutionForwardGetWorkSpaceSize");
    return miopenConvolutionForwardGetWorkSpaceSize_impl(
        handle, wDesc, xDesc, convDesc, yDesc, workSpaceSize);
}

extern "C" miopenStatus_t
miopenFindConvolutionForwardAlgorithm(miopenHandle_t handle,
                                      const miopenTensorDescriptor_t xDesc,
                                      const void* x,
                                      const miopenTensorDescriptor_t wDesc,
                                      const void* w,
                                      const miopenConvolutionDescriptor_t convDesc,
                                      const miopenTensorDescriptor_t yDesc,
                                      void* y,
                                      const int requestAlgoCount,
                                      int* returnedAlgoCount,
                                      miopenConvAlgoPerf_t* perfResults,
                                      void* workSpace,
                                      size_t workSpaceSize,
                                      bool exhaustiveSearch)
{
    if(miopen::wrapper::Dispatch("miopenFindConvolutionForwardAlgorithm") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenFindConvolutionForwardAlgorithm");
    return miopenFindConvolutionForwardAlgorithm_impl(handle,
                                                      xDesc,
                                                      x,
                                                      wDesc,
                                                      w,
                                                      convDesc,
                                                      yDesc,
                                                      y,
                                                      requestAlgoCount,
                                                      returnedAlgoCount,
                                                      perfResults,
                                                      workSpace,
                                                      workSpaceSize,
                                                      exhaustiveSearch);
}

extern "C" miopenStatus_t miopenConvolutionForward(miopenHandle_t handle,
                                                   const void* alpha,
                                                   const miopenTensorDescriptor_t xDesc,
                                                   const void* x,
                                                   const miopenTensorDescriptor_t wDesc,
                                                   const void* w,
                                                   const miopenConvolutionDescriptor_t convDesc,
                                                   miopenConvFwdAlgorithm_t algo,
                                                   const void* beta,
                                                   const miopenTensorDescriptor_t yDesc,
                                                   void* y,
                                                   void* workSpace,
                                                   size_t workSpaceSize)
{
    if(miopen::wrapper::Dispatch("miopenConvolutionForward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenConvolutionForward");
    return miopenConvolutionForward_impl(handle,
                                         alpha,
                                         xDesc,
                                         x,
                                         wDesc,
                                         w,
                                         convDesc,
                                         algo,
                                         beta,
                                         yDesc,
                                         y,
                                         workSpace,
                                         workSpaceSize);
}

extern "C" miopenStatus_t miopenConvolutionForwardBias(miopenHandle_t handle,
                                                       const void* alpha,
                                                       const miopenTensorDescriptor_t bDesc,
                                                       const void* b,
                                                       const void* beta,
                                                       const miopenTensorDescriptor_t yDesc,
                                                       void* y)
{
    if(miopen::wrapper::Dispatch("miopenConvolutionForwardBias") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenConvolutionForwardBias");
    return miopenConvolutionForwardBias_impl(handle, alpha, bDesc, b, beta, yDesc, y);
}

extern "C" miopenStatus_t
miopenConvolutionBackwardDataGetWorkSpaceSize(miopenHandle_t handle,
                                              const miopenTensorDescriptor_t dyDesc,
                                              const miopenTensorDescriptor_t wDesc,
                                              const miopenConvolutionDescriptor_t convDesc,
                                              const miopenTensorDescriptor_t dxDesc,
                                              size_t* workSpaceSize)
{
    if(miopen::wrapper::Dispatch("miopenConvolutionBackwardDataGetWorkSpaceSize") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenConvolutionBackwardDataGetWorkSpaceSize");
    return miopenConvolutionBackwardDataGetWorkSpaceSize_impl(
        handle, dyDesc, wDesc, convDesc, dxDesc, workSpaceSize);
}

extern "C" miopenStatus_t
miopenFindConvolutionBackwardDataAlgorithm(miopenHandle_t handle,
                                           const miopenTensorDescriptor_t dyDesc,
                                           const void* dy,
                                           const miopenTensorDescriptor_t wDesc,
                                           const void* w,
                                           const miopenConvolutionDescriptor_t convDesc,
                                           const miopenTensorDescriptor_t dxDesc,
                                           void* dx,
                                           const int requestAlgoCount,
                                           int* returnedAlgoCount,
                                           miopenConvAlgoPerf_t* perfResults,
                                           void* workSpace,
                                           size_t workSpaceSize,
                                           bool exhaustiveSearch)
{
    if(miopen::wrapper::Dispatch("miopenFindConvolutionBackwardDataAlgorithm") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenFindConvolutionBackwardDataAlgorithm");
    return miopenFindConvolutionBackwardDataAlgorithm_impl(handle,
                                                           dyDesc,
                                                           dy,
                                                           wDesc,
                                                           w,
                                                           convDesc,
                                                           dxDesc,
                                                           dx,
                                                           requestAlgoCount,
                                                           returnedAlgoCount,
                                                           perfResults,
                                                           workSpace,
                                                           workSpaceSize,
                                                           exhaustiveSearch);
}

extern "C" miopenStatus_t
miopenConvolutionBackwardData(miopenHandle_t handle,
                              const void* alpha,
                              const miopenTensorDescriptor_t dyDesc,
                              const void* dy,
                              const miopenTensorDescriptor_t wDesc,
                              const void* w,
                              const miopenConvolutionDescriptor_t convDesc,
                              miopenConvBwdDataAlgorithm_t algo,
                              const void* beta,
                              const miopenTensorDescriptor_t dxDesc,
                              void* dx,
                              void* workSpace,
                              size_t workSpaceSize)
{
    if(miopen::wrapper::Dispatch("miopenConvolutionBackwardData") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenConvolutionBackwardData");
    return miopenConvolutionBackwardData_impl(handle,
                                              alpha,
                                              dyDesc,
                                              dy,
                                              wDesc,
                                              w,
                                              convDesc,
                                              algo,
                                              beta,
                                              dxDesc,
                                              dx,
                                              workSpace,
                                              workSpaceSize);
}

extern "C" miopenStatus_t
miopenConvolutionBackwardWeightsGetWorkSpaceSize(miopenHandle_t handle,
                                                 const miopenTensorDescriptor_t dyDesc,
                                                 const miopenTensorDescriptor_t xDesc,
                                                 const miopenConvolutionDescriptor_t convDesc,
                                                 const miopenTensorDescriptor_t dwDesc,
                                                 size_t* workSpaceSize)
{
    if(miopen::wrapper::Dispatch("miopenConvolutionBackwardWeightsGetWorkSpaceSize") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenConvolutionBackwardWeightsGetWorkSpaceSize");
    return miopenConvolutionBackwardWeightsGetWorkSpaceSize_impl(
        handle, dyDesc, xDesc, convDesc, dwDesc, workSpaceSize);
}

extern "C" miopenStatus_t
miopenFindConvolutionBackwardWeightsAlgorithm(miopenHandle_t handle,
                                              const miopenTensorDescriptor_t dyDesc,
                                              const void* dy,
                                              const miopenTensorDescriptor_t xDesc,
                                              const void* x,
                                              const miopenConvolutionDescriptor_t convDesc,
                                              const miopenTensorDescriptor_t dwDesc,
                                              void* dw,
                                              const int requestAlgoCount,
                                              int* returnedAlgoCount,
                                              miopenConvAlgoPerf_t* perfResults,
                                              void* workSpace,
                                              size_t workSpaceSize,
                                              bool exhaustiveSearch)
{
    if(miopen::wrapper::Dispatch("miopenFindConvolutionBackwardWeightsAlgorithm") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenFindConvolutionBackwardWeightsAlgorithm");
    return miopenFindConvolutionBackwardWeightsAlgorithm_impl(handle,
                                                              dyDesc,
                                                              dy,
                                                              xDesc,
                                                              x,
                                                              convDesc,
                                                              dwDesc,
                                                              dw,
                                                              requestAlgoCount,
                                                              returnedAlgoCount,
                                                              perfResults,
                                                              workSpace,
                                                              workSpaceSize,
                                                              exhaustiveSearch);
}

extern "C" miopenStatus_t
miopenConvolutionBackwardWeights(miopenHandle_t handle,
                                 const void* alpha,
                                 const miopenTensorDescriptor_t dyDesc,
                                 const void* dy,
                                 const miopenTensorDescriptor_t xDesc,
                                 const void* x,
                                 const miopenConvolutionDescriptor_t convDesc,
                                 miopenConvBwdWeightsAlgorithm_t algo,
                                 const void* beta,
                                 const miopenTensorDescriptor_t dwDesc,
                                 void* dw,
                                 void* workSpace,
                                 size_t workSpaceSize)
{
    if(miopen::wrapper::Dispatch("miopenConvolutionBackwardWeights") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenConvolutionBackwardWeights");
    return miopenConvolutionBackwardWeights_impl(handle,
                                                 alpha,
                                                 dyDesc,
                                                 dy,
                                                 xDesc,
                                                 x,
                                                 convDesc,
                                                 algo,
                                                 beta,
                                                 dwDesc,
                                                 dw,
                                                 workSpace,
                                                 workSpaceSize);
}

extern "C" miopenStatus_t miopenConvolutionBackwardBias(miopenHandle_t handle,
                                                        const void* alpha,
                                                        const miopenTensorDescriptor_t dyDesc,
                                                        const void* dy,
                                                        const void* beta,
                                                        const miopenTensorDescriptor_t dbDesc,
                                                        void* db)
{
    if(miopen::wrapper::Dispatch("miopenConvolutionBackwardBias") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenConvolutionBackwardBias");
    return miopenConvolutionBackwardBias_impl(handle, alpha, dyDesc, dy, beta, dbDesc, db);
}

extern "C" miopenStatus_t miopenCreatePoolingDescriptor(miopenPoolingDescriptor_t* poolDesc)
{
    if(miopen::wrapper::Dispatch("miopenCreatePoolingDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenCreatePoolingDescriptor");
    return miopenCreatePoolingDescriptor_impl(poolDesc);
}

extern "C" miopenStatus_t miopenSetPoolingIndexType(miopenPoolingDescriptor_t poolDesc,
                                                    miopenIndexType_t index_type)
{
    if(miopen::wrapper::Dispatch("miopenSetPoolingIndexType") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetPoolingIndexType");
    return miopenSetPoolingIndexType_impl(poolDesc, index_type);
}

extern "C" miopenStatus_t miopenGetPoolingIndexType(miopenPoolingDescriptor_t poolDesc,
                                                    miopenIndexType_t* index_type)
{
    if(miopen::wrapper::Dispatch("miopenGetPoolingIndexType") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetPoolingIndexType");
    return miopenGetPoolingIndexType_impl(poolDesc, index_type);
}

extern "C" miopenStatus_t
miopenSetPoolingWorkSpaceIndexMode(miopenPoolingDescriptor_t poolDesc,
                                   miopenPoolingWorkspaceIndexMode_t workspace_index)
{
    if(miopen::wrapper::Dispatch("miopenSetPoolingWorkSpaceIndexMode") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetPoolingWorkSpaceIndexMode");
    return miopenSetPoolingWorkSpaceIndexMode_impl(poolDesc, workspace_index);
}

extern "C" miopenStatus_t
miopenGetPoolingWorkSpaceIndexMode(miopenPoolingDescriptor_t poolDesc,
                                   miopenPoolingWorkspaceIndexMode_t* workspace_index)
{
    if(miopen::wrapper::Dispatch("miopenGetPoolingWorkSpaceIndexMode") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetPoolingWorkSpaceIndexMode");
    return miopenGetPoolingWorkSpaceIndexMode_impl(poolDesc, workspace_index);
}

extern "C" miopenStatus_t miopenSet2dPoolingDescriptor(miopenPoolingDescriptor_t poolDesc,
                                                       miopenPoolingMode_t mode,
                                                       int windowHeight,
                                                       int windowWidth,
                                                       int pad_h,
                                                       int pad_w,
                                                       int stride_h,
                                                       int stride_w)
{
    if(miopen::wrapper::Dispatch("miopenSet2dPoolingDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSet2dPoolingDescriptor");
    return miopenSet2dPoolingDescriptor_impl(
        poolDesc, mode, windowHeight, windowWidth, pad_h, pad_w, stride_h, stride_w);
}

extern "C" miopenStatus_t miopenGet2dPoolingDescriptor(const miopenPoolingDescriptor_t poolDesc,
                                                       miopenPoolingMode_t* mode,
                                                       int* windowHeight,
                                                       int* windowWidth,
                                                       int* pad_h,
                                                       int* pad_w,
                                                       int* stride_h,
                                                       int* stride_w)
{
    if(miopen::wrapper::Dispatch("miopenGet2dPoolingDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGet2dPoolingDescriptor");
    return miopenGet2dPoolingDescriptor_impl(
        poolDesc, mode, windowHeight, windowWidth, pad_h, pad_w, stride_h, stride_w);
}

extern "C" miopenStatus_t
miopenGetPoolingForwardOutputDim(const miopenPoolingDescriptor_t poolDesc,
                                 const miopenTensorDescriptor_t tensorDesc,
                                 int* n,
                                 int* c,
                                 int* h,
                                 int* w)
{
    if(miopen::wrapper::Dispatch("miopenGetPoolingForwardOutputDim") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetPoolingForwardOutputDim");
    return miopenGetPoolingForwardOutputDim_impl(poolDesc, tensorDesc, n, c, h, w);
}

extern "C" miopenStatus_t miopenSetNdPoolingDescriptor(miopenPoolingDescriptor_t poolDesc,
                                                       const miopenPoolingMode_t mode,
                                                       int nbDims,
                                                       const int* windowDimA,
                                                       const int* padA,
                                                       const int* stridesA)
{
    if(miopen::wrapper::Dispatch("miopenSetNdPoolingDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetNdPoolingDescriptor");
    return miopenSetNdPoolingDescriptor_impl(poolDesc, mode, nbDims, windowDimA, padA, stridesA);
}

extern "C" miopenStatus_t miopenGetNdPoolingDescriptor(const miopenPoolingDescriptor_t poolDesc,
                                                       int nbDimsRequested,
                                                       miopenPoolingMode_t* mode,
                                                       int* nbDims,
                                                       int* windowDimA,
                                                       int* padA,
                                                       int* stridesA)
{
    if(miopen::wrapper::Dispatch("miopenGetNdPoolingDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetNdPoolingDescriptor");
    return miopenGetNdPoolingDescriptor_impl(
        poolDesc, nbDimsRequested, mode, nbDims, windowDimA, padA, stridesA);
}

extern "C" miopenStatus_t
miopenGetPoolingNdForwardOutputDim(const miopenPoolingDescriptor_t poolDesc,
                                   const miopenTensorDescriptor_t tensorDesc,
                                   int dims,
                                   int* tensorDimArr)
{
    if(miopen::wrapper::Dispatch("miopenGetPoolingNdForwardOutputDim") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetPoolingNdForwardOutputDim");
    return miopenGetPoolingNdForwardOutputDim_impl(poolDesc, tensorDesc, dims, tensorDimArr);
}

extern "C" miopenStatus_t miopenPoolingGetWorkSpaceSize(const miopenTensorDescriptor_t yDesc,
                                                        size_t* workSpaceSize)
{
    if(miopen::wrapper::Dispatch("miopenPoolingGetWorkSpaceSize") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenPoolingGetWorkSpaceSize");
    return miopenPoolingGetWorkSpaceSize_impl(yDesc, workSpaceSize);
}

extern "C" miopenStatus_t miopenPoolingGetWorkSpaceSizeV2(const miopenPoolingDescriptor_t poolDesc,
                                                          const miopenTensorDescriptor_t yDesc,
                                                          size_t* workSpaceSize)
{
    if(miopen::wrapper::Dispatch("miopenPoolingGetWorkSpaceSizeV2") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenPoolingGetWorkSpaceSizeV2");
    return miopenPoolingGetWorkSpaceSizeV2_impl(poolDesc, yDesc, workSpaceSize);
}

extern "C" miopenStatus_t miopenPoolingForward(miopenHandle_t handle,
                                               const miopenPoolingDescriptor_t poolDesc,
                                               const void* alpha,
                                               const miopenTensorDescriptor_t xDesc,
                                               const void* x,
                                               const void* beta,
                                               const miopenTensorDescriptor_t yDesc,
                                               void* y,
                                               bool do_backward,
                                               void* workSpace,
                                               size_t workSpaceSize)
{
    if(miopen::wrapper::Dispatch("miopenPoolingForward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenPoolingForward");
    return miopenPoolingForward_impl(
        handle, poolDesc, alpha, xDesc, x, beta, yDesc, y, do_backward, workSpace, workSpaceSize);
}

extern "C" miopenStatus_t miopenPoolingBackward(miopenHandle_t handle,
                                                const miopenPoolingDescriptor_t poolDesc,
                                                const void* alpha,
                                                const miopenTensorDescriptor_t yDesc,
                                                const void* y,
                                                const miopenTensorDescriptor_t dyDesc,
                                                const void* dy,
                                                const miopenTensorDescriptor_t xDesc,
                                                const void* x,
                                                const void* beta,
                                                const miopenTensorDescriptor_t dxDesc,
                                                void* dx,
                                                void* workSpace)
{
    if(miopen::wrapper::Dispatch("miopenPoolingBackward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenPoolingBackward");
    return miopenPoolingBackward_impl(
        handle, poolDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx, workSpace);
}

extern "C" miopenStatus_t miopenDestroyPoolingDescriptor(miopenPoolingDescriptor_t poolDesc)
{
    if(miopen::wrapper::Dispatch("miopenDestroyPoolingDescriptor") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenDestroyPoolingDescriptor");
    return miopenDestroyPoolingDescriptor_impl(poolDesc);
}

extern "C" miopenStatus_t miopenCreateLRNDescriptor(miopenLRNDescriptor_t* lrnDesc)
{
    if(miopen::wrapper::Dispatch("miopenCreateLRNDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenCreateLRNDescriptor");
    return miopenCreateLRNDescriptor_impl(lrnDesc);
}

extern "C" miopenStatus_t miopenSetLRNDescriptor(const miopenLRNDescriptor_t lrnDesc,
                                                 miopenLRNMode_t mode,
                                                 unsigned int lrnN,
                                                 double lrnAlpha,
                                                 double lrnBeta,
                                                 double lrnK)
{
    if(miopen::wrapper::Dispatch("miopenSetLRNDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetLRNDescriptor");
    return miopenSetLRNDescriptor_impl(lrnDesc, mode, lrnN, lrnAlpha, lrnBeta, lrnK);
}

extern "C" miopenStatus_t miopenGetLRNDescriptor(const miopenLRNDescriptor_t lrnDesc,
                                                 miopenLRNMode_t* mode,
                                                 unsigned int* lrnN,
                                                 double* lrnAlpha,
                                                 double* lrnBeta,
                                                 double* lrnK)
{
    if(miopen::wrapper::Dispatch("miopenGetLRNDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetLRNDescriptor");
    return miopenGetLRNDescriptor_impl(lrnDesc, mode, lrnN, lrnAlpha, lrnBeta, lrnK);
}

extern "C" miopenStatus_t miopenLRNGetWorkSpaceSize(const miopenTensorDescriptor_t yDesc,
                                                    size_t* workSpaceSize)
{
    if(miopen::wrapper::Dispatch("miopenLRNGetWorkSpaceSize") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenLRNGetWorkSpaceSize");
    return miopenLRNGetWorkSpaceSize_impl(yDesc, workSpaceSize);
}

extern "C" miopenStatus_t miopenLRNForward(miopenHandle_t handle,
                                           const miopenLRNDescriptor_t lrnDesc,
                                           const void* alpha,
                                           const miopenTensorDescriptor_t xDesc,
                                           const void* x,
                                           const void* beta,
                                           const miopenTensorDescriptor_t yDesc,
                                           void* y,
                                           bool do_backward,
                                           void* workSpace)
{
    if(miopen::wrapper::Dispatch("miopenLRNForward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenLRNForward");
    return miopenLRNForward_impl(
        handle, lrnDesc, alpha, xDesc, x, beta, yDesc, y, do_backward, workSpace);
}

extern "C" miopenStatus_t miopenLRNBackward(miopenHandle_t handle,
                                            const miopenLRNDescriptor_t lrnDesc,
                                            const void* alpha,
                                            const miopenTensorDescriptor_t yDesc,
                                            const void* y,
                                            const miopenTensorDescriptor_t dyDesc,
                                            const void* dy,
                                            const miopenTensorDescriptor_t xDesc,
                                            const void* x,
                                            const void* beta,
                                            const miopenTensorDescriptor_t dxDesc,
                                            void* dx,
                                            const void* workSpace)
{
    if(miopen::wrapper::Dispatch("miopenLRNBackward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenLRNBackward");
    return miopenLRNBackward_impl(
        handle, lrnDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx, workSpace);
}

extern "C" miopenStatus_t miopenDestroyLRNDescriptor(miopenLRNDescriptor_t lrnDesc)
{
    if(miopen::wrapper::Dispatch("miopenDestroyLRNDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenDestroyLRNDescriptor");
    return miopenDestroyLRNDescriptor_impl(lrnDesc);
}

extern "C" miopenStatus_t miopenLayerNormForward(miopenHandle_t handle,
                                                 miopenNormMode_t mode,
                                                 const miopenTensorDescriptor_t xDesc,
                                                 const void* x,
                                                 const miopenTensorDescriptor_t weightDesc,
                                                 const void* weight,
                                                 const miopenTensorDescriptor_t biasDesc,
                                                 const void* bias,
                                                 const float epsilon,
                                                 const int32_t normalized_dim,
                                                 const miopenTensorDescriptor_t yDesc,
                                                 void* y,
                                                 const miopenTensorDescriptor_t meanDesc,
                                                 void* mean,
                                                 const miopenTensorDescriptor_t rstdDesc,
                                                 void* rstd)
{
    if(miopen::wrapper::Dispatch("miopenLayerNormForward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenLayerNormForward");
    return miopenLayerNormForward_impl(handle,
                                       mode,
                                       xDesc,
                                       x,
                                       weightDesc,
                                       weight,
                                       biasDesc,
                                       bias,
                                       epsilon,
                                       normalized_dim,
                                       yDesc,
                                       y,
                                       meanDesc,
                                       mean,
                                       rstdDesc,
                                       rstd);
}

extern "C" miopenStatus_t
miopenGetLayerNormBackwardWorkspaceSize(miopenHandle_t handle,
                                        miopenNormMode_t mode,
                                        const miopenTensorDescriptor_t dyDesc,
                                        const miopenTensorDescriptor_t xDesc,
                                        const miopenTensorDescriptor_t weightDesc,
                                        const miopenTensorDescriptor_t meanDesc,
                                        const miopenTensorDescriptor_t rstdDesc,
                                        const int32_t normalized_dim,
                                        const miopenTensorDescriptor_t dxDesc,
                                        const miopenTensorDescriptor_t dwDesc,
                                        const miopenTensorDescriptor_t dbDesc,
                                        size_t* sizeInBytes)
{
    if(miopen::wrapper::Dispatch("miopenGetLayerNormBackwardWorkspaceSize") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetLayerNormBackwardWorkspaceSize");
    return miopenGetLayerNormBackwardWorkspaceSize_impl(handle,
                                                        mode,
                                                        dyDesc,
                                                        xDesc,
                                                        weightDesc,
                                                        meanDesc,
                                                        rstdDesc,
                                                        normalized_dim,
                                                        dxDesc,
                                                        dwDesc,
                                                        dbDesc,
                                                        sizeInBytes);
}

extern "C" miopenStatus_t miopenLayerNormBackward(miopenHandle_t handle,
                                                  miopenNormMode_t mode,
                                                  void* workspace,
                                                  size_t workspaceSizeInBytes,
                                                  const miopenTensorDescriptor_t dyDesc,
                                                  const void* dy,
                                                  const miopenTensorDescriptor_t xDesc,
                                                  const void* x,
                                                  const miopenTensorDescriptor_t weightDesc,
                                                  const void* weight,
                                                  const miopenTensorDescriptor_t meanDesc,
                                                  const void* mean,
                                                  const miopenTensorDescriptor_t rstdDesc,
                                                  const void* rstd,
                                                  const int32_t normalized_dim,
                                                  const miopenTensorDescriptor_t dxDesc,
                                                  void* dx,
                                                  const miopenTensorDescriptor_t dwDesc,
                                                  void* dw,
                                                  const miopenTensorDescriptor_t dbDesc,
                                                  void* db)
{
    if(miopen::wrapper::Dispatch("miopenLayerNormBackward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenLayerNormBackward");
    return miopenLayerNormBackward_impl(handle,
                                        mode,
                                        workspace,
                                        workspaceSizeInBytes,
                                        dyDesc,
                                        dy,
                                        xDesc,
                                        x,
                                        weightDesc,
                                        weight,
                                        meanDesc,
                                        mean,
                                        rstdDesc,
                                        rstd,
                                        normalized_dim,
                                        dxDesc,
                                        dx,
                                        dwDesc,
                                        dw,
                                        dbDesc,
                                        db);
}

extern "C" miopenStatus_t miopenCatForward(miopenHandle_t handle,
                                           const int32_t xCount,
                                           const miopenTensorDescriptor_t* xDescs,
                                           const void* const* xs,
                                           const miopenTensorDescriptor_t yDesc,
                                           void* y,
                                           const int32_t dim)
{
    if(miopen::wrapper::Dispatch("miopenCatForward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenCatForward");
    return miopenCatForward_impl(handle, xCount, xDescs, xs, yDesc, y, dim);
}

extern "C" miopenStatus_t miopenDeriveBNTensorDescriptor(miopenTensorDescriptor_t derivedBnDesc,
                                                         const miopenTensorDescriptor_t xDesc,
                                                         miopenBatchNormMode_t bn_mode)
{
    if(miopen::wrapper::Dispatch("miopenDeriveBNTensorDescriptor") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenDeriveBNTensorDescriptor");
    return miopenDeriveBNTensorDescriptor_impl(derivedBnDesc, xDesc, bn_mode);
}

extern "C" miopenStatus_t
miopenBatchNormalizationForwardTraining(miopenHandle_t handle,
                                        miopenBatchNormMode_t bn_mode,
                                        void* alpha,
                                        void* beta,
                                        const miopenTensorDescriptor_t xDesc,
                                        const void* x,
                                        const miopenTensorDescriptor_t yDesc,
                                        void* y,
                                        const miopenTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                        void* bnScale,
                                        void* bnBias,
                                        double expAvgFactor,
                                        void* resultRunningMean,
                                        void* resultRunningVariance,
                                        double epsilon,
                                        void* resultSaveMean,
                                        void* resultSaveInvVariance)
{
    if(miopen::wrapper::Dispatch("miopenBatchNormalizationForwardTraining") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenBatchNormalizationForwardTraining");
    return miopenBatchNormalizationForwardTraining_impl(handle,
                                                        bn_mode,
                                                        alpha,
                                                        beta,
                                                        xDesc,
                                                        x,
                                                        yDesc,
                                                        y,
                                                        bnScaleBiasMeanVarDesc,
                                                        bnScale,
                                                        bnBias,
                                                        expAvgFactor,
                                                        resultRunningMean,
                                                        resultRunningVariance,
                                                        epsilon,
                                                        resultSaveMean,
                                                        resultSaveInvVariance);
}

extern "C" miopenStatus_t
miopenBatchNormalizationForwardTraining_V2(miopenHandle_t handle,
                                           miopenBatchNormMode_t bn_mode,
                                           void* alpha,
                                           void* beta,
                                           const miopenTensorDescriptor_t xDesc,
                                           const void* x,
                                           const miopenTensorDescriptor_t yDesc,
                                           void* y,
                                           const miopenTensorDescriptor_t scaleDesc,
                                           const miopenTensorDescriptor_t biasVarDesc,
                                           const miopenTensorDescriptor_t savedMeanDesc,
                                           const miopenTensorDescriptor_t savedVarDesc,
                                           void* bnScale,
                                           void* bnBias,
                                           double expAvgFactor,
                                           void* resultRunningMean,
                                           void* resultRunningVariance,
                                           double epsilon,
                                           void* resultSaveMean,
                                           void* resultSaveInvVariance)
{
    if(miopen::wrapper::Dispatch("miopenBatchNormalizationForwardTraining_V2") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenBatchNormalizationForwardTraining_V2");
    return miopenBatchNormalizationForwardTraining_V2_impl(handle,
                                                           bn_mode,
                                                           alpha,
                                                           beta,
                                                           xDesc,
                                                           x,
                                                           yDesc,
                                                           y,
                                                           scaleDesc,
                                                           biasVarDesc,
                                                           savedMeanDesc,
                                                           savedVarDesc,
                                                           bnScale,
                                                           bnBias,
                                                           expAvgFactor,
                                                           resultRunningMean,
                                                           resultRunningVariance,
                                                           epsilon,
                                                           resultSaveMean,
                                                           resultSaveInvVariance);
}

extern "C" miopenStatus_t
miopenBatchNormalizationForwardTraining_V3(miopenHandle_t handle,
                                           miopenBatchNormMode_t bn_mode,
                                           void* alpha,
                                           void* beta,
                                           const miopenTensorDescriptor_t xDesc,
                                           const void* x,
                                           const miopenTensorDescriptor_t yDesc,
                                           void* y,
                                           const miopenTensorDescriptor_t scaleDesc,
                                           const miopenTensorDescriptor_t biasVarDesc,
                                           const miopenTensorDescriptor_t savedMeanDesc,
                                           const miopenTensorDescriptor_t savedVarDesc,
                                           void* bnScale,
                                           void* bnBias,
                                           double expAvgFactor,
                                           const void* prevResultRunningMean,
                                           const void* prevResultRunningVariance,
                                           void* nextResultRunningMean,
                                           void* nextResultRunningVariance,
                                           double epsilon,
                                           void* resultSaveMean,
                                           void* resultSaveInvVariance)
{
    if(miopen::wrapper::Dispatch("miopenBatchNormalizationForwardTraining_V3") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenBatchNormalizationForwardTraining_V3");
    return miopenBatchNormalizationForwardTraining_V3_impl(handle,
                                                           bn_mode,
                                                           alpha,
                                                           beta,
                                                           xDesc,
                                                           x,
                                                           yDesc,
                                                           y,
                                                           scaleDesc,
                                                           biasVarDesc,
                                                           savedMeanDesc,
                                                           savedVarDesc,
                                                           bnScale,
                                                           bnBias,
                                                           expAvgFactor,
                                                           prevResultRunningMean,
                                                           prevResultRunningVariance,
                                                           nextResultRunningMean,
                                                           nextResultRunningVariance,
                                                           epsilon,
                                                           resultSaveMean,
                                                           resultSaveInvVariance);
}

extern "C" miopenStatus_t
miopenBatchNormForwardTrainingActivation(miopenHandle_t handle,
                                         miopenBatchNormMode_t bn_mode,
                                         void* alpha,
                                         void* beta,
                                         const miopenTensorDescriptor_t xDesc,
                                         const void* x,
                                         const miopenTensorDescriptor_t yDesc,
                                         void* y,
                                         const miopenTensorDescriptor_t scaleDesc,
                                         const miopenTensorDescriptor_t biasVarDesc,
                                         const miopenTensorDescriptor_t savedMeanDesc,
                                         const miopenTensorDescriptor_t savedVarDesc,
                                         void* bnScale,
                                         void* bnBias,
                                         double expAvgFactor,
                                         void* resultRunningMean,
                                         void* resultRunningVariance,
                                         double epsilon,
                                         void* resultSaveMean,
                                         void* resultSaveInvVariance,
                                         const miopenActivationDescriptor_t activDesc)
{
    if(miopen::wrapper::Dispatch("miopenBatchNormForwardTrainingActivation") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenBatchNormForwardTrainingActivation");
    return miopenBatchNormForwardTrainingActivation_impl(handle,
                                                         bn_mode,
                                                         alpha,
                                                         beta,
                                                         xDesc,
                                                         x,
                                                         yDesc,
                                                         y,
                                                         scaleDesc,
                                                         biasVarDesc,
                                                         savedMeanDesc,
                                                         savedVarDesc,
                                                         bnScale,
                                                         bnBias,
                                                         expAvgFactor,
                                                         resultRunningMean,
                                                         resultRunningVariance,
                                                         epsilon,
                                                         resultSaveMean,
                                                         resultSaveInvVariance,
                                                         activDesc);
}

extern "C" miopenStatus_t
miopenBatchNormForwardTrainingActivation_V2(miopenHandle_t handle,
                                            miopenBatchNormMode_t bn_mode,
                                            void* alpha,
                                            void* beta,
                                            const miopenTensorDescriptor_t xDesc,
                                            const void* x,
                                            const miopenTensorDescriptor_t yDesc,
                                            void* y,
                                            const miopenTensorDescriptor_t scaleDesc,
                                            const miopenTensorDescriptor_t biasVarDesc,
                                            const miopenTensorDescriptor_t savedMeanDesc,
                                            const miopenTensorDescriptor_t savedVarDesc,
                                            void* bnScale,
                                            void* bnBias,
                                            double expAvgFactor,
                                            const void* prevResultRunningMean,
                                            const void* prevResultRunningVariance,
                                            void* nextResultRunningMean,
                                            void* nextResultRunningVariance,
                                            double epsilon,
                                            void* resultSaveMean,
                                            void* resultSaveInvVariance,
                                            const miopenActivationDescriptor_t activDesc)
{
    if(miopen::wrapper::Dispatch("miopenBatchNormForwardTrainingActivation_V2") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenBatchNormForwardTrainingActivation_V2");
    return miopenBatchNormForwardTrainingActivation_V2_impl(handle,
                                                            bn_mode,
                                                            alpha,
                                                            beta,
                                                            xDesc,
                                                            x,
                                                            yDesc,
                                                            y,
                                                            scaleDesc,
                                                            biasVarDesc,
                                                            savedMeanDesc,
                                                            savedVarDesc,
                                                            bnScale,
                                                            bnBias,
                                                            expAvgFactor,
                                                            prevResultRunningMean,
                                                            prevResultRunningVariance,
                                                            nextResultRunningMean,
                                                            nextResultRunningVariance,
                                                            epsilon,
                                                            resultSaveMean,
                                                            resultSaveInvVariance,
                                                            activDesc);
}

extern "C" miopenStatus_t
miopenBatchNormalizationForwardInference(miopenHandle_t handle,
                                         miopenBatchNormMode_t bn_mode,
                                         void* alpha,
                                         void* beta,
                                         const miopenTensorDescriptor_t xDesc,
                                         const void* x,
                                         const miopenTensorDescriptor_t yDesc,
                                         void* y,
                                         const miopenTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                         void* bnScale,
                                         void* bnBias,
                                         void* estimatedMean,
                                         void* estimatedVariance,
                                         double epsilon)
{
    if(miopen::wrapper::Dispatch("miopenBatchNormalizationForwardInference") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenBatchNormalizationForwardInference");
    return miopenBatchNormalizationForwardInference_impl(handle,
                                                         bn_mode,
                                                         alpha,
                                                         beta,
                                                         xDesc,
                                                         x,
                                                         yDesc,
                                                         y,
                                                         bnScaleBiasMeanVarDesc,
                                                         bnScale,
                                                         bnBias,
                                                         estimatedMean,
                                                         estimatedVariance,
                                                         epsilon);
}

extern "C" miopenStatus_t
miopenBatchNormalizationForwardInference_V2(miopenHandle_t handle,
                                            miopenBatchNormMode_t bn_mode,
                                            void* alpha,
                                            void* beta,
                                            const miopenTensorDescriptor_t xDesc,
                                            const void* x,
                                            const miopenTensorDescriptor_t yDesc,
                                            void* y,
                                            const miopenTensorDescriptor_t scaleDesc,
                                            const miopenTensorDescriptor_t biasDesc,
                                            const miopenTensorDescriptor_t estMeanDesc,
                                            const miopenTensorDescriptor_t estVarianceDesc,
                                            void* bnScale,
                                            void* bnBias,
                                            void* estimatedMean,
                                            void* estimatedVariance,
                                            double epsilon)
{
    if(miopen::wrapper::Dispatch("miopenBatchNormalizationForwardInference_V2") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenBatchNormalizationForwardInference_V2");
    return miopenBatchNormalizationForwardInference_V2_impl(handle,
                                                            bn_mode,
                                                            alpha,
                                                            beta,
                                                            xDesc,
                                                            x,
                                                            yDesc,
                                                            y,
                                                            scaleDesc,
                                                            biasDesc,
                                                            estMeanDesc,
                                                            estVarianceDesc,
                                                            bnScale,
                                                            bnBias,
                                                            estimatedMean,
                                                            estimatedVariance,
                                                            epsilon);
}

extern "C" miopenStatus_t miopenBatchNormalizationForwardInferenceInvVariance(
    miopenHandle_t handle,
    miopenBatchNormMode_t bn_mode,
    void* alpha,
    void* beta,
    const miopenTensorDescriptor_t xDesc,
    const void* x,
    const miopenTensorDescriptor_t yDesc,
    void* y,
    const miopenTensorDescriptor_t scaleDesc,
    const miopenTensorDescriptor_t biasDesc,
    const miopenTensorDescriptor_t estMeanDesc,
    const miopenTensorDescriptor_t estInvVarianceDesc,
    void* bnScale,
    void* bnBias,
    void* estimatedMean,
    void* estimatedInvVariance)
{
    if(miopen::wrapper::Dispatch("miopenBatchNormalizationForwardInferenceInvVariance") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenBatchNormalizationForwardInferenceInvVariance");
    return miopenBatchNormalizationForwardInferenceInvVariance_impl(handle,
                                                                    bn_mode,
                                                                    alpha,
                                                                    beta,
                                                                    xDesc,
                                                                    x,
                                                                    yDesc,
                                                                    y,
                                                                    scaleDesc,
                                                                    biasDesc,
                                                                    estMeanDesc,
                                                                    estInvVarianceDesc,
                                                                    bnScale,
                                                                    bnBias,
                                                                    estimatedMean,
                                                                    estimatedInvVariance);
}

extern "C" miopenStatus_t miopenBatchNormForwardInferenceActivationInvVariance(
    miopenHandle_t handle,
    miopenBatchNormMode_t bn_mode,
    void* alpha,
    void* beta,
    const miopenTensorDescriptor_t xDesc,
    const void* x,
    const miopenTensorDescriptor_t yDesc,
    void* y,
    const miopenTensorDescriptor_t scaleDesc,
    const miopenTensorDescriptor_t biasDesc,
    const miopenTensorDescriptor_t estMeanDesc,
    const miopenTensorDescriptor_t estInvVarianceDesc,
    void* bnScale,
    void* bnBias,
    void* estimatedMean,
    void* estimatedInvVariance,
    const miopenActivationDescriptor_t activDesc)
{
    if(miopen::wrapper::Dispatch("miopenBatchNormForwardInferenceActivationInvVariance") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenBatchNormForwardInferenceActivationInvVariance");
    return miopenBatchNormForwardInferenceActivationInvVariance_impl(handle,
                                                                     bn_mode,
                                                                     alpha,
                                                                     beta,
                                                                     xDesc,
                                                                     x,
                                                                     yDesc,
                                                                     y,
                                                                     scaleDesc,
                                                                     biasDesc,
                                                                     estMeanDesc,
                                                                     estInvVarianceDesc,
                                                                     bnScale,
                                                                     bnBias,
                                                                     estimatedMean,
                                                                     estimatedInvVariance,
                                                                     activDesc);
}

extern "C" miopenStatus_t
miopenBatchNormForwardInferenceActivation(miopenHandle_t handle,
                                          miopenBatchNormMode_t bn_mode,
                                          void* alpha,
                                          void* beta,
                                          const miopenTensorDescriptor_t xDesc,
                                          const void* x,
                                          const miopenTensorDescriptor_t yDesc,
                                          void* y,
                                          const miopenTensorDescriptor_t scaleDesc,
                                          const miopenTensorDescriptor_t biasDesc,
                                          const miopenTensorDescriptor_t estMeanDesc,
                                          const miopenTensorDescriptor_t estVarianceDesc,
                                          void* bnScale,
                                          void* bnBias,
                                          void* estimatedMean,
                                          void* estimatedVariance,
                                          double epsilon,
                                          const miopenActivationDescriptor_t activDesc)
{
    if(miopen::wrapper::Dispatch("miopenBatchNormForwardInferenceActivation") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenBatchNormForwardInferenceActivation");
    return miopenBatchNormForwardInferenceActivation_impl(handle,
                                                          bn_mode,
                                                          alpha,
                                                          beta,
                                                          xDesc,
                                                          x,
                                                          yDesc,
                                                          y,
                                                          scaleDesc,
                                                          biasDesc,
                                                          estMeanDesc,
                                                          estVarianceDesc,
                                                          bnScale,
                                                          bnBias,
                                                          estimatedMean,
                                                          estimatedVariance,
                                                          epsilon,
                                                          activDesc);
}

extern "C" miopenStatus_t
miopenBatchNormalizationBackward(miopenHandle_t handle,
                                 miopenBatchNormMode_t bn_mode,
                                 const void* alphaDataDiff,
                                 const void* betaDataDiff,
                                 const void* alphaParamDiff,
                                 const void* betaParamDiff,
                                 const miopenTensorDescriptor_t xDesc,
                                 const void* x,
                                 const miopenTensorDescriptor_t dyDesc,
                                 const void* dy,
                                 const miopenTensorDescriptor_t dxDesc,
                                 void* dx,
                                 const miopenTensorDescriptor_t bnScaleBiasDiffDesc,
                                 const void* bnScale,
                                 void* resultBnScaleDiff,
                                 void* resultBnBiasDiff,
                                 double epsilon,
                                 const void* savedMean,
                                 const void* savedInvVariance)
{
    if(miopen::wrapper::Dispatch("miopenBatchNormalizationBackward") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenBatchNormalizationBackward");
    return miopenBatchNormalizationBackward_impl(handle,
                                                 bn_mode,
                                                 alphaDataDiff,
                                                 betaDataDiff,
                                                 alphaParamDiff,
                                                 betaParamDiff,
                                                 xDesc,
                                                 x,
                                                 dyDesc,
                                                 dy,
                                                 dxDesc,
                                                 dx,
                                                 bnScaleBiasDiffDesc,
                                                 bnScale,
                                                 resultBnScaleDiff,
                                                 resultBnBiasDiff,
                                                 epsilon,
                                                 savedMean,
                                                 savedInvVariance);
}

extern "C" miopenStatus_t
miopenBatchNormalizationBackward_V2(miopenHandle_t handle,
                                    miopenBatchNormMode_t bn_mode,
                                    const void* alphaDataDiff,
                                    const void* betaDataDiff,
                                    const void* alphaParamDiff,
                                    const void* betaParamDiff,
                                    const miopenTensorDescriptor_t xDesc,
                                    const void* x,
                                    const miopenTensorDescriptor_t dyDesc,
                                    const void* dy,
                                    const miopenTensorDescriptor_t dxDesc,
                                    void* dx,
                                    const miopenTensorDescriptor_t scaleDesc,
                                    const miopenTensorDescriptor_t biasDesc,
                                    const miopenTensorDescriptor_t savedMeanDesc,
                                    const miopenTensorDescriptor_t savedVarDesc,
                                    const void* bnScale,
                                    void* resultBnScaleDiff,
                                    void* resultBnBiasDiff,
                                    double epsilon,
                                    const void* savedMean,
                                    const void* savedInvVariance)
{
    if(miopen::wrapper::Dispatch("miopenBatchNormalizationBackward_V2") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenBatchNormalizationBackward_V2");
    return miopenBatchNormalizationBackward_V2_impl(handle,
                                                    bn_mode,
                                                    alphaDataDiff,
                                                    betaDataDiff,
                                                    alphaParamDiff,
                                                    betaParamDiff,
                                                    xDesc,
                                                    x,
                                                    dyDesc,
                                                    dy,
                                                    dxDesc,
                                                    dx,
                                                    scaleDesc,
                                                    biasDesc,
                                                    savedMeanDesc,
                                                    savedVarDesc,
                                                    bnScale,
                                                    resultBnScaleDiff,
                                                    resultBnBiasDiff,
                                                    epsilon,
                                                    savedMean,
                                                    savedInvVariance);
}

extern "C" miopenStatus_t
miopenBatchNormBackwardActivation(miopenHandle_t handle,
                                  miopenBatchNormMode_t bn_mode,
                                  const void* alphaDataDiff,
                                  const void* betaDataDiff,
                                  const void* alphaParamDiff,
                                  const void* betaParamDiff,
                                  const miopenTensorDescriptor_t xDesc,
                                  const void* x,
                                  const miopenTensorDescriptor_t dyDesc,
                                  const void* dy,
                                  const miopenTensorDescriptor_t dxDesc,
                                  void* dx,
                                  const miopenTensorDescriptor_t scaleDesc,
                                  const miopenTensorDescriptor_t biasDesc,
                                  const miopenTensorDescriptor_t savedMeanDesc,
                                  const miopenTensorDescriptor_t savedVarianceDesc,
                                  const void* bnScale,
                                  const void* bnBias,
                                  void* resultBnScaleDiff,
                                  void* resultBnBiasDiff,
                                  double epsilon,
                                  const void* savedMean,
                                  const void* savedInvVariance,
                                  const miopenActivationDescriptor_t activDesc)
{
    if(miopen::wrapper::Dispatch("miopenBatchNormBackwardActivation") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenBatchNormBackwardActivation");
    return miopenBatchNormBackwardActivation_impl(handle,
                                                  bn_mode,
                                                  alphaDataDiff,
                                                  betaDataDiff,
                                                  alphaParamDiff,
                                                  betaParamDiff,
                                                  xDesc,
                                                  x,
                                                  dyDesc,
                                                  dy,
                                                  dxDesc,
                                                  dx,
                                                  scaleDesc,
                                                  biasDesc,
                                                  savedMeanDesc,
                                                  savedVarianceDesc,
                                                  bnScale,
                                                  bnBias,
                                                  resultBnScaleDiff,
                                                  resultBnBiasDiff,
                                                  epsilon,
                                                  savedMean,
                                                  savedInvVariance,
                                                  activDesc);
}

extern "C" miopenStatus_t miopenCreateActivationDescriptor(miopenActivationDescriptor_t* activDesc)
{
    if(miopen::wrapper::Dispatch("miopenCreateActivationDescriptor") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenCreateActivationDescriptor");
    return miopenCreateActivationDescriptor_impl(activDesc);
}

extern "C" miopenStatus_t
miopenSetActivationDescriptor(const miopenActivationDescriptor_t activDesc,
                              miopenActivationMode_t mode,
                              double activAlpha,
                              double activBeta,
                              double activGamma)
{
    if(miopen::wrapper::Dispatch("miopenSetActivationDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetActivationDescriptor");
    return miopenSetActivationDescriptor_impl(activDesc, mode, activAlpha, activBeta, activGamma);
}

extern "C" miopenStatus_t
miopenGetActivationDescriptor(const miopenActivationDescriptor_t activDesc,
                              miopenActivationMode_t* mode,
                              double* activAlpha,
                              double* activBeta,
                              double* activGamma)
{
    if(miopen::wrapper::Dispatch("miopenGetActivationDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetActivationDescriptor");
    return miopenGetActivationDescriptor_impl(activDesc, mode, activAlpha, activBeta, activGamma);
}

extern "C" miopenStatus_t miopenActivationForward(miopenHandle_t handle,
                                                  const miopenActivationDescriptor_t activDesc,
                                                  const void* alpha,
                                                  const miopenTensorDescriptor_t xDesc,
                                                  const void* x,
                                                  const void* beta,
                                                  const miopenTensorDescriptor_t yDesc,
                                                  void* y)
{
    if(miopen::wrapper::Dispatch("miopenActivationForward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenActivationForward");
    return miopenActivationForward_impl(handle, activDesc, alpha, xDesc, x, beta, yDesc, y);
}

extern "C" miopenStatus_t miopenActivationBackward(miopenHandle_t handle,
                                                   const miopenActivationDescriptor_t activDesc,
                                                   const void* alpha,
                                                   const miopenTensorDescriptor_t yDesc,
                                                   const void* y,
                                                   const miopenTensorDescriptor_t dyDesc,
                                                   const void* dy,
                                                   const miopenTensorDescriptor_t xDesc,
                                                   const void* x,
                                                   const void* beta,
                                                   const miopenTensorDescriptor_t dxDesc,
                                                   void* dx)
{
    if(miopen::wrapper::Dispatch("miopenActivationBackward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenActivationBackward");
    return miopenActivationBackward_impl(
        handle, activDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
}

extern "C" miopenStatus_t miopenDestroyActivationDescriptor(miopenActivationDescriptor_t activDesc)
{
    if(miopen::wrapper::Dispatch("miopenDestroyActivationDescriptor") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenDestroyActivationDescriptor");
    return miopenDestroyActivationDescriptor_impl(activDesc);
}

extern "C" miopenStatus_t miopenGLUForward(miopenHandle_t handle,
                                           const miopenTensorDescriptor_t inputDesc,
                                           const void* input,
                                           const miopenTensorDescriptor_t outputDesc,
                                           void* output,
                                           const uint32_t dim)
{
    if(miopen::wrapper::Dispatch("miopenGLUForward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGLUForward");
    return miopenGLUForward_impl(handle, inputDesc, input, outputDesc, output, dim);
}

extern "C" miopenStatus_t miopenGLUBackward(miopenHandle_t handle,
                                            const miopenTensorDescriptor_t inputDesc,
                                            const void* input,
                                            const miopenTensorDescriptor_t outputGradDesc,
                                            const void* outputGrad,
                                            const miopenTensorDescriptor_t inputGradDesc,
                                            void* inputGrad,
                                            const uint32_t dim)
{
    if(miopen::wrapper::Dispatch("miopenGLUBackward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGLUBackward");
    return miopenGLUBackward_impl(
        handle, inputDesc, input, outputGradDesc, outputGrad, inputGradDesc, inputGrad, dim);
}

extern "C" miopenStatus_t miopenSoftmaxForward(miopenHandle_t handle,
                                               const void* alpha,
                                               const miopenTensorDescriptor_t xDesc,
                                               const void* x,
                                               const void* beta,
                                               const miopenTensorDescriptor_t yDesc,
                                               void* y)
{
    if(miopen::wrapper::Dispatch("miopenSoftmaxForward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSoftmaxForward");
    return miopenSoftmaxForward_impl(handle, alpha, xDesc, x, beta, yDesc, y);
}

extern "C" miopenStatus_t miopenSoftmaxBackward(miopenHandle_t handle,
                                                const void* alpha,
                                                const miopenTensorDescriptor_t yDesc,
                                                const void* y,
                                                const miopenTensorDescriptor_t dyDesc,
                                                const void* dy,
                                                const void* beta,
                                                const miopenTensorDescriptor_t dxDesc,
                                                void* dx)
{
    if(miopen::wrapper::Dispatch("miopenSoftmaxBackward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSoftmaxBackward");
    return miopenSoftmaxBackward_impl(handle, alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx);
}

extern "C" miopenStatus_t miopenSoftmaxForward_V2(miopenHandle_t handle,
                                                  const void* alpha,
                                                  const miopenTensorDescriptor_t xDesc,
                                                  const void* x,
                                                  const void* beta,
                                                  const miopenTensorDescriptor_t yDesc,
                                                  void* y,
                                                  miopenSoftmaxAlgorithm_t algorithm,
                                                  miopenSoftmaxMode_t mode)
{
    if(miopen::wrapper::Dispatch("miopenSoftmaxForward_V2") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSoftmaxForward_V2");
    return miopenSoftmaxForward_V2_impl(handle, alpha, xDesc, x, beta, yDesc, y, algorithm, mode);
}

extern "C" miopenStatus_t miopenSoftmaxBackward_V2(miopenHandle_t handle,
                                                   const void* alpha,
                                                   const miopenTensorDescriptor_t yDesc,
                                                   const void* y,
                                                   const miopenTensorDescriptor_t dyDesc,
                                                   const void* dy,
                                                   const void* beta,
                                                   const miopenTensorDescriptor_t dxDesc,
                                                   void* dx,
                                                   miopenSoftmaxAlgorithm_t algorithm,
                                                   miopenSoftmaxMode_t mode)
{
    if(miopen::wrapper::Dispatch("miopenSoftmaxBackward_V2") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSoftmaxBackward_V2");
    return miopenSoftmaxBackward_V2_impl(
        handle, alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx, algorithm, mode);
}

extern "C" miopenStatus_t miopenCreateFusionPlan(miopenFusionPlanDescriptor_t* fusePlanDesc,
                                                 const miopenFusionDirection_t fuseDirection,
                                                 const miopenTensorDescriptor_t inputDesc)
{
    if(miopen::wrapper::Dispatch("miopenCreateFusionPlan") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenCreateFusionPlan");
    return miopenCreateFusionPlan_impl(fusePlanDesc, fuseDirection, inputDesc);
}

extern "C" miopenStatus_t miopenDestroyFusionPlan(miopenFusionPlanDescriptor_t fusePlanDesc)
{
    if(miopen::wrapper::Dispatch("miopenDestroyFusionPlan") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenDestroyFusionPlan");
    return miopenDestroyFusionPlan_impl(fusePlanDesc);
}

extern "C" miopenStatus_t miopenCompileFusionPlan(miopenHandle_t handle,
                                                  miopenFusionPlanDescriptor_t fusePlanDesc)
{
    if(miopen::wrapper::Dispatch("miopenCompileFusionPlan") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenCompileFusionPlan");
    return miopenCompileFusionPlan_impl(handle, fusePlanDesc);
}

extern "C" miopenStatus_t miopenFusionPlanGetOp(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                const int op_idx,
                                                miopenFusionOpDescriptor_t* op)
{
    if(miopen::wrapper::Dispatch("miopenFusionPlanGetOp") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenFusionPlanGetOp");
    return miopenFusionPlanGetOp_impl(fusePlanDesc, op_idx, op);
}

extern "C" miopenStatus_t
miopenFusionPlanGetWorkSpaceSize(miopenHandle_t handle,
                                 miopenFusionPlanDescriptor_t fusePlanDesc,
                                 size_t* workSpaceSize,
                                 miopenConvFwdAlgorithm_t algo)
{
    if(miopen::wrapper::Dispatch("miopenFusionPlanGetWorkSpaceSize") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenFusionPlanGetWorkSpaceSize");
    return miopenFusionPlanGetWorkSpaceSize_impl(handle, fusePlanDesc, workSpaceSize, algo);
}

extern "C" miopenStatus_t
miopenFusionPlanConvolutionGetAlgo(miopenFusionPlanDescriptor_t fusePlanDesc,
                                   const int requestAlgoCount,
                                   int* returnedAlgoCount,
                                   miopenConvFwdAlgorithm_t* returnedAlgos)
{
    if(miopen::wrapper::Dispatch("miopenFusionPlanConvolutionGetAlgo") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenFusionPlanConvolutionGetAlgo");
    return miopenFusionPlanConvolutionGetAlgo_impl(
        fusePlanDesc, requestAlgoCount, returnedAlgoCount, returnedAlgos);
}

extern "C" miopenStatus_t
miopenFusionPlanConvolutionSetAlgo(miopenFusionPlanDescriptor_t fusePlanDesc,
                                   miopenConvFwdAlgorithm_t algo)
{
    if(miopen::wrapper::Dispatch("miopenFusionPlanConvolutionSetAlgo") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenFusionPlanConvolutionSetAlgo");
    return miopenFusionPlanConvolutionSetAlgo_impl(fusePlanDesc, algo);
}

extern "C" miopenStatus_t miopenCreateOpConvForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                    miopenFusionOpDescriptor_t* convOp,
                                                    miopenConvolutionDescriptor_t convDesc,
                                                    const miopenTensorDescriptor_t wDesc)
{
    if(miopen::wrapper::Dispatch("miopenCreateOpConvForward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenCreateOpConvForward");
    return miopenCreateOpConvForward_impl(fusePlanDesc, convOp, convDesc, wDesc);
}

extern "C" miopenStatus_t miopenCreateOpActivationForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                          miopenFusionOpDescriptor_t* activFwdOp,
                                                          miopenActivationMode_t mode)
{
    if(miopen::wrapper::Dispatch("miopenCreateOpActivationForward") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenCreateOpActivationForward");
    return miopenCreateOpActivationForward_impl(fusePlanDesc, activFwdOp, mode);
}

extern "C" miopenStatus_t
miopenCreateOpActivationBackward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                 miopenFusionOpDescriptor_t* activBwdOp,
                                 miopenActivationMode_t mode)
{
    if(miopen::wrapper::Dispatch("miopenCreateOpActivationBackward") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenCreateOpActivationBackward");
    return miopenCreateOpActivationBackward_impl(fusePlanDesc, activBwdOp, mode);
}

extern "C" miopenStatus_t miopenCreateOpBiasForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                    miopenFusionOpDescriptor_t* biasOp,
                                                    const miopenTensorDescriptor_t bDesc)
{
    if(miopen::wrapper::Dispatch("miopenCreateOpBiasForward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenCreateOpBiasForward");
    return miopenCreateOpBiasForward_impl(fusePlanDesc, biasOp, bDesc);
}

extern "C" miopenStatus_t
miopenCreateOpBatchNormInference(miopenFusionPlanDescriptor_t fusePlanDesc,
                                 miopenFusionOpDescriptor_t* bnOp,
                                 const miopenBatchNormMode_t bn_mode,
                                 const miopenTensorDescriptor_t bnScaleBiasMeanVarDesc)
{
    if(miopen::wrapper::Dispatch("miopenCreateOpBatchNormInference") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenCreateOpBatchNormInference");
    return miopenCreateOpBatchNormInference_impl(
        fusePlanDesc, bnOp, bn_mode, bnScaleBiasMeanVarDesc);
}

extern "C" miopenStatus_t miopenCreateOpBatchNormForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                         miopenFusionOpDescriptor_t* bnFwdOp,
                                                         const miopenBatchNormMode_t bn_mode,
                                                         bool runningMeanVariance)
{
    if(miopen::wrapper::Dispatch("miopenCreateOpBatchNormForward") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenCreateOpBatchNormForward");
    return miopenCreateOpBatchNormForward_impl(fusePlanDesc, bnFwdOp, bn_mode, runningMeanVariance);
}

extern "C" miopenStatus_t miopenCreateOpBatchNormBackward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                          miopenFusionOpDescriptor_t* bnBwdOp,
                                                          const miopenBatchNormMode_t bn_mode)
{
    if(miopen::wrapper::Dispatch("miopenCreateOpBatchNormBackward") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenCreateOpBatchNormBackward");
    return miopenCreateOpBatchNormBackward_impl(fusePlanDesc, bnBwdOp, bn_mode);
}

extern "C" miopenStatus_t miopenCreateOperatorArgs(miopenOperatorArgs_t* args)
{
    if(miopen::wrapper::Dispatch("miopenCreateOperatorArgs") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenCreateOperatorArgs");
    return miopenCreateOperatorArgs_impl(args);
}

extern "C" miopenStatus_t miopenDestroyOperatorArgs(miopenOperatorArgs_t args)
{
    if(miopen::wrapper::Dispatch("miopenDestroyOperatorArgs") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenDestroyOperatorArgs");
    return miopenDestroyOperatorArgs_impl(args);
}

extern "C" miopenStatus_t miopenSetOpArgsConvForward(miopenOperatorArgs_t args,
                                                     const miopenFusionOpDescriptor_t convOp,
                                                     const void* alpha,
                                                     const void* beta,
                                                     const void* w)
{
    if(miopen::wrapper::Dispatch("miopenSetOpArgsConvForward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetOpArgsConvForward");
    return miopenSetOpArgsConvForward_impl(args, convOp, alpha, beta, w);
}

extern "C" miopenStatus_t miopenSetOpArgsActivForward(miopenOperatorArgs_t args,
                                                      const miopenFusionOpDescriptor_t activFwdOp,
                                                      const void* alpha,
                                                      const void* beta,
                                                      double activAlpha,
                                                      double activBeta,
                                                      double activGamma)
{
    if(miopen::wrapper::Dispatch("miopenSetOpArgsActivForward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetOpArgsActivForward");
    return miopenSetOpArgsActivForward_impl(
        args, activFwdOp, alpha, beta, activAlpha, activBeta, activGamma);
}

extern "C" miopenStatus_t miopenSetOpArgsActivBackward(miopenOperatorArgs_t args,
                                                       const miopenFusionOpDescriptor_t activBwdOp,
                                                       const void* alpha,
                                                       const void* beta,
                                                       const void* y,
                                                       const void* reserved,
                                                       double activAlpha,
                                                       double activBeta,
                                                       double activGamma)
{
    if(miopen::wrapper::Dispatch("miopenSetOpArgsActivBackward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetOpArgsActivBackward");
    return miopenSetOpArgsActivBackward_impl(
        args, activBwdOp, alpha, beta, y, reserved, activAlpha, activBeta, activGamma);
}

extern "C" miopenStatus_t miopenSetOpArgsBatchNormInference(miopenOperatorArgs_t args,
                                                            const miopenFusionOpDescriptor_t bnOp,
                                                            const void* alpha,
                                                            const void* beta,
                                                            const void* bnScale,
                                                            const void* bnBias,
                                                            const void* estimatedMean,
                                                            const void* estimatedVariance,
                                                            double epsilon)
{
    if(miopen::wrapper::Dispatch("miopenSetOpArgsBatchNormInference") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetOpArgsBatchNormInference");
    return miopenSetOpArgsBatchNormInference_impl(
        args, bnOp, alpha, beta, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon);
}

extern "C" miopenStatus_t miopenSetOpArgsBatchNormForward(miopenOperatorArgs_t args,
                                                          const miopenFusionOpDescriptor_t bnOp,
                                                          const void* alpha,
                                                          const void* beta,
                                                          const void* bnScale,
                                                          const void* bnBias,
                                                          void* savedMean,
                                                          void* savedInvVariance,
                                                          void* runningMean,
                                                          void* runningVariance,
                                                          double expAvgFactor,
                                                          double epsilon)
{
    if(miopen::wrapper::Dispatch("miopenSetOpArgsBatchNormForward") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetOpArgsBatchNormForward");
    return miopenSetOpArgsBatchNormForward_impl(args,
                                                bnOp,
                                                alpha,
                                                beta,
                                                bnScale,
                                                bnBias,
                                                savedMean,
                                                savedInvVariance,
                                                runningMean,
                                                runningVariance,
                                                expAvgFactor,
                                                epsilon);
}

extern "C" miopenStatus_t miopenSetOpArgsBatchNormBackward(miopenOperatorArgs_t args,
                                                           const miopenFusionOpDescriptor_t bnOp,
                                                           const void* alpha,
                                                           const void* beta,
                                                           const void* x,
                                                           const void* bnScale,
                                                           const void* bnBias,
                                                           void* resultBnScaleDiff,
                                                           void* resultBnBiasDiff,
                                                           const void* savedMean,
                                                           const void* savedInvVariance)
{
    if(miopen::wrapper::Dispatch("miopenSetOpArgsBatchNormBackward") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetOpArgsBatchNormBackward");
    return miopenSetOpArgsBatchNormBackward_impl(args,
                                                 bnOp,
                                                 alpha,
                                                 beta,
                                                 x,
                                                 bnScale,
                                                 bnBias,
                                                 resultBnScaleDiff,
                                                 resultBnBiasDiff,
                                                 savedMean,
                                                 savedInvVariance);
}

extern "C" miopenStatus_t miopenSetOpArgsBiasForward(miopenOperatorArgs_t args,
                                                     const miopenFusionOpDescriptor_t biasOp,
                                                     const void* alpha,
                                                     const void* beta,
                                                     const void* bias)
{
    if(miopen::wrapper::Dispatch("miopenSetOpArgsBiasForward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetOpArgsBiasForward");
    return miopenSetOpArgsBiasForward_impl(args, biasOp, alpha, beta, bias);
}

extern "C" miopenStatus_t miopenExecuteFusionPlan(const miopenHandle_t handle,
                                                  const miopenFusionPlanDescriptor_t fusePlanDesc,
                                                  const miopenTensorDescriptor_t inputDesc,
                                                  const void* input,
                                                  const miopenTensorDescriptor_t outputDesc,
                                                  void* output,
                                                  miopenOperatorArgs_t args)
{
    if(miopen::wrapper::Dispatch("miopenExecuteFusionPlan") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenExecuteFusionPlan");
    return miopenExecuteFusionPlan_impl(
        handle, fusePlanDesc, inputDesc, input, outputDesc, output, args);
}

extern "C" miopenStatus_t
miopenExecuteFusionPlan_v2(const miopenHandle_t handle,
                           const miopenFusionPlanDescriptor_t fusePlanDesc,
                           const miopenTensorDescriptor_t inputDesc,
                           const void* input,
                           const miopenTensorDescriptor_t outputDesc,
                           void* output,
                           miopenOperatorArgs_t args,
                           void* workspace,
                           size_t workspaceSize)
{
    if(miopen::wrapper::Dispatch("miopenExecuteFusionPlan_v2") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenExecuteFusionPlan_v2");
    return miopenExecuteFusionPlan_v2_impl(
        handle, fusePlanDesc, inputDesc, input, outputDesc, output, args, workspace, workspaceSize);
}

extern "C" miopenStatus_t
miopenConvolutionBiasActivationForward(miopenHandle_t handle,
                                       const void* alpha1,
                                       const miopenTensorDescriptor_t xDesc,
                                       const void* x,
                                       const miopenTensorDescriptor_t wDesc,
                                       const void* w,
                                       const miopenConvolutionDescriptor_t convDesc,
                                       miopenConvFwdAlgorithm_t algo,
                                       void* workspace,
                                       size_t workspaceSizeInBytes,
                                       const void* alpha2,
                                       const miopenTensorDescriptor_t zDesc,
                                       const void* z,
                                       const miopenTensorDescriptor_t biasDesc,
                                       const void* bias,
                                       const miopenActivationDescriptor_t activationDesc,
                                       const miopenTensorDescriptor_t yDesc,
                                       void* y)
{
    if(miopen::wrapper::Dispatch("miopenConvolutionBiasActivationForward") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenConvolutionBiasActivationForward");
    return miopenConvolutionBiasActivationForward_impl(handle,
                                                       alpha1,
                                                       xDesc,
                                                       x,
                                                       wDesc,
                                                       w,
                                                       convDesc,
                                                       algo,
                                                       workspace,
                                                       workspaceSizeInBytes,
                                                       alpha2,
                                                       zDesc,
                                                       z,
                                                       biasDesc,
                                                       bias,
                                                       activationDesc,
                                                       yDesc,
                                                       y);
}

extern "C" miopenStatus_t miopenCreateRNNDescriptor(miopenRNNDescriptor_t* rnnDesc)
{
    if(miopen::wrapper::Dispatch("miopenCreateRNNDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenCreateRNNDescriptor");
    return miopenCreateRNNDescriptor_impl(rnnDesc);
}

extern "C" miopenStatus_t miopenGetRNNDescriptor(miopenRNNDescriptor_t rnnDesc,
                                                 miopenRNNMode_t* rnnMode,
                                                 miopenRNNAlgo_t* algoMode,
                                                 miopenRNNInputMode_t* inputMode,
                                                 miopenRNNDirectionMode_t* dirMode,
                                                 miopenRNNBiasMode_t* biasMode,
                                                 int* hiddenSize,
                                                 int* layer)
{
    if(miopen::wrapper::Dispatch("miopenGetRNNDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetRNNDescriptor");
    return miopenGetRNNDescriptor_impl(
        rnnDesc, rnnMode, algoMode, inputMode, dirMode, biasMode, hiddenSize, layer);
}

extern "C" miopenStatus_t miopenGetRNNDescriptor_V2(miopenRNNDescriptor_t rnnDesc,
                                                    int* hiddenSize,
                                                    int* layer,
                                                    miopenDropoutDescriptor_t* dropoutDesc,
                                                    miopenRNNInputMode_t* inputMode,
                                                    miopenRNNDirectionMode_t* dirMode,
                                                    miopenRNNMode_t* rnnMode,
                                                    miopenRNNBiasMode_t* biasMode,
                                                    miopenRNNAlgo_t* algoMode,
                                                    miopenDataType_t* dataType)
{
    if(miopen::wrapper::Dispatch("miopenGetRNNDescriptor_V2") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetRNNDescriptor_V2");
    return miopenGetRNNDescriptor_V2_impl(rnnDesc,
                                          hiddenSize,
                                          layer,
                                          dropoutDesc,
                                          inputMode,
                                          dirMode,
                                          rnnMode,
                                          biasMode,
                                          algoMode,
                                          dataType);
}

extern "C" miopenStatus_t miopenDestroyRNNDescriptor(miopenRNNDescriptor_t rnnDesc)
{
    if(miopen::wrapper::Dispatch("miopenDestroyRNNDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenDestroyRNNDescriptor");
    return miopenDestroyRNNDescriptor_impl(rnnDesc);
}

extern "C" miopenStatus_t miopenSetRNNDescriptor(miopenRNNDescriptor_t rnnDesc,
                                                 const int hsize,
                                                 const int nlayers,
                                                 miopenRNNInputMode_t inMode,
                                                 miopenRNNDirectionMode_t direction,
                                                 miopenRNNMode_t rnnMode,
                                                 miopenRNNBiasMode_t biasMode,
                                                 miopenRNNAlgo_t algo,
                                                 miopenDataType_t dataType)
{
    if(miopen::wrapper::Dispatch("miopenSetRNNDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetRNNDescriptor");
    return miopenSetRNNDescriptor_impl(
        rnnDesc, hsize, nlayers, inMode, direction, rnnMode, biasMode, algo, dataType);
}

extern "C" miopenStatus_t miopenSetRNNDescriptor_V2(miopenRNNDescriptor_t rnnDesc,
                                                    const int hsize,
                                                    const int nlayers,
                                                    miopenDropoutDescriptor_t dropoutDesc,
                                                    miopenRNNInputMode_t inMode,
                                                    miopenRNNDirectionMode_t direction,
                                                    miopenRNNMode_t rnnMode,
                                                    miopenRNNBiasMode_t biasMode,
                                                    miopenRNNAlgo_t algo,
                                                    miopenDataType_t dataType)
{
    if(miopen::wrapper::Dispatch("miopenSetRNNDescriptor_V2") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetRNNDescriptor_V2");
    return miopenSetRNNDescriptor_V2_impl(
        rnnDesc, hsize, nlayers, dropoutDesc, inMode, direction, rnnMode, biasMode, algo, dataType);
}

extern "C" miopenStatus_t
miopenSetRNNDataSeqTensorDescriptor(miopenSeqTensorDescriptor_t seqTensorDesc,
                                    miopenDataType_t dataType,
                                    miopenRNNBaseLayout_t layout,
                                    int maxSequenceLen,
                                    int batchSize,
                                    int vectorSize,
                                    const int* sequenceLenArray,
                                    void* paddingMarker)
{
    if(miopen::wrapper::Dispatch("miopenSetRNNDataSeqTensorDescriptor") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetRNNDataSeqTensorDescriptor");
    return miopenSetRNNDataSeqTensorDescriptor_impl(seqTensorDesc,
                                                    dataType,
                                                    layout,
                                                    maxSequenceLen,
                                                    batchSize,
                                                    vectorSize,
                                                    sequenceLenArray,
                                                    paddingMarker);
}

extern "C" miopenStatus_t
miopenGetRNNDataSeqTensorDescriptor(miopenSeqTensorDescriptor_t seqTensorDesc,
                                    miopenDataType_t* dataType,
                                    miopenRNNBaseLayout_t* layout,
                                    int* maxSequenceLen,
                                    int* batchSize,
                                    int* vectorSize,
                                    int sequenceLenArrayLimit,
                                    int* sequenceLenArray,
                                    void* paddingMarker)
{
    if(miopen::wrapper::Dispatch("miopenGetRNNDataSeqTensorDescriptor") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetRNNDataSeqTensorDescriptor");
    return miopenGetRNNDataSeqTensorDescriptor_impl(seqTensorDesc,
                                                    dataType,
                                                    layout,
                                                    maxSequenceLen,
                                                    batchSize,
                                                    vectorSize,
                                                    sequenceLenArrayLimit,
                                                    sequenceLenArray,
                                                    paddingMarker);
}

extern "C" miopenStatus_t miopenGetRNNWorkspaceSize(miopenHandle_t handle,
                                                    const miopenRNNDescriptor_t rnnDesc,
                                                    const int sequenceLen,
                                                    const miopenTensorDescriptor_t* xDesc,
                                                    size_t* numBytes)
{
    if(miopen::wrapper::Dispatch("miopenGetRNNWorkspaceSize") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetRNNWorkspaceSize");
    return miopenGetRNNWorkspaceSize_impl(handle, rnnDesc, sequenceLen, xDesc, numBytes);
}

extern "C" miopenStatus_t miopenGetRNNTrainingReserveSize(miopenHandle_t handle,
                                                          miopenRNNDescriptor_t rnnDesc,
                                                          const int sequenceLen,
                                                          const miopenTensorDescriptor_t* xDesc,
                                                          size_t* numBytes)
{
    if(miopen::wrapper::Dispatch("miopenGetRNNTrainingReserveSize") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetRNNTrainingReserveSize");
    return miopenGetRNNTrainingReserveSize_impl(handle, rnnDesc, sequenceLen, xDesc, numBytes);
}

extern "C" miopenStatus_t miopenGetRNNTempSpaceSizes(miopenHandle_t handle,
                                                     miopenRNNDescriptor_t rnnDesc,
                                                     miopenSeqTensorDescriptor_t xDesc,
                                                     miopenRNNFWDMode_t fwdMode,
                                                     size_t* workSpaceSize,
                                                     size_t* reserveSpaceSize)
{
    if(miopen::wrapper::Dispatch("miopenGetRNNTempSpaceSizes") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetRNNTempSpaceSizes");
    return miopenGetRNNTempSpaceSizes_impl(
        handle, rnnDesc, xDesc, fwdMode, workSpaceSize, reserveSpaceSize);
}

extern "C" miopenStatus_t miopenGetRNNParamsSize(miopenHandle_t handle,
                                                 miopenRNNDescriptor_t rnnDesc,
                                                 miopenTensorDescriptor_t xDesc,
                                                 size_t* numBytes,
                                                 miopenDataType_t dtype)
{
    if(miopen::wrapper::Dispatch("miopenGetRNNParamsSize") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetRNNParamsSize");
    return miopenGetRNNParamsSize_impl(handle, rnnDesc, xDesc, numBytes, dtype);
}

extern "C" miopenStatus_t miopenGetRNNParamsDescriptor(miopenHandle_t handle,
                                                       miopenRNNDescriptor_t rnnDesc,
                                                       miopenTensorDescriptor_t xDesc,
                                                       miopenTensorDescriptor_t wDesc,
                                                       miopenDataType_t dtype)
{
    if(miopen::wrapper::Dispatch("miopenGetRNNParamsDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetRNNParamsDescriptor");
    return miopenGetRNNParamsDescriptor_impl(handle, rnnDesc, xDesc, wDesc, dtype);
}

extern "C" miopenStatus_t miopenGetRNNInputTensorSize(miopenHandle_t handle,
                                                      miopenRNNDescriptor_t rnnDesc,
                                                      const int seqLen,
                                                      miopenTensorDescriptor_t* xDesc,
                                                      size_t* numBytes)
{
    if(miopen::wrapper::Dispatch("miopenGetRNNInputTensorSize") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetRNNInputTensorSize");
    return miopenGetRNNInputTensorSize_impl(handle, rnnDesc, seqLen, xDesc, numBytes);
}

extern "C" miopenStatus_t miopenGetRNNHiddenTensorSize(miopenHandle_t handle,
                                                       miopenRNNDescriptor_t rnnDesc,
                                                       const int seqLen,
                                                       miopenTensorDescriptor_t* xDesc,
                                                       size_t* numBytes)
{
    if(miopen::wrapper::Dispatch("miopenGetRNNHiddenTensorSize") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetRNNHiddenTensorSize");
    return miopenGetRNNHiddenTensorSize_impl(handle, rnnDesc, seqLen, xDesc, numBytes);
}

extern "C" miopenStatus_t miopenGetRNNLayerParamSize(miopenHandle_t handle,
                                                     miopenRNNDescriptor_t rnnDesc,
                                                     const int layer,
                                                     miopenTensorDescriptor_t xDesc,
                                                     const int paramID,
                                                     size_t* numBytes)
{
    if(miopen::wrapper::Dispatch("miopenGetRNNLayerParamSize") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetRNNLayerParamSize");
    return miopenGetRNNLayerParamSize_impl(handle, rnnDesc, layer, xDesc, paramID, numBytes);
}

extern "C" miopenStatus_t miopenGetRNNLayerBiasSize(miopenHandle_t handle,
                                                    miopenRNNDescriptor_t rnnDesc,
                                                    const int layer,
                                                    const int biasID,
                                                    size_t* numBytes)
{
    if(miopen::wrapper::Dispatch("miopenGetRNNLayerBiasSize") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetRNNLayerBiasSize");
    return miopenGetRNNLayerBiasSize_impl(handle, rnnDesc, layer, biasID, numBytes);
}

extern "C" miopenStatus_t miopenGetRNNLayerParam(miopenHandle_t handle,
                                                 miopenRNNDescriptor_t rnnDesc,
                                                 const int layer,
                                                 miopenTensorDescriptor_t xDesc,
                                                 miopenTensorDescriptor_t wDesc,
                                                 const void* w,
                                                 const int paramID,
                                                 miopenTensorDescriptor_t paramDesc,
                                                 void* layerParam)
{
    if(miopen::wrapper::Dispatch("miopenGetRNNLayerParam") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetRNNLayerParam");
    return miopenGetRNNLayerParam_impl(
        handle, rnnDesc, layer, xDesc, wDesc, w, paramID, paramDesc, layerParam);
}

extern "C" miopenStatus_t miopenGetRNNLayerBias(miopenHandle_t handle,
                                                miopenRNNDescriptor_t rnnDesc,
                                                const int layer,
                                                miopenTensorDescriptor_t xDesc,
                                                miopenTensorDescriptor_t wDesc,
                                                const void* w,
                                                const int biasID,
                                                miopenTensorDescriptor_t biasDesc,
                                                void* layerBias)
{
    if(miopen::wrapper::Dispatch("miopenGetRNNLayerBias") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetRNNLayerBias");
    return miopenGetRNNLayerBias_impl(
        handle, rnnDesc, layer, xDesc, wDesc, w, biasID, biasDesc, layerBias);
}

extern "C" miopenStatus_t miopenGetRNNLayerParamOffset(miopenRNNDescriptor_t rnnDesc,
                                                       const int layer,
                                                       miopenTensorDescriptor_t xDesc,
                                                       const int paramID,
                                                       miopenTensorDescriptor_t paramDesc,
                                                       size_t* layerParamOffset)
{
    if(miopen::wrapper::Dispatch("miopenGetRNNLayerParamOffset") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetRNNLayerParamOffset");
    return miopenGetRNNLayerParamOffset_impl(
        rnnDesc, layer, xDesc, paramID, paramDesc, layerParamOffset);
}

extern "C" miopenStatus_t miopenGetRNNLayerBiasOffset(miopenRNNDescriptor_t rnnDesc,
                                                      const int layer,
                                                      miopenTensorDescriptor_t xDesc,
                                                      const int biasID,
                                                      miopenTensorDescriptor_t biasDesc,
                                                      size_t* layerBiasOffset)
{
    if(miopen::wrapper::Dispatch("miopenGetRNNLayerBiasOffset") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetRNNLayerBiasOffset");
    return miopenGetRNNLayerBiasOffset_impl(
        rnnDesc, layer, xDesc, biasID, biasDesc, layerBiasOffset);
}

extern "C" miopenStatus_t miopenSetRNNLayerParam(miopenHandle_t handle,
                                                 miopenRNNDescriptor_t rnnDesc,
                                                 const int layer,
                                                 miopenTensorDescriptor_t xDesc,
                                                 miopenTensorDescriptor_t wDesc,
                                                 void* w,
                                                 const int paramID,
                                                 miopenTensorDescriptor_t paramDesc,
                                                 const void* layerParam)
{
    if(miopen::wrapper::Dispatch("miopenSetRNNLayerParam") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetRNNLayerParam");
    return miopenSetRNNLayerParam_impl(
        handle, rnnDesc, layer, xDesc, wDesc, w, paramID, paramDesc, layerParam);
}

extern "C" miopenStatus_t miopenSetRNNLayerBias(miopenHandle_t handle,
                                                miopenRNNDescriptor_t rnnDesc,
                                                const int layer,
                                                miopenTensorDescriptor_t xDesc,
                                                miopenTensorDescriptor_t wDesc,
                                                void* w,
                                                const int biasID,
                                                miopenTensorDescriptor_t biasDesc,
                                                const void* layerBias)
{
    if(miopen::wrapper::Dispatch("miopenSetRNNLayerBias") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetRNNLayerBias");
    return miopenSetRNNLayerBias_impl(
        handle, rnnDesc, layer, xDesc, wDesc, w, biasID, biasDesc, layerBias);
}

extern "C" miopenStatus_t miopenSetRNNPaddingMode(miopenRNNDescriptor_t rnnDesc,
                                                  miopenRNNPaddingMode_t paddingMode)
{
    if(miopen::wrapper::Dispatch("miopenSetRNNPaddingMode") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetRNNPaddingMode");
    return miopenSetRNNPaddingMode_impl(rnnDesc, paddingMode);
}

extern "C" miopenStatus_t miopenGetRNNPaddingMode(miopenRNNDescriptor_t rnnDesc,
                                                  miopenRNNPaddingMode_t* paddingMode)
{
    if(miopen::wrapper::Dispatch("miopenGetRNNPaddingMode") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetRNNPaddingMode");
    return miopenGetRNNPaddingMode_impl(rnnDesc, paddingMode);
}

extern "C" miopenStatus_t miopenRNNForward(miopenHandle_t handle,
                                           const miopenRNNDescriptor_t rnnDesc,
                                           miopenRNNFWDMode_t fwdMode,
                                           const miopenSeqTensorDescriptor_t xDesc,
                                           const void* x,
                                           const miopenTensorDescriptor_t hDesc,
                                           const void* hx,
                                           void* hy,
                                           const miopenTensorDescriptor_t cDesc,
                                           const void* cx,
                                           void* cy,
                                           const miopenSeqTensorDescriptor_t yDesc,
                                           void* y,
                                           const void* w,
                                           size_t weightSpaceSize,
                                           void* workSpace,
                                           size_t workSpaceNumBytes,
                                           void* reserveSpace,
                                           size_t reserveSpaceNumBytes)
{
    if(miopen::wrapper::Dispatch("miopenRNNForward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenRNNForward");
    return miopenRNNForward_impl(handle,
                                 rnnDesc,
                                 fwdMode,
                                 xDesc,
                                 x,
                                 hDesc,
                                 hx,
                                 hy,
                                 cDesc,
                                 cx,
                                 cy,
                                 yDesc,
                                 y,
                                 w,
                                 weightSpaceSize,
                                 workSpace,
                                 workSpaceNumBytes,
                                 reserveSpace,
                                 reserveSpaceNumBytes);
}

extern "C" miopenStatus_t miopenRNNBackwardSeqData(miopenHandle_t handle,
                                                   const miopenRNNDescriptor_t rnnDesc,
                                                   const miopenSeqTensorDescriptor_t yDesc,
                                                   const void* y,
                                                   const void* dy,
                                                   const miopenTensorDescriptor_t hDesc,
                                                   const void* hx,
                                                   const void* dhy,
                                                   void* dhx,
                                                   const miopenTensorDescriptor_t cDesc,
                                                   const void* cx,
                                                   const void* dcy,
                                                   void* dcx,
                                                   const miopenSeqTensorDescriptor_t xDesc,
                                                   void* dx,
                                                   const void* w,
                                                   size_t weightSpaceSize,
                                                   void* workSpace,
                                                   size_t workSpaceNumBytes,
                                                   void* reserveSpace,
                                                   size_t reserveSpaceNumBytes)
{
    if(miopen::wrapper::Dispatch("miopenRNNBackwardSeqData") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenRNNBackwardSeqData");
    return miopenRNNBackwardSeqData_impl(handle,
                                         rnnDesc,
                                         yDesc,
                                         y,
                                         dy,
                                         hDesc,
                                         hx,
                                         dhy,
                                         dhx,
                                         cDesc,
                                         cx,
                                         dcy,
                                         dcx,
                                         xDesc,
                                         dx,
                                         w,
                                         weightSpaceSize,
                                         workSpace,
                                         workSpaceNumBytes,
                                         reserveSpace,
                                         reserveSpaceNumBytes);
}

extern "C" miopenStatus_t miopenRNNBackwardWeightsSeqTensor(miopenHandle_t handle,
                                                            const miopenRNNDescriptor_t rnnDesc,
                                                            const miopenSeqTensorDescriptor_t xDesc,
                                                            const void* x,
                                                            const miopenTensorDescriptor_t hDesc,
                                                            const void* hx,
                                                            const miopenSeqTensorDescriptor_t yDesc,
                                                            const void* y,
                                                            void* dw,
                                                            size_t weightSpaceSize,
                                                            void* workSpace,
                                                            size_t workSpaceNumBytes,
                                                            const void* reserveSpace,
                                                            size_t reserveSpaceNumBytes)
{
    if(miopen::wrapper::Dispatch("miopenRNNBackwardWeightsSeqTensor") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenRNNBackwardWeightsSeqTensor");
    return miopenRNNBackwardWeightsSeqTensor_impl(handle,
                                                  rnnDesc,
                                                  xDesc,
                                                  x,
                                                  hDesc,
                                                  hx,
                                                  yDesc,
                                                  y,
                                                  dw,
                                                  weightSpaceSize,
                                                  workSpace,
                                                  workSpaceNumBytes,
                                                  reserveSpace,
                                                  reserveSpaceNumBytes);
}

extern "C" miopenStatus_t miopenRNNForwardTraining(miopenHandle_t handle,
                                                   const miopenRNNDescriptor_t rnnDesc,
                                                   const int sequenceLen,
                                                   const miopenTensorDescriptor_t* xDesc,
                                                   const void* x,
                                                   const miopenTensorDescriptor_t hxDesc,
                                                   const void* hx,
                                                   const miopenTensorDescriptor_t cxDesc,
                                                   const void* cx,
                                                   const miopenTensorDescriptor_t wDesc,
                                                   const void* w,
                                                   const miopenTensorDescriptor_t* yDesc,
                                                   void* y,
                                                   const miopenTensorDescriptor_t hyDesc,
                                                   void* hy,
                                                   const miopenTensorDescriptor_t cyDesc,
                                                   void* cy,
                                                   void* workSpace,
                                                   size_t workSpaceNumBytes,
                                                   void* reserveSpace,
                                                   size_t reserveSpaceNumBytes)
{
    if(miopen::wrapper::Dispatch("miopenRNNForwardTraining") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenRNNForwardTraining");
    return miopenRNNForwardTraining_impl(handle,
                                         rnnDesc,
                                         sequenceLen,
                                         xDesc,
                                         x,
                                         hxDesc,
                                         hx,
                                         cxDesc,
                                         cx,
                                         wDesc,
                                         w,
                                         yDesc,
                                         y,
                                         hyDesc,
                                         hy,
                                         cyDesc,
                                         cy,
                                         workSpace,
                                         workSpaceNumBytes,
                                         reserveSpace,
                                         reserveSpaceNumBytes);
}

extern "C" miopenStatus_t miopenRNNBackwardData(miopenHandle_t handle,
                                                const miopenRNNDescriptor_t rnnDesc,
                                                const int sequenceLen,
                                                const miopenTensorDescriptor_t* yDesc,
                                                const void* y,
                                                const miopenTensorDescriptor_t* dyDesc,
                                                const void* dy,
                                                const miopenTensorDescriptor_t dhyDesc,
                                                const void* dhy,
                                                const miopenTensorDescriptor_t dcyDesc,
                                                const void* dcy,
                                                const miopenTensorDescriptor_t wDesc,
                                                const void* w,
                                                const miopenTensorDescriptor_t hxDesc,
                                                const void* hx,
                                                const miopenTensorDescriptor_t cxDesc,
                                                const void* cx,
                                                const miopenTensorDescriptor_t* dxDesc,
                                                void* dx,
                                                const miopenTensorDescriptor_t dhxDesc,
                                                void* dhx,
                                                const miopenTensorDescriptor_t dcxDesc,
                                                void* dcx,
                                                void* workSpace,
                                                size_t workSpaceNumBytes,
                                                void* reserveSpace,
                                                size_t reserveSpaceNumBytes)
{
    if(miopen::wrapper::Dispatch("miopenRNNBackwardData") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenRNNBackwardData");
    return miopenRNNBackwardData_impl(handle,
                                      rnnDesc,
                                      sequenceLen,
                                      yDesc,
                                      y,
                                      dyDesc,
                                      dy,
                                      dhyDesc,
                                      dhy,
                                      dcyDesc,
                                      dcy,
                                      wDesc,
                                      w,
                                      hxDesc,
                                      hx,
                                      cxDesc,
                                      cx,
                                      dxDesc,
                                      dx,
                                      dhxDesc,
                                      dhx,
                                      dcxDesc,
                                      dcx,
                                      workSpace,
                                      workSpaceNumBytes,
                                      reserveSpace,
                                      reserveSpaceNumBytes);
}

extern "C" miopenStatus_t miopenRNNBackwardWeights(miopenHandle_t handle,
                                                   const miopenRNNDescriptor_t rnnDesc,
                                                   const int sequenceLen,
                                                   const miopenTensorDescriptor_t* xDesc,
                                                   const void* x,
                                                   const miopenTensorDescriptor_t hxDesc,
                                                   const void* hx,
                                                   const miopenTensorDescriptor_t* yDesc,
                                                   const void* y,
                                                   const miopenTensorDescriptor_t dwDesc,
                                                   void* dw,
                                                   void* workSpace,
                                                   size_t workSpaceNumBytes,
                                                   const void* reserveSpace,
                                                   size_t reserveSpaceNumBytes)
{
    if(miopen::wrapper::Dispatch("miopenRNNBackwardWeights") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenRNNBackwardWeights");
    return miopenRNNBackwardWeights_impl(handle,
                                         rnnDesc,
                                         sequenceLen,
                                         xDesc,
                                         x,
                                         hxDesc,
                                         hx,
                                         yDesc,
                                         y,
                                         dwDesc,
                                         dw,
                                         workSpace,
                                         workSpaceNumBytes,
                                         reserveSpace,
                                         reserveSpaceNumBytes);
}

extern "C" miopenStatus_t miopenRNNForwardInference(miopenHandle_t handle,
                                                    miopenRNNDescriptor_t rnnDesc,
                                                    const int sequenceLen,
                                                    const miopenTensorDescriptor_t* xDesc,
                                                    const void* x,
                                                    const miopenTensorDescriptor_t hxDesc,
                                                    const void* hx,
                                                    const miopenTensorDescriptor_t cxDesc,
                                                    const void* cx,
                                                    const miopenTensorDescriptor_t wDesc,
                                                    const void* w,
                                                    const miopenTensorDescriptor_t* yDesc,
                                                    void* y,
                                                    const miopenTensorDescriptor_t hyDesc,
                                                    void* hy,
                                                    const miopenTensorDescriptor_t cyDesc,
                                                    void* cy,
                                                    void* workSpace,
                                                    size_t workSpaceNumBytes)
{
    if(miopen::wrapper::Dispatch("miopenRNNForwardInference") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenRNNForwardInference");
    return miopenRNNForwardInference_impl(handle,
                                          rnnDesc,
                                          sequenceLen,
                                          xDesc,
                                          x,
                                          hxDesc,
                                          hx,
                                          cxDesc,
                                          cx,
                                          wDesc,
                                          w,
                                          yDesc,
                                          y,
                                          hyDesc,
                                          hy,
                                          cyDesc,
                                          cy,
                                          workSpace,
                                          workSpaceNumBytes);
}

extern "C" miopenStatus_t miopenCreateCTCLossDescriptor(miopenCTCLossDescriptor_t* ctcLossDesc)
{
    if(miopen::wrapper::Dispatch("miopenCreateCTCLossDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenCreateCTCLossDescriptor");
    return miopenCreateCTCLossDescriptor_impl(ctcLossDesc);
}

extern "C" miopenStatus_t miopenGetCTCLossDescriptor(miopenCTCLossDescriptor_t ctcLossDesc,
                                                     miopenDataType_t* dataType,
                                                     int* blank_label_id,
                                                     bool* apply_softmax_layer)
{
    if(miopen::wrapper::Dispatch("miopenGetCTCLossDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetCTCLossDescriptor");
    return miopenGetCTCLossDescriptor_impl(
        ctcLossDesc, dataType, blank_label_id, apply_softmax_layer);
}

extern "C" miopenStatus_t miopenDestroyCTCLossDescriptor(miopenCTCLossDescriptor_t ctcLossDesc)
{
    if(miopen::wrapper::Dispatch("miopenDestroyCTCLossDescriptor") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenDestroyCTCLossDescriptor");
    return miopenDestroyCTCLossDescriptor_impl(ctcLossDesc);
}

extern "C" miopenStatus_t miopenSetCTCLossDescriptor(miopenCTCLossDescriptor_t ctcLossDesc,
                                                     miopenDataType_t dataType,
                                                     const int blank_label_id,
                                                     bool apply_softmax_layer)
{
    if(miopen::wrapper::Dispatch("miopenSetCTCLossDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetCTCLossDescriptor");
    return miopenSetCTCLossDescriptor_impl(
        ctcLossDesc, dataType, blank_label_id, apply_softmax_layer);
}

extern "C" miopenStatus_t
miopenGetCTCLossWorkspaceSize(miopenHandle_t handle,
                              const miopenTensorDescriptor_t probsDesc,
                              const miopenTensorDescriptor_t gradientsDesc,
                              const int* labels,
                              const int* labelLengths,
                              const int* inputLengths,
                              miopenCTCLossAlgo_t algo,
                              const miopenCTCLossDescriptor_t ctcLossDesc,
                              size_t* workSpaceSize)
{
    if(miopen::wrapper::Dispatch("miopenGetCTCLossWorkspaceSize") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetCTCLossWorkspaceSize");
    return miopenGetCTCLossWorkspaceSize_impl(handle,
                                              probsDesc,
                                              gradientsDesc,
                                              labels,
                                              labelLengths,
                                              inputLengths,
                                              algo,
                                              ctcLossDesc,
                                              workSpaceSize);
}

extern "C" miopenStatus_t miopenCTCLoss(miopenHandle_t handle,
                                        const miopenTensorDescriptor_t probsDesc,
                                        const void* probs,
                                        const int* labels,
                                        const int* labelLengths,
                                        const int* inputLengths,
                                        void* losses,
                                        const miopenTensorDescriptor_t gradientsDesc,
                                        void* gradients,
                                        miopenCTCLossAlgo_t algo,
                                        const miopenCTCLossDescriptor_t ctcLossDesc,
                                        void* workSpace,
                                        size_t workSpaceSize)
{
    if(miopen::wrapper::Dispatch("miopenCTCLoss") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenCTCLoss");
    return miopenCTCLoss_impl(handle,
                              probsDesc,
                              probs,
                              labels,
                              labelLengths,
                              inputLengths,
                              losses,
                              gradientsDesc,
                              gradients,
                              algo,
                              ctcLossDesc,
                              workSpace,
                              workSpaceSize);
}

extern "C" miopenStatus_t miopenCreateDropoutDescriptor(miopenDropoutDescriptor_t* dropoutDesc)
{
    if(miopen::wrapper::Dispatch("miopenCreateDropoutDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenCreateDropoutDescriptor");
    return miopenCreateDropoutDescriptor_impl(dropoutDesc);
}

extern "C" miopenStatus_t miopenDestroyDropoutDescriptor(miopenDropoutDescriptor_t dropoutDesc)
{
    if(miopen::wrapper::Dispatch("miopenDestroyDropoutDescriptor") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenDestroyDropoutDescriptor");
    return miopenDestroyDropoutDescriptor_impl(dropoutDesc);
}

extern "C" miopenStatus_t miopenDropoutGetReserveSpaceSize(const miopenTensorDescriptor_t xDesc,
                                                           size_t* reserveSpaceSizeInBytes)
{
    if(miopen::wrapper::Dispatch("miopenDropoutGetReserveSpaceSize") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenDropoutGetReserveSpaceSize");
    return miopenDropoutGetReserveSpaceSize_impl(xDesc, reserveSpaceSizeInBytes);
}

extern "C" miopenStatus_t miopenDropoutGetStatesSize(miopenHandle_t handle,
                                                     size_t* stateSizeInBytes)
{
    if(miopen::wrapper::Dispatch("miopenDropoutGetStatesSize") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenDropoutGetStatesSize");
    return miopenDropoutGetStatesSize_impl(handle, stateSizeInBytes);
}

extern "C" miopenStatus_t miopenGetDropoutDescriptor(miopenDropoutDescriptor_t dropoutDesc,
                                                     miopenHandle_t handle,
                                                     float* dropout,
                                                     void** states,
                                                     unsigned long long* seed,
                                                     bool* use_mask,
                                                     bool* state_evo,
                                                     miopenRNGType_t* rng_mode)
{
    if(miopen::wrapper::Dispatch("miopenGetDropoutDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetDropoutDescriptor");
    return miopenGetDropoutDescriptor_impl(
        dropoutDesc, handle, dropout, states, seed, use_mask, state_evo, rng_mode);
}

extern "C" miopenStatus_t miopenRestoreDropoutDescriptor(miopenDropoutDescriptor_t dropoutDesc,
                                                         miopenHandle_t handle,
                                                         float dropout,
                                                         void* states,
                                                         size_t stateSizeInBytes,
                                                         unsigned long long seed,
                                                         bool use_mask,
                                                         bool state_evo,
                                                         miopenRNGType_t rng_mode)
{
    if(miopen::wrapper::Dispatch("miopenRestoreDropoutDescriptor") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenRestoreDropoutDescriptor");
    return miopenRestoreDropoutDescriptor_impl(dropoutDesc,
                                               handle,
                                               dropout,
                                               states,
                                               stateSizeInBytes,
                                               seed,
                                               use_mask,
                                               state_evo,
                                               rng_mode);
}

extern "C" miopenStatus_t miopenSetDropoutDescriptor(miopenDropoutDescriptor_t dropoutDesc,
                                                     miopenHandle_t handle,
                                                     float dropout,
                                                     void* states,
                                                     size_t stateSizeInBytes,
                                                     unsigned long long seed,
                                                     bool use_mask,
                                                     bool state_evo,
                                                     miopenRNGType_t rng_mode)
{
    if(miopen::wrapper::Dispatch("miopenSetDropoutDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetDropoutDescriptor");
    return miopenSetDropoutDescriptor_impl(dropoutDesc,
                                           handle,
                                           dropout,
                                           states,
                                           stateSizeInBytes,
                                           seed,
                                           use_mask,
                                           state_evo,
                                           rng_mode);
}

extern "C" miopenStatus_t miopenDropoutForward(miopenHandle_t handle,
                                               const miopenDropoutDescriptor_t dropoutDesc,
                                               const miopenTensorDescriptor_t noise_shape,
                                               const miopenTensorDescriptor_t xDesc,
                                               const void* x,
                                               const miopenTensorDescriptor_t yDesc,
                                               void* y,
                                               void* reserveSpace,
                                               size_t reserveSpaceSizeInBytes)
{
    if(miopen::wrapper::Dispatch("miopenDropoutForward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenDropoutForward");
    return miopenDropoutForward_impl(handle,
                                     dropoutDesc,
                                     noise_shape,
                                     xDesc,
                                     x,
                                     yDesc,
                                     y,
                                     reserveSpace,
                                     reserveSpaceSizeInBytes);
}

extern "C" miopenStatus_t miopenDropoutBackward(miopenHandle_t handle,
                                                const miopenDropoutDescriptor_t dropoutDesc,
                                                const miopenTensorDescriptor_t noise_shape,
                                                const miopenTensorDescriptor_t dyDesc,
                                                const void* dy,
                                                const miopenTensorDescriptor_t dxDesc,
                                                void* dx,
                                                void* reserveSpace,
                                                size_t reserveSpaceSizeInBytes)
{
    if(miopen::wrapper::Dispatch("miopenDropoutBackward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenDropoutBackward");
    return miopenDropoutBackward_impl(handle,
                                      dropoutDesc,
                                      noise_shape,
                                      dyDesc,
                                      dy,
                                      dxDesc,
                                      dx,
                                      reserveSpace,
                                      reserveSpaceSizeInBytes);
}

extern "C" miopenStatus_t
miopenCreateReduceTensorDescriptor(miopenReduceTensorDescriptor_t* reduceTensorDesc)
{
    if(miopen::wrapper::Dispatch("miopenCreateReduceTensorDescriptor") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenCreateReduceTensorDescriptor");
    return miopenCreateReduceTensorDescriptor_impl(reduceTensorDesc);
}

extern "C" miopenStatus_t
miopenDestroyReduceTensorDescriptor(miopenReduceTensorDescriptor_t reduceTensorDesc)
{
    if(miopen::wrapper::Dispatch("miopenDestroyReduceTensorDescriptor") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenDestroyReduceTensorDescriptor");
    return miopenDestroyReduceTensorDescriptor_impl(reduceTensorDesc);
}

extern "C" miopenStatus_t
miopenSetReduceTensorDescriptor(miopenReduceTensorDescriptor_t reduceTensorDesc,
                                miopenReduceTensorOp_t reduceTensorOp,
                                miopenDataType_t reduceTensorCompType,
                                miopenNanPropagation_t reduceTensorNanOpt,
                                miopenReduceTensorIndices_t reduceTensorIndices,
                                miopenIndicesType_t reduceTensorIndicesType)
{
    if(miopen::wrapper::Dispatch("miopenSetReduceTensorDescriptor") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetReduceTensorDescriptor");
    return miopenSetReduceTensorDescriptor_impl(reduceTensorDesc,
                                                reduceTensorOp,
                                                reduceTensorCompType,
                                                reduceTensorNanOpt,
                                                reduceTensorIndices,
                                                reduceTensorIndicesType);
}

extern "C" miopenStatus_t
miopenGetReduceTensorDescriptor(const miopenReduceTensorDescriptor_t reduceTensorDesc,
                                miopenReduceTensorOp_t* reduceTensorOp,
                                miopenDataType_t* reduceTensorCompType,
                                miopenNanPropagation_t* reduceTensorNanOpt,
                                miopenReduceTensorIndices_t* reduceTensorIndices,
                                miopenIndicesType_t* reduceTensorIndicesType)
{
    if(miopen::wrapper::Dispatch("miopenGetReduceTensorDescriptor") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetReduceTensorDescriptor");
    return miopenGetReduceTensorDescriptor_impl(reduceTensorDesc,
                                                reduceTensorOp,
                                                reduceTensorCompType,
                                                reduceTensorNanOpt,
                                                reduceTensorIndices,
                                                reduceTensorIndicesType);
}

extern "C" miopenStatus_t
miopenGetReductionIndicesSize(miopenHandle_t handle,
                              const miopenReduceTensorDescriptor_t reduceTensorDesc,
                              const miopenTensorDescriptor_t aDesc,
                              const miopenTensorDescriptor_t cDesc,
                              size_t* sizeInBytes)
{
    if(miopen::wrapper::Dispatch("miopenGetReductionIndicesSize") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetReductionIndicesSize");
    return miopenGetReductionIndicesSize_impl(handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);
}

extern "C" miopenStatus_t
miopenGetReductionWorkspaceSize(miopenHandle_t handle,
                                const miopenReduceTensorDescriptor_t reduceTensorDesc,
                                const miopenTensorDescriptor_t aDesc,
                                const miopenTensorDescriptor_t cDesc,
                                size_t* sizeInBytes)
{
    if(miopen::wrapper::Dispatch("miopenGetReductionWorkspaceSize") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetReductionWorkspaceSize");
    return miopenGetReductionWorkspaceSize_impl(
        handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);
}

extern "C" miopenStatus_t miopenReduceTensor(miopenHandle_t handle,
                                             const miopenReduceTensorDescriptor_t reduceTensorDesc,
                                             void* indices,
                                             size_t indicesSizeInBytes,
                                             void* workspace,
                                             size_t workspaceSizeInBytes,
                                             const void* alpha,
                                             const miopenTensorDescriptor_t aDesc,
                                             const void* A,
                                             const void* beta,
                                             const miopenTensorDescriptor_t cDesc,
                                             void* C)
{
    if(miopen::wrapper::Dispatch("miopenReduceTensor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenReduceTensor");
    return miopenReduceTensor_impl(handle,
                                   reduceTensorDesc,
                                   indices,
                                   indicesSizeInBytes,
                                   workspace,
                                   workspaceSizeInBytes,
                                   alpha,
                                   aDesc,
                                   A,
                                   beta,
                                   cDesc,
                                   C);
}

extern "C" miopenStatus_t miopenCreateConvProblem(miopenProblem_t* problem,
                                                  miopenConvolutionDescriptor_t operatorDesc,
                                                  miopenProblemDirection_t direction)
{
    if(miopen::wrapper::Dispatch("miopenCreateConvProblem") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenCreateConvProblem");
    return miopenCreateConvProblem_impl(problem, operatorDesc, direction);
}

extern "C" miopenStatus_t miopenCreateMhaProblem(miopenProblem_t* problem,
                                                 miopenMhaDescriptor_t operatorDesc,
                                                 miopenProblemDirection_t direction)
{
    if(miopen::wrapper::Dispatch("miopenCreateMhaProblem") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenCreateMhaProblem");
    return miopenCreateMhaProblem_impl(problem, operatorDesc, direction);
}

extern "C" miopenStatus_t miopenCreateMhaDescriptor(miopenMhaDescriptor_t* mhaDesc)
{
    if(miopen::wrapper::Dispatch("miopenCreateMhaDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenCreateMhaDescriptor");
    return miopenCreateMhaDescriptor_impl(mhaDesc);
}

extern "C" miopenStatus_t miopenSetMhaDescriptor(miopenMhaDescriptor_t mhaDesc, float scale)
{
    if(miopen::wrapper::Dispatch("miopenSetMhaDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetMhaDescriptor");
    return miopenSetMhaDescriptor_impl(mhaDesc, scale);
}

extern "C" miopenStatus_t miopenGetMhaDescriptor(miopenMhaDescriptor_t mhaDesc, float* scale)
{
    if(miopen::wrapper::Dispatch("miopenGetMhaDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetMhaDescriptor");
    return miopenGetMhaDescriptor_impl(mhaDesc, scale);
}

extern "C" miopenStatus_t miopenCreateSoftmaxDescriptor(miopenSoftmaxDescriptor_t* softmaxDesc)
{
    if(miopen::wrapper::Dispatch("miopenCreateSoftmaxDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenCreateSoftmaxDescriptor");
    return miopenCreateSoftmaxDescriptor_impl(softmaxDesc);
}

extern "C" miopenStatus_t miopenSetSoftmaxDescriptor(miopenSoftmaxDescriptor_t softmaxDesc,
                                                     float alpha,
                                                     float beta,
                                                     miopenSoftmaxAlgorithm_t algorithm,
                                                     miopenSoftmaxMode_t mode)
{
    if(miopen::wrapper::Dispatch("miopenSetSoftmaxDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetSoftmaxDescriptor");
    return miopenSetSoftmaxDescriptor_impl(softmaxDesc, alpha, beta, algorithm, mode);
}

extern "C" miopenStatus_t miopenGetSoftmaxDescriptor(const miopenSoftmaxDescriptor_t softmaxDesc,
                                                     float* alpha,
                                                     float* beta,
                                                     miopenSoftmaxAlgorithm_t* algorithm,
                                                     miopenSoftmaxMode_t* mode)
{
    if(miopen::wrapper::Dispatch("miopenGetSoftmaxDescriptor") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetSoftmaxDescriptor");
    return miopenGetSoftmaxDescriptor_impl(softmaxDesc, alpha, beta, algorithm, mode);
}

extern "C" miopenStatus_t miopenDestroyProblem(miopenProblem_t problem)
{
    if(miopen::wrapper::Dispatch("miopenDestroyProblem") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenDestroyProblem");
    return miopenDestroyProblem_impl(problem);
}

extern "C" miopenStatus_t miopenSetProblemTensorDescriptor(
    miopenProblem_t problem, miopenTensorArgumentId_t id, const miopenTensorDescriptor_t descriptor)
{
    if(miopen::wrapper::Dispatch("miopenSetProblemTensorDescriptor") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetProblemTensorDescriptor");
    return miopenSetProblemTensorDescriptor_impl(problem, id, descriptor);
}

extern "C" miopenStatus_t miopenCreateFindOptions(miopenFindOptions_t* options)
{
    if(miopen::wrapper::Dispatch("miopenCreateFindOptions") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenCreateFindOptions");
    return miopenCreateFindOptions_impl(options);
}

extern "C" miopenStatus_t miopenDestroyFindOptions(miopenFindOptions_t options)
{
    if(miopen::wrapper::Dispatch("miopenDestroyFindOptions") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenDestroyFindOptions");
    return miopenDestroyFindOptions_impl(options);
}

extern "C" miopenStatus_t miopenSetFindOptionTuning(miopenFindOptions_t options, int value)
{
    if(miopen::wrapper::Dispatch("miopenSetFindOptionTuning") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetFindOptionTuning");
    return miopenSetFindOptionTuning_impl(options, value);
}

extern "C" miopenStatus_t miopenSetFindOptionResultsOrder(miopenFindOptions_t options,
                                                          miopenFindResultsOrder_t value)
{
    if(miopen::wrapper::Dispatch("miopenSetFindOptionResultsOrder") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetFindOptionResultsOrder");
    return miopenSetFindOptionResultsOrder_impl(options, value);
}

extern "C" miopenStatus_t miopenSetFindOptionWorkspaceLimit(miopenFindOptions_t options,
                                                            size_t value)
{
    if(miopen::wrapper::Dispatch("miopenSetFindOptionWorkspaceLimit") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetFindOptionWorkspaceLimit");
    return miopenSetFindOptionWorkspaceLimit_impl(options, value);
}

extern "C" miopenStatus_t
miopenSetFindOptionPreallocatedWorkspace(miopenFindOptions_t options, void* buffer, size_t size)
{
    if(miopen::wrapper::Dispatch("miopenSetFindOptionPreallocatedWorkspace") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetFindOptionPreallocatedWorkspace");
    return miopenSetFindOptionPreallocatedWorkspace_impl(options, buffer, size);
}

extern "C" miopenStatus_t miopenSetFindOptionPreallocatedTensor(miopenFindOptions_t options,
                                                                miopenTensorArgumentId_t id,
                                                                void* buffer)
{
    if(miopen::wrapper::Dispatch("miopenSetFindOptionPreallocatedTensor") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetFindOptionPreallocatedTensor");
    return miopenSetFindOptionPreallocatedTensor_impl(options, id, buffer);
}

extern "C" miopenStatus_t miopenSetFindOptionAttachBinaries(miopenFindOptions_t options,
                                                            unsigned attach)
{
    if(miopen::wrapper::Dispatch("miopenSetFindOptionAttachBinaries") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetFindOptionAttachBinaries");
    return miopenSetFindOptionAttachBinaries_impl(options, attach);
}

extern "C" miopenStatus_t miopenFindSolutions(miopenHandle_t handle,
                                              miopenProblem_t problem,
                                              miopenFindOptions_t options,
                                              miopenSolution_t* solutions,
                                              size_t* numSolutions,
                                              size_t maxSolutions)
{
    if(miopen::wrapper::Dispatch("miopenFindSolutions") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenFindSolutions");
    return miopenFindSolutions_impl(
        handle, problem, options, solutions, numSolutions, maxSolutions);
}

extern "C" miopenStatus_t miopenRunSolution(miopenHandle_t handle,
                                            miopenSolution_t solution,
                                            size_t nInputs,
                                            const miopenTensorArgument_t* tensors,
                                            void* workspace,
                                            size_t workspaceSize)
{
    if(miopen::wrapper::Dispatch("miopenRunSolution") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenRunSolution");
    return miopenRunSolution_impl(handle, solution, nInputs, tensors, workspace, workspaceSize);
}

extern "C" miopenStatus_t miopenDestroySolution(miopenSolution_t solution)
{
    if(miopen::wrapper::Dispatch("miopenDestroySolution") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenDestroySolution");
    return miopenDestroySolution_impl(solution);
}

extern "C" miopenStatus_t
miopenLoadSolution(miopenSolution_t* solution, const char* data, size_t size)
{
    if(miopen::wrapper::Dispatch("miopenLoadSolution") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenLoadSolution");
    return miopenLoadSolution_impl(solution, data, size);
}

extern "C" miopenStatus_t miopenSaveSolution(miopenSolution_t solution, char* data)
{
    if(miopen::wrapper::Dispatch("miopenSaveSolution") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSaveSolution");
    return miopenSaveSolution_impl(solution, data);
}

extern "C" miopenStatus_t miopenGetSolutionSize(miopenSolution_t solution, size_t* size)
{
    if(miopen::wrapper::Dispatch("miopenGetSolutionSize") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetSolutionSize");
    return miopenGetSolutionSize_impl(solution, size);
}

extern "C" miopenStatus_t miopenGetSolutionWorkspaceSize(miopenSolution_t solution,
                                                         size_t* workspaceSize)
{
    if(miopen::wrapper::Dispatch("miopenGetSolutionWorkspaceSize") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetSolutionWorkspaceSize");
    return miopenGetSolutionWorkspaceSize_impl(solution, workspaceSize);
}

extern "C" miopenStatus_t miopenGetSolutionTime(miopenSolution_t solution, float* time)
{
    if(miopen::wrapper::Dispatch("miopenGetSolutionTime") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetSolutionTime");
    return miopenGetSolutionTime_impl(solution, time);
}

extern "C" miopenStatus_t miopenGetSolutionSolverId(miopenSolution_t solution, uint64_t* solverId)
{
    if(miopen::wrapper::Dispatch("miopenGetSolutionSolverId") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetSolutionSolverId");
    return miopenGetSolutionSolverId_impl(solution, solverId);
}

extern "C" miopenStatus_t miopenGetSolverIdConvAlgorithm(uint64_t solverId,
                                                         miopenConvAlgorithm_t* result)
{
    if(miopen::wrapper::Dispatch("miopenGetSolverIdConvAlgorithm") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetSolverIdConvAlgorithm");
    return miopenGetSolverIdConvAlgorithm_impl(solverId, result);
}

extern "C" miopenStatus_t miopenCreateActivationProblem(miopenProblem_t* problem,
                                                        miopenActivationDescriptor_t operatorDesc,
                                                        miopenProblemDirection_t direction)
{
    if(miopen::wrapper::Dispatch("miopenCreateActivationProblem") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenCreateActivationProblem");
    return miopenCreateActivationProblem_impl(problem, operatorDesc, direction);
}

extern "C" miopenStatus_t miopenCreateBatchnormProblem(miopenProblem_t* problem,
                                                       miopenBatchNormMode_t mode,
                                                       bool runningMeanVariance,
                                                       miopenProblemDirection_t direction)
{
    if(miopen::wrapper::Dispatch("miopenCreateBatchnormProblem") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenCreateBatchnormProblem");
    return miopenCreateBatchnormProblem_impl(problem, mode, runningMeanVariance, direction);
}

extern "C" miopenStatus_t miopenFuseProblems(miopenProblem_t problem1, miopenProblem_t problem2)
{
    if(miopen::wrapper::Dispatch("miopenFuseProblems") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenFuseProblems");
    return miopenFuseProblems_impl(problem1, problem2);
}

extern "C" miopenStatus_t miopenCreateBiasProblem(miopenProblem_t* problem,
                                                  miopenProblemDirection_t direction)
{
    if(miopen::wrapper::Dispatch("miopenCreateBiasProblem") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenCreateBiasProblem");
    return miopenCreateBiasProblem_impl(problem, direction);
}

extern "C" miopenStatus_t miopenCreateSoftmaxProblem(miopenProblem_t* problem,
                                                     miopenSoftmaxDescriptor_t operatorDesc,
                                                     miopenProblemDirection_t direction)
{
    if(miopen::wrapper::Dispatch("miopenCreateSoftmaxProblem") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenCreateSoftmaxProblem");
    return miopenCreateSoftmaxProblem_impl(problem, operatorDesc, direction);
}

extern "C" miopenStatus_t
miopenGetReduceCalculationWorkspaceSize(miopenHandle_t handle,
                                        const miopenTensorDescriptor_t xDesc,
                                        const int32_t dim,
                                        const miopenReduceCalculationOp_t reduceCalculationOp,
                                        const miopenTensorDescriptor_t reduceDesc,
                                        size_t* sizeInBytes)
{
    if(miopen::wrapper::Dispatch("miopenGetReduceCalculationWorkspaceSize") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetReduceCalculationWorkspaceSize");
    return miopenGetReduceCalculationWorkspaceSize_impl(
        handle, xDesc, dim, reduceCalculationOp, reduceDesc, sizeInBytes);
}

extern "C" miopenStatus_t
miopenReduceCalculationForward(miopenHandle_t handle,
                               miopenReduceCalculationNanPropagation_t nanPropagation,
                               void* workspace,
                               size_t workspaceSizeInBytes,
                               const miopenTensorDescriptor_t xDesc,
                               const void* x,
                               const int32_t dim,
                               const miopenReduceCalculationOp_t reduceCalculationOp,
                               const miopenTensorDescriptor_t reduceDesc,
                               void* y)
{
    if(miopen::wrapper::Dispatch("miopenReduceCalculationForward") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenReduceCalculationForward");
    return miopenReduceCalculationForward_impl(handle,
                                               nanPropagation,
                                               workspace,
                                               workspaceSizeInBytes,
                                               xDesc,
                                               x,
                                               dim,
                                               reduceCalculationOp,
                                               reduceDesc,
                                               y);
}

extern "C" miopenStatus_t miopenReduceExtremeForward(miopenHandle_t handle,
                                                     const miopenTensorDescriptor_t xDesc,
                                                     const void* x,
                                                     const int32_t dim,
                                                     const miopenReduceExtremeOp_t reduceExtremeOp,
                                                     const miopenTensorDescriptor_t yDesc,
                                                     void* y,
                                                     const miopenTensorDescriptor_t indiceDesc,
                                                     void* indice)
{
    if(miopen::wrapper::Dispatch("miopenReduceExtremeForward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenReduceExtremeForward");
    return miopenReduceExtremeForward_impl(
        handle, xDesc, x, dim, reduceExtremeOp, yDesc, y, indiceDesc, indice);
}

extern "C" miopenStatus_t miopenGroupNormForward(miopenHandle_t handle,
                                                 miopenNormMode_t mode,
                                                 const miopenTensorDescriptor_t xDesc,
                                                 const void* x,
                                                 const miopenTensorDescriptor_t weightDesc,
                                                 const void* weight,
                                                 const miopenTensorDescriptor_t biasDesc,
                                                 const void* bias,
                                                 const uint64_t num_groups,
                                                 const float epsilon,
                                                 const miopenTensorDescriptor_t yDesc,
                                                 void* y,
                                                 const miopenTensorDescriptor_t meanDesc,
                                                 void* mean,
                                                 const miopenTensorDescriptor_t rstdDesc,
                                                 void* rstd)
{
    if(miopen::wrapper::Dispatch("miopenGroupNormForward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGroupNormForward");
    return miopenGroupNormForward_impl(handle,
                                       mode,
                                       xDesc,
                                       x,
                                       weightDesc,
                                       weight,
                                       biasDesc,
                                       bias,
                                       num_groups,
                                       epsilon,
                                       yDesc,
                                       y,
                                       meanDesc,
                                       mean,
                                       rstdDesc,
                                       rstd);
}

extern "C" miopenStatus_t miopenAddLayerNormForward(miopenHandle_t handle,
                                                    miopenNormMode_t mode,
                                                    const miopenTensorDescriptor_t xDesc,
                                                    const void* x,
                                                    const miopenTensorDescriptor_t x2Desc,
                                                    const void* x2,
                                                    const miopenTensorDescriptor_t weightDesc,
                                                    const void* weight,
                                                    const miopenTensorDescriptor_t biasDesc,
                                                    const void* bias,
                                                    const float epsilon,
                                                    const int32_t normalized_dim,
                                                    const miopenTensorDescriptor_t yDesc,
                                                    void* y,
                                                    const miopenTensorDescriptor_t meanDesc,
                                                    void* mean,
                                                    const miopenTensorDescriptor_t rstdDesc,
                                                    void* rstd)
{
    if(miopen::wrapper::Dispatch("miopenAddLayerNormForward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenAddLayerNormForward");
    return miopenAddLayerNormForward_impl(handle,
                                          mode,
                                          xDesc,
                                          x,
                                          x2Desc,
                                          x2,
                                          weightDesc,
                                          weight,
                                          biasDesc,
                                          bias,
                                          epsilon,
                                          normalized_dim,
                                          yDesc,
                                          y,
                                          meanDesc,
                                          mean,
                                          rstdDesc,
                                          rstd);
}

extern "C" miopenStatus_t miopenT5LayerNormForward(miopenHandle_t handle,
                                                   miopenNormMode_t mode,
                                                   const miopenTensorDescriptor_t xDesc,
                                                   const void* x,
                                                   const miopenTensorDescriptor_t weightDesc,
                                                   const void* weight,
                                                   const float epsilon,
                                                   const miopenTensorDescriptor_t yDesc,
                                                   void* y,
                                                   const miopenTensorDescriptor_t rstdDesc,
                                                   void* rstd)
{
    if(miopen::wrapper::Dispatch("miopenT5LayerNormForward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenT5LayerNormForward");
    return miopenT5LayerNormForward_impl(
        handle, mode, xDesc, x, weightDesc, weight, epsilon, yDesc, y, rstdDesc, rstd);
}

extern "C" miopenStatus_t
miopenGetT5LayerNormBackwardWorkspaceSize(miopenHandle_t handle,
                                          miopenNormMode_t mode,
                                          const miopenTensorDescriptor_t dyDesc,
                                          const miopenTensorDescriptor_t xDesc,
                                          const miopenTensorDescriptor_t weightDesc,
                                          const miopenTensorDescriptor_t rstdDesc,
                                          const miopenTensorDescriptor_t dxDesc,
                                          const miopenTensorDescriptor_t dwDesc,
                                          size_t* sizeInBytes)
{
    if(miopen::wrapper::Dispatch("miopenGetT5LayerNormBackwardWorkspaceSize") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetT5LayerNormBackwardWorkspaceSize");
    return miopenGetT5LayerNormBackwardWorkspaceSize_impl(
        handle, mode, dyDesc, xDesc, weightDesc, rstdDesc, dxDesc, dwDesc, sizeInBytes);
}

extern "C" miopenStatus_t miopenT5LayerNormBackward(miopenHandle_t handle,
                                                    miopenNormMode_t mode,
                                                    void* workspace,
                                                    size_t workspaceSizeInBytes,
                                                    const miopenTensorDescriptor_t dyDesc,
                                                    const void* dy,
                                                    const miopenTensorDescriptor_t xDesc,
                                                    const void* x,
                                                    const miopenTensorDescriptor_t weightDesc,
                                                    const void* weight,
                                                    const miopenTensorDescriptor_t rstdDesc,
                                                    const void* rstd,
                                                    const miopenTensorDescriptor_t dxDesc,
                                                    void* dx,
                                                    const miopenTensorDescriptor_t dwDesc,
                                                    void* dw)
{
    if(miopen::wrapper::Dispatch("miopenT5LayerNormBackward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenT5LayerNormBackward");
    return miopenT5LayerNormBackward_impl(handle,
                                          mode,
                                          workspace,
                                          workspaceSizeInBytes,
                                          dyDesc,
                                          dy,
                                          xDesc,
                                          x,
                                          weightDesc,
                                          weight,
                                          rstdDesc,
                                          rstd,
                                          dxDesc,
                                          dx,
                                          dwDesc,
                                          dw);
}

extern "C" miopenStatus_t miopenFusedAdam(miopenHandle_t handle,
                                          const miopenTensorDescriptor_t paramDesc,
                                          void* param,
                                          const miopenTensorDescriptor_t gradDesc,
                                          const void* grad,
                                          const miopenTensorDescriptor_t expAvgDesc,
                                          void* expAvg,
                                          const miopenTensorDescriptor_t expAvgSqDesc,
                                          void* expAvgSq,
                                          const miopenTensorDescriptor_t maxExpAvgSqDesc,
                                          void* maxExpAvgSq,
                                          const miopenTensorDescriptor_t stateStepDesc,
                                          void* stateStep,
                                          const unsigned int state_step,
                                          const float lr,
                                          const float beta1,
                                          const float beta2,
                                          const float weight_decay,
                                          const float eps,
                                          const bool amsgrad,
                                          const bool maximize,
                                          const bool adamw,
                                          const miopenTensorDescriptor_t gradScaleDesc,
                                          const void* gradScale,
                                          const miopenTensorDescriptor_t foundInfDesc,
                                          const void* foundInf)
{
    if(miopen::wrapper::Dispatch("miopenFusedAdam") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenFusedAdam");
    return miopenFusedAdam_impl(handle,
                                paramDesc,
                                param,
                                gradDesc,
                                grad,
                                expAvgDesc,
                                expAvg,
                                expAvgSqDesc,
                                expAvgSq,
                                maxExpAvgSqDesc,
                                maxExpAvgSq,
                                stateStepDesc,
                                stateStep,
                                state_step,
                                lr,
                                beta1,
                                beta2,
                                weight_decay,
                                eps,
                                amsgrad,
                                maximize,
                                adamw,
                                gradScaleDesc,
                                gradScale,
                                foundInfDesc,
                                foundInf);
}

extern "C" miopenStatus_t
miopenFusedAdamWithOutput(miopenHandle_t handle,
                          const miopenTensorDescriptor_t paramInDesc,
                          void* paramIn,
                          const miopenTensorDescriptor_t paramOutDesc,
                          void* paramOut,
                          const miopenTensorDescriptor_t paramOutFloat16Desc,
                          void* paramOutFloat16,
                          const miopenTensorDescriptor_t gradInDesc,
                          const void* gradIn,
                          const miopenTensorDescriptor_t expAvgInDesc,
                          void* expAvgIn,
                          const miopenTensorDescriptor_t expAvgOutDesc,
                          void* expAvgOut,
                          const miopenTensorDescriptor_t expAvgSqInDesc,
                          void* expAvgSqIn,
                          const miopenTensorDescriptor_t expAvgSqOutDesc,
                          void* expAvgSqOut,
                          const miopenTensorDescriptor_t maxExpAvgSqInDesc,
                          void* maxExpAvgSqIn,
                          const miopenTensorDescriptor_t maxExpAvgSqOutDesc,
                          void* maxExpAvgSqOut,
                          const miopenTensorDescriptor_t stateStepInDesc,
                          void* stateStepIn,
                          const miopenTensorDescriptor_t stateStepOutDesc,
                          void* stateStepOut,
                          const unsigned int state_step,
                          const float lr,
                          const float beta1,
                          const float beta2,
                          const float weight_decay,
                          const float eps,
                          const bool amsgrad,
                          const bool maximize,
                          const bool adamw,
                          const miopenTensorDescriptor_t gradScaleDesc,
                          const void* gradScale,
                          const miopenTensorDescriptor_t foundInfDesc,
                          const void* foundInf)
{
    if(miopen::wrapper::Dispatch("miopenFusedAdamWithOutput") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenFusedAdamWithOutput");
    return miopenFusedAdamWithOutput_impl(handle,
                                          paramInDesc,
                                          paramIn,
                                          paramOutDesc,
                                          paramOut,
                                          paramOutFloat16Desc,
                                          paramOutFloat16,
                                          gradInDesc,
                                          gradIn,
                                          expAvgInDesc,
                                          expAvgIn,
                                          expAvgOutDesc,
                                          expAvgOut,
                                          expAvgSqInDesc,
                                          expAvgSqIn,
                                          expAvgSqOutDesc,
                                          expAvgSqOut,
                                          maxExpAvgSqInDesc,
                                          maxExpAvgSqIn,
                                          maxExpAvgSqOutDesc,
                                          maxExpAvgSqOut,
                                          stateStepInDesc,
                                          stateStepIn,
                                          stateStepOutDesc,
                                          stateStepOut,
                                          state_step,
                                          lr,
                                          beta1,
                                          beta2,
                                          weight_decay,
                                          eps,
                                          amsgrad,
                                          maximize,
                                          adamw,
                                          gradScaleDesc,
                                          gradScale,
                                          foundInfDesc,
                                          foundInf);
}

extern "C" miopenStatus_t miopenTransformersAdamW(miopenHandle_t handle,
                                                  const miopenTensorDescriptor_t paramDesc,
                                                  void* param,
                                                  const miopenTensorDescriptor_t gradDesc,
                                                  const void* grad,
                                                  const miopenTensorDescriptor_t expAvgDesc,
                                                  void* expAvg,
                                                  const miopenTensorDescriptor_t expAvgSqDesc,
                                                  void* expAvgSq,
                                                  const miopenTensorDescriptor_t stateStepDesc,
                                                  void* stateStep,
                                                  const unsigned int state_step,
                                                  const float lr,
                                                  const float beta1,
                                                  const float beta2,
                                                  const float weight_decay,
                                                  const float eps,
                                                  const bool correct_bias,
                                                  const miopenTensorDescriptor_t gradScaleDesc,
                                                  const void* gradScale,
                                                  const miopenTensorDescriptor_t foundInfDesc,
                                                  const void* foundInf)
{
    if(miopen::wrapper::Dispatch("miopenTransformersAdamW") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenTransformersAdamW");
    return miopenTransformersAdamW_impl(handle,
                                        paramDesc,
                                        param,
                                        gradDesc,
                                        grad,
                                        expAvgDesc,
                                        expAvg,
                                        expAvgSqDesc,
                                        expAvgSq,
                                        stateStepDesc,
                                        stateStep,
                                        state_step,
                                        lr,
                                        beta1,
                                        beta2,
                                        weight_decay,
                                        eps,
                                        correct_bias,
                                        gradScaleDesc,
                                        gradScale,
                                        foundInfDesc,
                                        foundInf);
}

extern "C" miopenStatus_t
miopenTransformersAdamWWithOutput(miopenHandle_t handle,
                                  const miopenTensorDescriptor_t paramInDesc,
                                  void* paramIn,
                                  const miopenTensorDescriptor_t paramOutDesc,
                                  void* paramOut,
                                  const miopenTensorDescriptor_t paramOutFloat16Desc,
                                  void* paramOutFloat16,
                                  const miopenTensorDescriptor_t gradInDesc,
                                  const void* gradIn,
                                  const miopenTensorDescriptor_t expAvgInDesc,
                                  void* expAvgIn,
                                  const miopenTensorDescriptor_t expAvgOutDesc,
                                  void* expAvgOut,
                                  const miopenTensorDescriptor_t expAvgSqInDesc,
                                  void* expAvgSqIn,
                                  const miopenTensorDescriptor_t expAvgSqOutDesc,
                                  void* expAvgSqOut,
                                  const miopenTensorDescriptor_t stateStepInDesc,
                                  void* stateStepIn,
                                  const miopenTensorDescriptor_t stateStepOutDesc,
                                  void* stateStepOut,
                                  const unsigned int state_step,
                                  const float lr,
                                  const float beta1,
                                  const float beta2,
                                  const float weight_decay,
                                  const float eps,
                                  const float step_size,
                                  const bool correct_bias,
                                  const miopenTensorDescriptor_t gradScaleDesc,
                                  const void* gradScale,
                                  const miopenTensorDescriptor_t foundInfDesc,
                                  const void* foundInf)
{
    if(miopen::wrapper::Dispatch("miopenTransformersAdamWWithOutput") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenTransformersAdamWWithOutput");
    return miopenTransformersAdamWWithOutput_impl(handle,
                                                  paramInDesc,
                                                  paramIn,
                                                  paramOutDesc,
                                                  paramOut,
                                                  paramOutFloat16Desc,
                                                  paramOutFloat16,
                                                  gradInDesc,
                                                  gradIn,
                                                  expAvgInDesc,
                                                  expAvgIn,
                                                  expAvgOutDesc,
                                                  expAvgOut,
                                                  expAvgSqInDesc,
                                                  expAvgSqIn,
                                                  expAvgSqOutDesc,
                                                  expAvgSqOut,
                                                  stateStepInDesc,
                                                  stateStepIn,
                                                  stateStepOutDesc,
                                                  stateStepOut,
                                                  state_step,
                                                  lr,
                                                  beta1,
                                                  beta2,
                                                  weight_decay,
                                                  eps,
                                                  step_size,
                                                  correct_bias,
                                                  gradScaleDesc,
                                                  gradScale,
                                                  foundInfDesc,
                                                  foundInf);
}

extern "C" miopenStatus_t miopenGetGetitemWorkspaceSize(miopenHandle_t handle,
                                                        uint32_t indexCount,
                                                        const miopenTensorDescriptor_t* indexDescs,
                                                        size_t* sizeInBytes)
{
    if(miopen::wrapper::Dispatch("miopenGetGetitemWorkspaceSize") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetGetitemWorkspaceSize");
    return miopenGetGetitemWorkspaceSize_impl(handle, indexCount, indexDescs, sizeInBytes);
}

extern "C" miopenStatus_t miopenGetitemBackward(miopenHandle_t handle,
                                                void* workspace,
                                                size_t workspaceSizeInBytes,
                                                const miopenTensorDescriptor_t dyDesc,
                                                const void* dy,
                                                uint32_t indexCount,
                                                const miopenTensorDescriptor_t* indexDescs,
                                                const void* const* indexs,
                                                const miopenTensorDescriptor_t dxDesc,
                                                void* dx,
                                                const miopenTensorDescriptor_t errorDesc,
                                                void* error,
                                                uint32_t dimCount,
                                                const int32_t* dims,
                                                uint32_t sliceCount,
                                                const int32_t* slices,
                                                uint32_t offset)
{
    if(miopen::wrapper::Dispatch("miopenGetitemBackward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetitemBackward");
    return miopenGetitemBackward_impl(handle,
                                      workspace,
                                      workspaceSizeInBytes,
                                      dyDesc,
                                      dy,
                                      indexCount,
                                      indexDescs,
                                      indexs,
                                      dxDesc,
                                      dx,
                                      errorDesc,
                                      error,
                                      dimCount,
                                      dims,
                                      sliceCount,
                                      slices,
                                      offset);
}

extern "C" miopenStatus_t miopenRoPEForward(miopenHandle_t handle,
                                            const miopenTensorDescriptor_t xDesc,
                                            const void* x,
                                            const miopenTensorDescriptor_t cosDesc,
                                            const void* cos,
                                            const miopenTensorDescriptor_t sinDesc,
                                            const void* sin,
                                            const miopenTensorDescriptor_t yDesc,
                                            void* y)
{
    if(miopen::wrapper::Dispatch("miopenRoPEForward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenRoPEForward");
    return miopenRoPEForward_impl(handle, xDesc, x, cosDesc, cos, sinDesc, sin, yDesc, y);
}

extern "C" miopenStatus_t miopenRoPEBackward(miopenHandle_t handle,
                                             const miopenTensorDescriptor_t dyDesc,
                                             const void* dy,
                                             const miopenTensorDescriptor_t cosDesc,
                                             const void* cos,
                                             const miopenTensorDescriptor_t sinDesc,
                                             const void* sin,
                                             const miopenTensorDescriptor_t dxDesc,
                                             void* dx)
{
    if(miopen::wrapper::Dispatch("miopenRoPEBackward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenRoPEBackward");
    return miopenRoPEBackward_impl(handle, dyDesc, dy, cosDesc, cos, sinDesc, sin, dxDesc, dx);
}

extern "C" miopenStatus_t miopenKthvalueForward(miopenHandle_t handle,
                                                miopenTensorDescriptor_t inputDesc,
                                                const void* input,
                                                miopenTensorDescriptor_t outputDesc,
                                                void* output,
                                                miopenTensorDescriptor_t indicesDesc,
                                                size_t* indices,
                                                size_t k,
                                                int32_t dim,
                                                bool keepDim)
{
    if(miopen::wrapper::Dispatch("miopenKthvalueForward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenKthvalueForward");
    return miopenKthvalueForward_impl(
        handle, inputDesc, input, outputDesc, output, indicesDesc, indices, k, dim, keepDim);
}

extern "C" miopenStatus_t miopenGetPReLUBackwardWorkspaceSize(miopenHandle_t handle,
                                                              miopenTensorDescriptor_t inputDesc,
                                                              miopenTensorDescriptor_t weightDesc,
                                                              size_t* sizeInBytes)
{
    if(miopen::wrapper::Dispatch("miopenGetPReLUBackwardWorkspaceSize") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetPReLUBackwardWorkspaceSize");
    return miopenGetPReLUBackwardWorkspaceSize_impl(handle, inputDesc, weightDesc, sizeInBytes);
}

extern "C" miopenStatus_t miopenPReLUBackward(miopenHandle_t handle,
                                              void* workspace,
                                              size_t workspaceSizeInBytes,
                                              miopenTensorDescriptor_t inputDesc,
                                              const void* input,
                                              miopenTensorDescriptor_t weightDesc,
                                              const void* weight,
                                              miopenTensorDescriptor_t doutputDesc,
                                              const void* doutput,
                                              miopenTensorDescriptor_t dinputDesc,
                                              void* dinput,
                                              miopenTensorDescriptor_t dweightDesc,
                                              void* dweight)
{
    if(miopen::wrapper::Dispatch("miopenPReLUBackward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenPReLUBackward");
    return miopenPReLUBackward_impl(handle,
                                    workspace,
                                    workspaceSizeInBytes,
                                    inputDesc,
                                    input,
                                    weightDesc,
                                    weight,
                                    doutputDesc,
                                    doutput,
                                    dinputDesc,
                                    dinput,
                                    dweightDesc,
                                    dweight);
}

extern "C" miopenStatus_t
miopenGetSoftMarginLossForwardWorkspaceSize(miopenHandle_t handle,
                                            miopenTensorDescriptor_t inputDesc,
                                            miopenTensorDescriptor_t targetDesc,
                                            miopenTensorDescriptor_t outputDesc,
                                            miopenLossReductionMode_t reduction,
                                            size_t* sizeInBytes)
{
    if(miopen::wrapper::Dispatch("miopenGetSoftMarginLossForwardWorkspaceSize") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetSoftMarginLossForwardWorkspaceSize");
    return miopenGetSoftMarginLossForwardWorkspaceSize_impl(
        handle, inputDesc, targetDesc, outputDesc, reduction, sizeInBytes);
}

extern "C" miopenStatus_t miopenSoftMarginLossForward(miopenHandle_t handle,
                                                      miopenTensorDescriptor_t inputDesc,
                                                      const void* input,
                                                      miopenTensorDescriptor_t targetDesc,
                                                      const void* target,
                                                      miopenTensorDescriptor_t outputDesc,
                                                      void* output,
                                                      miopenLossReductionMode_t reduction,
                                                      void* workspace,
                                                      size_t workspaceSizeInBytes)
{
    if(miopen::wrapper::Dispatch("miopenSoftMarginLossForward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSoftMarginLossForward");
    return miopenSoftMarginLossForward_impl(handle,
                                            inputDesc,
                                            input,
                                            targetDesc,
                                            target,
                                            outputDesc,
                                            output,
                                            reduction,
                                            workspace,
                                            workspaceSizeInBytes);
}

extern "C" miopenStatus_t miopenSoftMarginLossBackward(miopenHandle_t handle,
                                                       miopenTensorDescriptor_t inputDesc,
                                                       const void* input,
                                                       miopenTensorDescriptor_t targetDesc,
                                                       const void* target,
                                                       miopenTensorDescriptor_t doutputDesc,
                                                       const void* doutput,
                                                       miopenTensorDescriptor_t dinputDesc,
                                                       void* dinput,
                                                       miopenLossReductionMode_t reduction)
{
    if(miopen::wrapper::Dispatch("miopenSoftMarginLossBackward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSoftMarginLossBackward");
    return miopenSoftMarginLossBackward_impl(handle,
                                             inputDesc,
                                             input,
                                             targetDesc,
                                             target,
                                             doutputDesc,
                                             doutput,
                                             dinputDesc,
                                             dinput,
                                             reduction);
}

extern "C" miopenStatus_t
miopenGetMultiMarginLossForwardWorkspaceSize(miopenHandle_t handle,
                                             miopenTensorDescriptor_t inputDesc,
                                             miopenTensorDescriptor_t targetDesc,
                                             miopenTensorDescriptor_t weightDesc,
                                             miopenTensorDescriptor_t outputDesc,
                                             long p,
                                             float margin,
                                             miopenLossReductionMode_t reduction,
                                             size_t* sizeInBytes)
{
    if(miopen::wrapper::Dispatch("miopenGetMultiMarginLossForwardWorkspaceSize") ==
       miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetMultiMarginLossForwardWorkspaceSize");
    return miopenGetMultiMarginLossForwardWorkspaceSize_impl(
        handle, inputDesc, targetDesc, weightDesc, outputDesc, p, margin, reduction, sizeInBytes);
}

extern "C" miopenStatus_t miopenMultiMarginLossForward(miopenHandle_t handle,
                                                       miopenTensorDescriptor_t inputDesc,
                                                       const void* input,
                                                       miopenTensorDescriptor_t targetDesc,
                                                       const void* target,
                                                       miopenTensorDescriptor_t weightDesc,
                                                       const void* weight,
                                                       miopenTensorDescriptor_t outputDesc,
                                                       void* output,
                                                       long p,
                                                       float margin,
                                                       miopenLossReductionMode_t reduction,
                                                       void* workspace,
                                                       size_t workspaceSizeInBytes)
{
    if(miopen::wrapper::Dispatch("miopenMultiMarginLossForward") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenMultiMarginLossForward");
    return miopenMultiMarginLossForward_impl(handle,
                                             inputDesc,
                                             input,
                                             targetDesc,
                                             target,
                                             weightDesc,
                                             weight,
                                             outputDesc,
                                             output,
                                             p,
                                             margin,
                                             reduction,
                                             workspace,
                                             workspaceSizeInBytes);
}

extern "C" miopenStatus_t miopenSetTuningPolicy(miopenHandle_t handle,
                                                miopenTuningPolicy_t newValue)
{
    if(miopen::wrapper::Dispatch("miopenSetTuningPolicy") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenSetTuningPolicy");
    return miopenSetTuningPolicy_impl(handle, newValue);
}

extern "C" miopenStatus_t miopenGetTuningPolicy(miopenHandle_t handle, miopenTuningPolicy_t* value)
{
    if(miopen::wrapper::Dispatch("miopenGetTuningPolicy") == miopen::wrapper::Route::Hipdnn)
        return forward_to_hipdnn("miopenGetTuningPolicy");
    return miopenGetTuningPolicy_impl(handle, value);
}
