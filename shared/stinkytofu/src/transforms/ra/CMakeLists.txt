# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(stinkytofu_objs PRIVATE
    PhysRegMatrix.cpp
    AllocationConstraints.cpp
    AllocationVerifier.cpp
    AllocatorRegistry.cpp
    RegisterAllocationPass.cpp
    allocators/LegacyIdentityAllocator.cpp
)
