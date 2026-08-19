/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include <iostream>
#include <rocsparse/rocsparse.h>

#define HIP_CHECK(stat)                                                                       \
    {                                                                                         \
        if(stat != hipSuccess)                                                                \
        {                                                                                     \
            std::cerr << "Error: hip error " << stat << " in line " << __LINE__ << std::endl; \
            return -1;                                                                        \
        }                                                                                     \
    }

#define ROCSPARSE_CHECK(stat)                                                         \
    {                                                                                 \
        if(stat != rocsparse_status_success)                                          \
        {                                                                             \
            std::cerr << "Error: rocsparse error " << stat << " in line " << __LINE__ \
                      << std::endl;                                                   \
            return -1;                                                                \
        }                                                                             \
    }

//! [doc example]
int main()
{
    //
    // Define a symmetric positive definite matrix (lower triangular stored).
    //
    //     4 2 0 0
    // A = 2 8 1 0
    //     0 1 8 2
    //     0 0 2 4
    //
    static constexpr int32_t              m            = 4;
    static constexpr int32_t              nnz          = 7;
    static constexpr rocsparse_index_base idx_base     = rocsparse_index_base_zero;
    static constexpr rocsparse_indextype  row_idx_type = rocsparse_indextype_i32;
    static constexpr rocsparse_indextype  col_idx_type = rocsparse_indextype_i32;
    static constexpr rocsparse_datatype   data_type    = rocsparse_datatype_f64_r;

    // Lower triangular part of A in CSR format
    const int32_t hcsr_row_ptr[m + 1] = {0, 1, 3, 5, 7};

    const int32_t hcsr_col_ind[nnz] = {0, 0, 1, 1, 2, 2, 3};

    const double hcsr_val[nnz] = {4.0, 2.0, 8.0, 1.0, 8.0, 2.0, 4.0};

    const double singularity_tolerance = 1e-10;

    const int32_t boost_enable    = 0;
    const double  boost_tolerance = 1e-10;
    const double  boost_value     = 1.0;

    //
    // Offload data to device
    //
    int32_t* dcsr_row_ptr;
    HIP_CHECK(hipMalloc(&dcsr_row_ptr, sizeof(int32_t) * (m + 1)));
    HIP_CHECK(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr, sizeof(int32_t) * (m + 1), hipMemcpyHostToDevice));

    int32_t* dcsr_col_ind;
    HIP_CHECK(hipMalloc(&dcsr_col_ind, sizeof(int32_t) * nnz));
    HIP_CHECK(hipMemcpy(dcsr_col_ind, hcsr_col_ind, sizeof(int32_t) * nnz, hipMemcpyHostToDevice));

    double* dcsr_val;
    HIP_CHECK(hipMalloc(&dcsr_val, sizeof(double) * nnz));
    HIP_CHECK(hipMemcpy(dcsr_val, hcsr_val, sizeof(double) * nnz, hipMemcpyHostToDevice));

    //
    // Create handle.
    //
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    //
    // Create sparse matrix A
    //
    rocsparse_spmat_descr matA;
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(&matA,
                                               m,
                                               m,
                                               nnz,
                                               dcsr_row_ptr,
                                               dcsr_col_ind,
                                               dcsr_val,
                                               row_idx_type,
                                               col_idx_type,
                                               idx_base,
                                               data_type));

    //
    // Create the descriptor of the Incomplete LDL^H algorithm of level 0.
    //
    rocsparse_spildlt0_descr spildlt0_descr;
    ROCSPARSE_CHECK(rocsparse_spildlt0_descr_create(handle, &spildlt0_descr, nullptr));

    //
    // Configure the descriptor.
    //
    const rocsparse_spildlt0_alg spildlt0_alg = rocsparse_spildlt0_alg_default;
    ROCSPARSE_CHECK(rocsparse_spildlt0_set_input(handle,
                                                 spildlt0_descr,
                                                 rocsparse_spildlt0_input_alg,
                                                 &spildlt0_alg,
                                                 sizeof(spildlt0_alg),
                                                 nullptr));

    const rocsparse_datatype spildlt0_compute_datatype = rocsparse_datatype_f64_r;
    ROCSPARSE_CHECK(rocsparse_spildlt0_set_input(handle,
                                                 spildlt0_descr,
                                                 rocsparse_spildlt0_input_compute_datatype,
                                                 &spildlt0_compute_datatype,
                                                 sizeof(spildlt0_compute_datatype),
                                                 nullptr));

    const rocsparse_analysis_policy spildlt0_analysis_policy = rocsparse_analysis_policy_reuse;
    ROCSPARSE_CHECK(rocsparse_spildlt0_set_input(handle,
                                                 spildlt0_descr,
                                                 rocsparse_spildlt0_input_analysis_policy,
                                                 &spildlt0_analysis_policy,
                                                 sizeof(spildlt0_analysis_policy),
                                                 nullptr));

    ROCSPARSE_CHECK(rocsparse_spildlt0_set_input(handle,
                                                 spildlt0_descr,
                                                 rocsparse_spildlt0_input_singularity_tolerance,
                                                 &singularity_tolerance,
                                                 sizeof(double),
                                                 nullptr));

    ROCSPARSE_CHECK(rocsparse_spildlt0_set_input(handle,
                                                 spildlt0_descr,
                                                 rocsparse_spildlt0_input_boost_enable,
                                                 &boost_enable,
                                                 sizeof(int32_t),
                                                 nullptr));

    ROCSPARSE_CHECK(rocsparse_spildlt0_set_input(handle,
                                                 spildlt0_descr,
                                                 rocsparse_spildlt0_input_boost_tolerance,
                                                 &boost_tolerance,
                                                 sizeof(double),
                                                 nullptr));

    ROCSPARSE_CHECK(rocsparse_spildlt0_set_input(handle,
                                                 spildlt0_descr,
                                                 rocsparse_spildlt0_input_boost_value,
                                                 &boost_value,
                                                 sizeof(double),
                                                 nullptr));

    hipStream_t stream;
    ROCSPARSE_CHECK(rocsparse_get_stream(handle, &stream));

    //
    // SpILDLT0 Analysis phase
    //
    size_t non_persistent_buffer_size_in_bytes;
    void*  non_persistent_buffer;

    ROCSPARSE_CHECK(rocsparse_spildlt0_buffer_size(handle,
                                                   spildlt0_descr,
                                                   matA,
                                                   matA,
                                                   rocsparse_spildlt0_stage_analysis,
                                                   &non_persistent_buffer_size_in_bytes,
                                                   nullptr));
    HIP_CHECK(hipMalloc(&non_persistent_buffer, non_persistent_buffer_size_in_bytes));

    ROCSPARSE_CHECK(rocsparse_spildlt0(handle,
                                       spildlt0_descr,
                                       matA,
                                       matA,
                                       rocsparse_spildlt0_stage_analysis,
                                       non_persistent_buffer_size_in_bytes,
                                       non_persistent_buffer,
                                       nullptr));

    //
    // Check for any singularities after analysis.
    //
    ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    rocsparse_singularity post_analysis_singularity;
    ROCSPARSE_CHECK(rocsparse_spildlt0_get_output(handle,
                                                  spildlt0_descr,
                                                  rocsparse_spildlt0_output_singularity,
                                                  &post_analysis_singularity,
                                                  sizeof(rocsparse_singularity),
                                                  nullptr));

    int64_t singularity_position;
    ROCSPARSE_CHECK(rocsparse_spildlt0_get_output(handle,
                                                  spildlt0_descr,
                                                  rocsparse_spildlt0_output_singularity_position,
                                                  &singularity_position,
                                                  sizeof(int64_t),
                                                  nullptr));
    HIP_CHECK(hipStreamSynchronize(stream));

    switch(post_analysis_singularity)
    {
    case rocsparse_singularity_none:
    {
        break;
    }
    case rocsparse_singularity_symbolic:
    {
        std::cout << "symbolic singularity detected at position: " << singularity_position
                  << std::endl;
        ROCSPARSE_CHECK(rocsparse_status_zero_pivot);
        break;
    }
    case rocsparse_singularity_numeric_exact:
    case rocsparse_singularity_numeric_near:
    {
        ROCSPARSE_CHECK(rocsparse_status_internal_error);
        break;
    }
    }

    //
    // Compute phase.
    //
    ROCSPARSE_CHECK(rocsparse_spildlt0_buffer_size(handle,
                                                   spildlt0_descr,
                                                   matA,
                                                   matA,
                                                   rocsparse_spildlt0_stage_compute,
                                                   &non_persistent_buffer_size_in_bytes,
                                                   nullptr));
    HIP_CHECK(hipFree(non_persistent_buffer));
    non_persistent_buffer = nullptr;
    HIP_CHECK(hipMalloc(&non_persistent_buffer, non_persistent_buffer_size_in_bytes));

    ROCSPARSE_CHECK(rocsparse_spildlt0(handle,
                                       spildlt0_descr,
                                       matA,
                                       matA,
                                       rocsparse_spildlt0_stage_compute,
                                       non_persistent_buffer_size_in_bytes,
                                       non_persistent_buffer,
                                       nullptr));

    //
    // Check for any singularities after compute.
    //
    rocsparse_singularity post_compute_singularity;
    ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    ROCSPARSE_CHECK(rocsparse_spildlt0_get_output(handle,
                                                  spildlt0_descr,
                                                  rocsparse_spildlt0_output_singularity,
                                                  &post_compute_singularity,
                                                  sizeof(rocsparse_singularity),
                                                  nullptr));

    ROCSPARSE_CHECK(rocsparse_spildlt0_get_output(handle,
                                                  spildlt0_descr,
                                                  rocsparse_spildlt0_output_singularity_position,
                                                  &singularity_position,
                                                  sizeof(int64_t),
                                                  nullptr));
    HIP_CHECK(hipStreamSynchronize(stream));

    switch(post_compute_singularity)
    {
    case rocsparse_singularity_none:
    {
        break;
    }
    case rocsparse_singularity_symbolic:
    {
        std::cout << "numeric symbolic singularity detected at position: " << singularity_position
                  << std::endl;
        ROCSPARSE_CHECK(rocsparse_status_internal_error);
        break;
    }
    case rocsparse_singularity_numeric_exact:
    {
        std::cout << "numeric exact singularity detected at position: " << singularity_position
                  << std::endl;
        break;
    }
    case rocsparse_singularity_numeric_near:
    {
        std::cout << "numeric near singularity detected at position: " << singularity_position
                  << std::endl;
        break;
    }
    }

#if defined(ROCSPARSE_WITH_DIAGONAL_SOLVE)
    //
    // Back-solve A x = b using the computed factor M ~= L |D| L^H.
    //
    // The compute stage overwrote matA in place: its strictly-lower part holds L
    // (with an implicit unit diagonal) and its diagonal holds D. Solving A x = b
    // therefore splits into three solves that all reuse the same factor matrix:
    //
    //     L y = b     (lower, unit diagonal, regular triangular solve)
    //     |D| z = y   (diagonal-only solve, absolute mode)
    //     L^H x = z   (lower + transpose, unit diagonal, regular triangular solve)
    //
    // The absolute diagonal mode divides by |D|, which turns the incomplete factor
    // into the SPD operator L |D| L^H -- a valid preconditioner for symmetric
    // Krylov methods even when D is indefinite.
    //

    // Right-hand side b = [1, 1, 1, 1] plus work vectors y, z and solution x.
    const double hb[m] = {1.0, 1.0, 1.0, 1.0};

    double* d_b;
    double* d_y;
    double* d_z;
    double* d_x;
    HIP_CHECK(hipMalloc(&d_b, sizeof(double) * m));
    HIP_CHECK(hipMalloc(&d_y, sizeof(double) * m));
    HIP_CHECK(hipMalloc(&d_z, sizeof(double) * m));
    HIP_CHECK(hipMalloc(&d_x, sizeof(double) * m));
    HIP_CHECK(hipMemcpy(d_b, hb, sizeof(double) * m, hipMemcpyHostToDevice));

    rocsparse_dnvec_descr vecB, vecY, vecZ, vecX;
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vecB, m, d_b, data_type));
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vecY, m, d_y, data_type));
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vecZ, m, d_z, data_type));
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vecX, m, d_x, data_type));

    // The factor is lower triangular with an implicit unit diagonal (L); D lives on
    // the diagonal and is applied through the diagonal solve mode.
    const rocsparse_fill_mode factor_fill_mode = rocsparse_fill_mode_lower;
    const rocsparse_diag_type factor_diag_type = rocsparse_diag_type_unit;
    ROCSPARSE_CHECK(rocsparse_spmat_set_attribute(
        matA, rocsparse_spmat_fill_mode, &factor_fill_mode, sizeof(factor_fill_mode)));
    ROCSPARSE_CHECK(rocsparse_spmat_set_attribute(
        matA, rocsparse_spmat_diag_type, &factor_diag_type, sizeof(factor_diag_type)));

    const double sptrsv_alpha = 1.0;

    // Helper running one SpTRSV stage, allocating and freeing its temporary buffer.
    auto sptrsv_run_stage = [&](rocsparse_sptrsv_descr descr,
                                rocsparse_dnvec_descr  vec_in,
                                rocsparse_dnvec_descr  vec_out,
                                rocsparse_sptrsv_stage stage) -> int {
        size_t buffer_size;
        void*  buffer = nullptr;
        ROCSPARSE_CHECK(rocsparse_sptrsv_buffer_size(
            handle, descr, matA, vec_in, vec_out, stage, &buffer_size, nullptr));
        HIP_CHECK(hipMalloc(&buffer, buffer_size));
        ROCSPARSE_CHECK(rocsparse_sptrsv(
            handle, descr, matA, vec_in, vec_out, stage, buffer_size, buffer, nullptr));
        HIP_CHECK(hipFree(buffer));
        return 0;
    };

    const rocsparse_sptrsv_alg      sptrsv_alg   = rocsparse_sptrsv_alg_default;
    const rocsparse_datatype        sptrsv_dtype = rocsparse_datatype_f64_r;
    const rocsparse_analysis_policy sptrsv_apol  = rocsparse_analysis_policy_reuse;

    //
    // Descriptor #1 (operation = none): solves L y = b, then |D| z = y by toggling
    // the diagonal mode between the two compute calls. The analysis is shared
    // because both reuse the same op(A).
    //
    rocsparse_sptrsv_descr sptrsv_descr;
    ROCSPARSE_CHECK(rocsparse_create_sptrsv_descr(&sptrsv_descr));

    const rocsparse_operation op_none = rocsparse_operation_none;
    ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                               sptrsv_descr,
                                               rocsparse_sptrsv_input_alg,
                                               &sptrsv_alg,
                                               sizeof(sptrsv_alg),
                                               nullptr));
    ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                               sptrsv_descr,
                                               rocsparse_sptrsv_input_operation,
                                               &op_none,
                                               sizeof(op_none),
                                               nullptr));
    ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                               sptrsv_descr,
                                               rocsparse_sptrsv_input_scalar_datatype,
                                               &sptrsv_dtype,
                                               sizeof(sptrsv_dtype),
                                               nullptr));
    ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                               sptrsv_descr,
                                               rocsparse_sptrsv_input_compute_datatype,
                                               &sptrsv_dtype,
                                               sizeof(sptrsv_dtype),
                                               nullptr));
    ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                               sptrsv_descr,
                                               rocsparse_sptrsv_input_analysis_policy,
                                               &sptrsv_apol,
                                               sizeof(sptrsv_apol),
                                               nullptr));
    ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                               sptrsv_descr,
                                               rocsparse_sptrsv_input_scalar_alpha,
                                               &sptrsv_alpha,
                                               sizeof(&sptrsv_alpha),
                                               nullptr));

    // Shared analysis for op(A) = A (lower).
    if(sptrsv_run_stage(sptrsv_descr, vecB, vecY, rocsparse_sptrsv_stage_analysis) != 0)
    {
        return -1;
    }

    // L y = b (regular lower solve).
    rocsparse_diagonal_mode diag_mode = rocsparse_diagonal_mode_none;
    ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                               sptrsv_descr,
                                               rocsparse_sptrsv_input_diagonal_mode,
                                               &diag_mode,
                                               sizeof(diag_mode),
                                               nullptr));
    if(sptrsv_run_stage(sptrsv_descr, vecB, vecY, rocsparse_sptrsv_stage_compute) != 0)
    {
        return -1;
    }

    // |D| z = y (diagonal-only solve, absolute mode) reusing the same analysis.
    diag_mode = rocsparse_diagonal_mode_absolute;
    ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                               sptrsv_descr,
                                               rocsparse_sptrsv_input_diagonal_mode,
                                               &diag_mode,
                                               sizeof(diag_mode),
                                               nullptr));
    if(sptrsv_run_stage(sptrsv_descr, vecY, vecZ, rocsparse_sptrsv_stage_compute) != 0)
    {
        return -1;
    }

    ROCSPARSE_CHECK(rocsparse_destroy_sptrsv_descr(sptrsv_descr));

    //
    // Descriptor #2 (operation = transpose): solves L^H x = z. The transpose needs
    // its own analysis, so it uses a separate descriptor.
    //
    rocsparse_sptrsv_descr sptrsv_descr_t;
    ROCSPARSE_CHECK(rocsparse_create_sptrsv_descr(&sptrsv_descr_t));

    const rocsparse_operation     op_transpose = rocsparse_operation_transpose;
    const rocsparse_diagonal_mode diag_none    = rocsparse_diagonal_mode_none;
    ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                               sptrsv_descr_t,
                                               rocsparse_sptrsv_input_alg,
                                               &sptrsv_alg,
                                               sizeof(sptrsv_alg),
                                               nullptr));
    ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                               sptrsv_descr_t,
                                               rocsparse_sptrsv_input_operation,
                                               &op_transpose,
                                               sizeof(op_transpose),
                                               nullptr));
    ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                               sptrsv_descr_t,
                                               rocsparse_sptrsv_input_scalar_datatype,
                                               &sptrsv_dtype,
                                               sizeof(sptrsv_dtype),
                                               nullptr));
    ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                               sptrsv_descr_t,
                                               rocsparse_sptrsv_input_compute_datatype,
                                               &sptrsv_dtype,
                                               sizeof(sptrsv_dtype),
                                               nullptr));
    ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                               sptrsv_descr_t,
                                               rocsparse_sptrsv_input_analysis_policy,
                                               &sptrsv_apol,
                                               sizeof(sptrsv_apol),
                                               nullptr));
    ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                               sptrsv_descr_t,
                                               rocsparse_sptrsv_input_diagonal_mode,
                                               &diag_none,
                                               sizeof(diag_none),
                                               nullptr));
    ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                               sptrsv_descr_t,
                                               rocsparse_sptrsv_input_scalar_alpha,
                                               &sptrsv_alpha,
                                               sizeof(&sptrsv_alpha),
                                               nullptr));

    if(sptrsv_run_stage(sptrsv_descr_t, vecZ, vecX, rocsparse_sptrsv_stage_analysis) != 0)
    {
        return -1;
    }
    if(sptrsv_run_stage(sptrsv_descr_t, vecZ, vecX, rocsparse_sptrsv_stage_compute) != 0)
    {
        return -1;
    }

    ROCSPARSE_CHECK(rocsparse_destroy_sptrsv_descr(sptrsv_descr_t));

    //
    // Copy the solution back to the host and print it.
    //
    HIP_CHECK(hipStreamSynchronize(stream));
    double hx[m];
    HIP_CHECK(hipMemcpy(hx, d_x, sizeof(double) * m, hipMemcpyDeviceToHost));

    std::cout << "Solution x of the L |D| L^H back-solve:";
    for(int32_t i = 0; i < m; ++i)
    {
        std::cout << " " << hx[i];
    }
    std::cout << std::endl;

    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vecB));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vecY));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vecZ));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vecX));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_y));
    HIP_CHECK(hipFree(d_z));
    HIP_CHECK(hipFree(d_x));
#endif

    HIP_CHECK(hipFree(non_persistent_buffer));

    ROCSPARSE_CHECK(rocsparse_spildlt0_descr_destroy(handle, spildlt0_descr, nullptr));

    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(matA));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    HIP_CHECK(hipFree(dcsr_row_ptr));
    HIP_CHECK(hipFree(dcsr_col_ind));
    HIP_CHECK(hipFree(dcsr_val));

    return 0;
}
//! [doc example]
