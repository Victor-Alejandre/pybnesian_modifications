#include <kde/SCV.hpp>
#include <kde/NormalReferenceRule.hpp>
#include <util/math_constants.hpp>
#include <util/vech_ops.hpp>
#include <nlopt.hpp>
#include <vector>
#include <cmath>
#include <unordered_set>
#include <algorithm>
#include <Eigen/Dense>
#include <iostream>

using Eigen::LLT;
using opencl::OpenCLConfig, opencl::OpenCL_kernel_traits;

// 1e-3 cambiado de tolerancia para nelder mead
namespace kde {

class UnivariateSCVScore {
public:
    template <typename ArrowType>
    static void AB_criterion(const cl::Buffer& training_data,
                                                 const unsigned int training_rows,
                                                 const unsigned int training_cols,
                                                 const unsigned int derivate_order,
                                                 typename ArrowType::c_type gaussian_const,
                                                 const cl::Buffer& cholesky,
                                                 const cl::Buffer& vec_H,
                                                 const cl::Buffer&,
                                                 const cl::Buffer& inv_bandwidth,
                                                 cl::Buffer&,
                                                 cl::Buffer&,
                                                 cl::Buffer&,
                                                 cl::Buffer& psi_r_plus_2,
                                                 cl::Buffer&,
                                                 cl::Buffer&,
                                                 cl::Buffer& mse);
    template <typename ArrowType>
    static void Psi_r(const cl::Buffer& training_data,
                                                 const unsigned int training_rows,
                                                 const unsigned int training_cols,
                                                 const unsigned int index_offset,
                                                 const unsigned int length,
                                                 const unsigned int derivate_order,
                                                 typename ArrowType::c_type gaussian_const,
                                                 const cl::Buffer& cholesky,
                                                 cl::Buffer&,
                                                 const cl::Buffer&,
                                                 const cl::Buffer& vec_inv_bandwidth,
                                                 cl::Buffer&,
                                                 cl::Buffer&,
                                                 cl::Buffer&,
                                                 cl::Buffer&,
                                                 cl::Buffer& psi_r);                                             
    template <typename ArrowType>
    static void scv(const cl::Buffer& training_data,
                                                 const unsigned int training_rows,
                                                 const unsigned int training_cols,
                                                 const unsigned int index_offset,
                                                 const unsigned int length,
                                                 typename ArrowType::c_type gaussian_const_2H_2G,
                                                 typename ArrowType::c_type gaussian_const_H_2G,
                                                 typename ArrowType::c_type gaussian_const_G,
                                                 const cl::Buffer& cholesky_2H_2G,
                                                 const cl::Buffer& cholesky_H_2G,
                                                 cl::Buffer& cholesky_G,
                                                 cl::Buffer&,
                                                 cl::Buffer&,
                                                 cl::Buffer&,
                                                 cl::Buffer& result);
};

template <typename ArrowType>
void UnivariateSCVScore::AB_criterion(const cl::Buffer& training_data,
                                                 const unsigned int training_rows,
                                                 const unsigned int training_cols,
                                                 const unsigned int derivate_order,
                                                 typename ArrowType::c_type gaussian_const,
                                                 const cl::Buffer& cholesky,
                                                 const cl::Buffer& vec_H,
                                                 const cl::Buffer&,
                                                 const cl::Buffer& inv_bandwidth,
                                                 cl::Buffer&,
                                                 cl::Buffer&,
                                                 cl::Buffer&,
                                                 cl::Buffer& psi_r_plus_2,
                                                 cl::Buffer&,
                                                 cl::Buffer&,
                                                 cl::Buffer& mse) {
                                                    
    auto& opencl = OpenCLConfig::get();
    auto& k_ab_criterion_1d = opencl.kernel(OpenCL_kernel_traits<ArrowType>::ab_criterion_1d);
    k_ab_criterion_1d.setArg(0, training_data);
    k_ab_criterion_1d.setArg(1, vec_H);
    k_ab_criterion_1d.setArg(2, derivate_order);
    k_ab_criterion_1d.setArg(3, training_rows);
    k_ab_criterion_1d.setArg(4, gaussian_const);
    k_ab_criterion_1d.setArg(5, psi_r_plus_2);
    k_ab_criterion_1d.setArg(6, mse);

    auto& queue = opencl.queue();
    RAISE_ENQUEUEKERNEL_ERROR(
        queue.enqueueNDRangeKernel(k_ab_criterion_1d, cl::NullRange, cl::NDRange(1), cl::NullRange));
}

template <typename ArrowType>
void UnivariateSCVScore::Psi_r(const cl::Buffer& training_data,
                                                 const unsigned int training_rows,
                                                 const unsigned int training_cols,
                                                 const unsigned int index_offset,
                                                 const unsigned int length,
                                                 const unsigned int derivate_order,
                                                 typename ArrowType::c_type gaussian_const,
                                                 const cl::Buffer& cholesky,
                                                 cl::Buffer&,
                                                 const cl::Buffer&,
                                                 const cl::Buffer& vec_inv_bandwidth,
                                                 cl::Buffer&,
                                                 cl::Buffer&,
                                                 cl::Buffer&,
                                                 cl::Buffer& diff_derivates,
                                                 cl::Buffer& psi_r) {
    auto& opencl = OpenCLConfig::get();
    auto& k_sum_bcv_1d = opencl.kernel(OpenCL_kernel_traits<ArrowType>::sum_bcv_1d);
    k_sum_bcv_1d.setArg(0, training_data);
    k_sum_bcv_1d.setArg(1, index_offset);
    k_sum_bcv_1d.setArg(2, cholesky);
    k_sum_bcv_1d.setArg(3, derivate_order);
    k_sum_bcv_1d.setArg(4, gaussian_const);
    k_sum_bcv_1d.setArg(5, diff_derivates);

    auto& queue = opencl.queue();
    RAISE_ENQUEUEKERNEL_ERROR(
        queue.enqueueNDRangeKernel(k_sum_bcv_1d, cl::NullRange, cl::NDRange(length), cl::NullRange));

    unsigned int derivate_cols = std::pow(training_cols, derivate_order);
    auto& k_sum_bcv_mat = opencl.kernel(OpenCL_kernel_traits<ArrowType>::sum_bcv_mat);
    k_sum_bcv_mat.setArg(0, diff_derivates);
    k_sum_bcv_mat.setArg(1, derivate_cols);
    k_sum_bcv_mat.setArg(2, length);
    k_sum_bcv_mat.setArg(3, psi_r);
    RAISE_ENQUEUEKERNEL_ERROR(
        queue.enqueueNDRangeKernel(k_sum_bcv_mat, cl::NullRange, cl::NDRange(1), cl::NullRange)); 
  
}

template <typename ArrowType>
void UnivariateSCVScore::scv(const cl::Buffer& training_data,
                                                 const unsigned int training_rows,
                                                 const unsigned int training_cols,
                                                 const unsigned int index_offset,
                                                 const unsigned int length,
                                                 typename ArrowType::c_type gaussian_const_2H_2G,
                                                 typename ArrowType::c_type gaussian_const_H_2G,
                                                 typename ArrowType::c_type gaussian_const_G,
                                                 const cl::Buffer& cholesky_2H_2G,
                                                 const cl::Buffer& cholesky_H_2G,
                                                 cl::Buffer& cholesky_G,
                                                 cl::Buffer&,
                                                 cl::Buffer&,
                                                 cl::Buffer&,
                                                 cl::Buffer& result){                                             
    auto& opencl = OpenCLConfig::get();
    auto& k_scv_1d = opencl.kernel(OpenCL_kernel_traits<ArrowType>::scv_1d);
    k_scv_1d.setArg(0, training_data);
    k_scv_1d.setArg(1, cholesky_2H_2G);
    k_scv_1d.setArg(2, cholesky_H_2G);
    k_scv_1d.setArg(3, cholesky_G);
    k_scv_1d.setArg(4, index_offset);
    k_scv_1d.setArg(5, gaussian_const_2H_2G);
    k_scv_1d.setArg(6, gaussian_const_H_2G);
    k_scv_1d.setArg(7, gaussian_const_G);
    k_scv_1d.setArg(8, result);
    
    auto& queue = opencl.queue();
    RAISE_ENQUEUEKERNEL_ERROR(queue.enqueueNDRangeKernel(
        k_scv_1d, cl::NullRange, cl::NDRange(length), cl::NullRange));

}

class ProductSCVScore {
public:
    template <typename ArrowType>
    static void sum_triangular_scores(const cl::Buffer& training_data,
                                      const unsigned int,
                                      const unsigned int,
                                      const unsigned int index_offset,
                                      const unsigned int length,
                                      const cl::Buffer& cholesky,
                                      typename ArrowType::c_type lognorm_2H,
                                      typename ArrowType::c_type lognorm_H,
                                      cl::Buffer&,
                                      cl::Buffer& output_2h,
                                      cl::Buffer& output_h);
};

template <typename ArrowType>
void ProductSCVScore::sum_triangular_scores(const cl::Buffer& training_data,
                                            const unsigned int training_rows,
                                            const unsigned int training_cols,
                                            const unsigned int index_offset,
                                            const unsigned int length,
                                            const cl::Buffer& h_vector,
                                            typename ArrowType::c_type lognorm_2H,
                                            typename ArrowType::c_type lognorm_H,
                                            cl::Buffer& tmp_h,
                                            cl::Buffer& output_2h,
                                            cl::Buffer& output_h) {
    auto& opencl = OpenCLConfig::get();
    auto& k_ucv_diag = opencl.kernel(OpenCL_kernel_traits<ArrowType>::ucv_diag);
    k_ucv_diag.setArg(0, training_data);
    k_ucv_diag.setArg(1, index_offset);
    k_ucv_diag.setArg(2, h_vector);
    k_ucv_diag.setArg(3, tmp_h);
    auto& queue = opencl.queue();
    RAISE_ENQUEUEKERNEL_ERROR(
        queue.enqueueNDRangeKernel(k_ucv_diag, cl::NullRange, cl::NDRange(length), cl::NullRange));

    auto& k_sum_ucv_diag = opencl.kernel(OpenCL_kernel_traits<ArrowType>::sum_ucv_diag);
    k_sum_ucv_diag.setArg(0, training_data);
    k_sum_ucv_diag.setArg(1, training_rows);
    k_sum_ucv_diag.setArg(2, index_offset);
    k_sum_ucv_diag.setArg(3, h_vector);
    k_sum_ucv_diag.setArg(5, tmp_h);

    for (unsigned int i = 1; i < training_cols; ++i) {
        k_sum_ucv_diag.setArg(4, i);
        RAISE_ENQUEUEKERNEL_ERROR(
            queue.enqueueNDRangeKernel(k_sum_ucv_diag, cl::NullRange, cl::NDRange(length), cl::NullRange));
    }

    auto& k_copy_ucv_diag = opencl.kernel(OpenCL_kernel_traits<ArrowType>::copy_ucv_diag);
    k_copy_ucv_diag.setArg(0, tmp_h);
    k_copy_ucv_diag.setArg(1, lognorm_2H);
    k_copy_ucv_diag.setArg(2, lognorm_H);
    k_copy_ucv_diag.setArg(3, output_2h);
    k_copy_ucv_diag.setArg(4, output_h);
    RAISE_ENQUEUEKERNEL_ERROR(
        queue.enqueueNDRangeKernel(k_copy_ucv_diag, cl::NullRange, cl::NDRange(length), cl::NullRange));
}

class MultivariateSCVScore {
public:
    template <typename ArrowType>
    static void AB_criterion(const cl::Buffer& training_data,
                                                 const unsigned int training_rows,
                                                 const unsigned int training_cols,
                                                 const unsigned int derivate_order,
                                                 typename ArrowType::c_type gaussian_const,
                                                 const cl::Buffer& cholesky,
                                                 const cl::Buffer& vec_H,
                                                 const cl::Buffer& gaussian_density_values,
                                                 const cl::Buffer& inv_bandwidth,
                                                 cl::Buffer& w_new,
                                                 cl::Buffer& u_k_1,
                                                 cl::Buffer& u_k_2,
                                                 cl::Buffer& psi_r_plus_2,
                                                 cl::Buffer& diff_derivates,
                                                 cl::Buffer& result_kron_Idr_psi,
                                                 cl::Buffer& mse) ;

    template <typename ArrowType>
    static void Psi_r(const cl::Buffer& training_data,
                                                 const unsigned int training_rows,
                                                 const unsigned int training_cols,
                                                 const unsigned int index_offset,
                                                 const unsigned int length,
                                                 const unsigned int derivate_order,
                                                 typename ArrowType::c_type gaussian_const,
                                                 const cl::Buffer& cholesky,
                                                 cl::Buffer& tmp_diff_mat,
                                                 const cl::Buffer& gaussian_density_values,
                                                 const cl::Buffer& vec_inv_bandwidth,
                                                 cl::Buffer& w_new,
                                                 cl::Buffer& u_k_1,
                                                 cl::Buffer& u_k_2,
                                                 cl::Buffer& diff_derivates,
                                                 cl::Buffer& psi_r);                                            
    template <typename ArrowType>
    static void scv(const cl::Buffer& training_data,
                                                 const unsigned int training_rows,
                                                 const unsigned int training_cols,
                                                 const unsigned int index_offset,
                                                 const unsigned int length,
                                                 typename ArrowType::c_type gaussian_const_2H_2G,
                                                 typename ArrowType::c_type gaussian_const_H_2G,
                                                 typename ArrowType::c_type gaussian_const_G,
                                                 const cl::Buffer& cholesky_2H_2G,
                                                 const cl::Buffer& cholesky_H_2G,
                                                 cl::Buffer& cholesky_G,
                                                 cl::Buffer& tmp_diff_mat_2H_2G,
                                                 cl::Buffer& tmp_diff_mat_H_2G,
                                                 cl::Buffer& tmp_diff_mat_G,
                                                 cl::Buffer& result);
};

template <typename ArrowType>
void MultivariateSCVScore::AB_criterion(const cl::Buffer& training_data,
                                                 const unsigned int training_rows,
                                                 const unsigned int training_cols,
                                                 const unsigned int derivate_order,
                                                 typename ArrowType::c_type gaussian_const,
                                                 const cl::Buffer& cholesky,
                                                 const cl::Buffer& vec_H,
                                                 const cl::Buffer& gaussian_density_values,
                                                 const cl::Buffer& inv_bandwidth,
                                                 cl::Buffer& w_new,
                                                 cl::Buffer& u_k_1,
                                                 cl::Buffer& u_k_2,
                                                 cl::Buffer& psi_r_plus_2,
                                                 cl::Buffer& diff_derivates,
                                                 cl::Buffer& result_kron_Idr_psi,
                                                 cl::Buffer& mse) {
    auto& opencl = OpenCLConfig::get();    
    auto& queue = opencl.queue();
    auto& k_gaussian_density = opencl.kernel(OpenCL_kernel_traits<ArrowType>::gaussian_density);
    k_gaussian_density.setArg(0, training_data);
    k_gaussian_density.setArg(1, training_cols);
    k_gaussian_density.setArg(2, gaussian_const);
    k_gaussian_density.setArg(3, gaussian_density_values);

    RAISE_ENQUEUEKERNEL_ERROR(
        queue.enqueueNDRangeKernel(k_gaussian_density, cl::NullRange, cl::NDRange(1), cl::NullRange));


    unsigned int derivate_cols = std::pow(training_cols, derivate_order);
    unsigned int instances = 1;
    auto& k_higher_derivate = opencl.kernel(OpenCL_kernel_traits<ArrowType>::complete_higher_derivate);
    k_higher_derivate.setArg(0, training_data);
    k_higher_derivate.setArg(1, instances);
    k_higher_derivate.setArg(2, training_cols);
    k_higher_derivate.setArg(3, derivate_order);
    k_higher_derivate.setArg(4, derivate_cols);
    k_higher_derivate.setArg(5, gaussian_density_values);
    k_higher_derivate.setArg(6, inv_bandwidth);
    k_higher_derivate.setArg(7, w_new);
    k_higher_derivate.setArg(8, u_k_1);
    k_higher_derivate.setArg(9, u_k_2);
    k_higher_derivate.setArg(10, diff_derivates);
    RAISE_ENQUEUEKERNEL_ERROR(queue.enqueueNDRangeKernel(k_higher_derivate, cl::NullRange, cl::NDRange(1), cl::NullRange));

    auto& k_vecG_kron_Idr_psi = opencl.kernel(OpenCL_kernel_traits<ArrowType>::vecG_kron_Idr_psi);
    k_vecG_kron_Idr_psi.setArg(0, vec_H);
    k_vecG_kron_Idr_psi.setArg(1, psi_r_plus_2);
    k_vecG_kron_Idr_psi.setArg(2, result_kron_Idr_psi);
    k_vecG_kron_Idr_psi.setArg(3, derivate_cols);
    k_vecG_kron_Idr_psi.setArg(4, training_cols);
    k_vecG_kron_Idr_psi.setArg(5, derivate_order);

    RAISE_ENQUEUEKERNEL_ERROR(
        queue.enqueueNDRangeKernel(k_vecG_kron_Idr_psi, cl::NullRange, cl::NDRange(derivate_cols), cl::NullRange));

    auto& k_sum_mse = opencl.kernel(OpenCL_kernel_traits<ArrowType>::sum_mse);
    k_sum_mse.setArg(0, diff_derivates);
    k_sum_mse.setArg(1, training_rows);
    k_sum_mse.setArg(2, result_kron_Idr_psi);
    k_sum_mse.setArg(3, mse);

    RAISE_ENQUEUEKERNEL_ERROR(
        queue.enqueueNDRangeKernel(k_sum_mse, cl::NullRange, cl::NDRange(derivate_cols), cl::NullRange));

}

template <typename ArrowType>
void MultivariateSCVScore::Psi_r(const cl::Buffer& training_data,
                                                 const unsigned int training_rows,
                                                 const unsigned int training_cols,
                                                 const unsigned int index_offset,
                                                 const unsigned int length,
                                                 const unsigned int derivate_order,
                                                 typename ArrowType::c_type gaussian_const,
                                                 const cl::Buffer& cholesky,
                                                 cl::Buffer& tmp_diff_mat,
                                                 const cl::Buffer& gaussian_density_values,
                                                 const cl::Buffer& vec_inv_bandwidth,
                                                 cl::Buffer& w_new,
                                                 cl::Buffer& u_k_1,
                                                 cl::Buffer& u_k_2,
                                                 cl::Buffer& diff_derivates,
                                                 cl::Buffer& psi_r){
    auto& opencl = OpenCLConfig::get();    

    auto& k_triangular_substract_mat = opencl.kernel(OpenCL_kernel_traits<ArrowType>::triangular_substract_mat);
    k_triangular_substract_mat.setArg(0, training_data);
    k_triangular_substract_mat.setArg(1, training_rows);
    k_triangular_substract_mat.setArg(2, training_cols);
    k_triangular_substract_mat.setArg(3, index_offset);
    k_triangular_substract_mat.setArg(4, length);
    k_triangular_substract_mat.setArg(5, tmp_diff_mat);
    auto& queue = opencl.queue();
    RAISE_ENQUEUEKERNEL_ERROR(queue.enqueueNDRangeKernel(
        k_triangular_substract_mat, cl::NullRange, cl::NDRange(length * training_cols), cl::NullRange));

    auto& k_solve = opencl.kernel(OpenCL_kernel_traits<ArrowType>::solve);
    k_solve.setArg(0, tmp_diff_mat);
    k_solve.setArg(1, length);
    k_solve.setArg(2, training_cols);
    k_solve.setArg(3, cholesky);
    RAISE_ENQUEUEKERNEL_ERROR(queue.enqueueNDRangeKernel(k_solve, cl::NullRange, cl::NDRange(length), cl::NullRange));

    auto& k_square = opencl.kernel(OpenCL_kernel_traits<ArrowType>::square);
    k_square.setArg(0, tmp_diff_mat);
    RAISE_ENQUEUEKERNEL_ERROR(
        queue.enqueueNDRangeKernel(k_square, cl::NullRange, cl::NDRange(length * training_cols), cl::NullRange));

    auto& k_gaussian_density = opencl.kernel(OpenCL_kernel_traits<ArrowType>::gaussian_density);
    k_gaussian_density.setArg(0, tmp_diff_mat);
    k_gaussian_density.setArg(1, training_cols);
    k_gaussian_density.setArg(2, gaussian_const);
    k_gaussian_density.setArg(3, gaussian_density_values);

    RAISE_ENQUEUEKERNEL_ERROR(
        queue.enqueueNDRangeKernel(k_gaussian_density, cl::NullRange, cl::NDRange(length), cl::NullRange));

    k_triangular_substract_mat.setArg(0, training_data);
    k_triangular_substract_mat.setArg(1, training_rows);
    k_triangular_substract_mat.setArg(2, training_cols);
    k_triangular_substract_mat.setArg(3, index_offset);
    k_triangular_substract_mat.setArg(4, length);
    k_triangular_substract_mat.setArg(5, tmp_diff_mat);

    RAISE_ENQUEUEKERNEL_ERROR(queue.enqueueNDRangeKernel(
        k_triangular_substract_mat, cl::NullRange, cl::NDRange(length * training_cols), cl::NullRange));

    auto& k_eta_1 = opencl.kernel(OpenCL_kernel_traits<ArrowType>::eta_1);
    k_eta_1.setArg(0, tmp_diff_mat);
    k_eta_1.setArg(1, length);
    k_eta_1.setArg(2, training_cols);
    k_eta_1.setArg(3, vec_inv_bandwidth);
    RAISE_ENQUEUEKERNEL_ERROR(queue.enqueueNDRangeKernel(k_eta_1, cl::NullRange, cl::NDRange(length), cl::NullRange));

    unsigned int derivate_cols = std::pow(training_cols, derivate_order);

    auto& k_higher_derivate = opencl.kernel(OpenCL_kernel_traits<ArrowType>::complete_higher_derivate);
    k_higher_derivate.setArg(0, tmp_diff_mat);
    k_higher_derivate.setArg(1, length);
    k_higher_derivate.setArg(2, training_cols);
    k_higher_derivate.setArg(3, derivate_order);
    k_higher_derivate.setArg(4, derivate_cols);
    k_higher_derivate.setArg(5, gaussian_density_values);
    k_higher_derivate.setArg(6, vec_inv_bandwidth);
    k_higher_derivate.setArg(7, w_new);
    k_higher_derivate.setArg(8, u_k_1);
    k_higher_derivate.setArg(9, u_k_2);
    k_higher_derivate.setArg(10, diff_derivates);
    RAISE_ENQUEUEKERNEL_ERROR(queue.enqueueNDRangeKernel(k_higher_derivate, cl::NullRange, cl::NDRange(length), cl::NullRange));


    auto& k_sum_bcv_mat = opencl.kernel(OpenCL_kernel_traits<ArrowType>::sum_bcv_mat);
    k_sum_bcv_mat.setArg(0, diff_derivates);
    k_sum_bcv_mat.setArg(1, derivate_cols);
    k_sum_bcv_mat.setArg(2, length);
    k_sum_bcv_mat.setArg(3, psi_r);

    RAISE_ENQUEUEKERNEL_ERROR(
        queue.enqueueNDRangeKernel(k_sum_bcv_mat, cl::NullRange, cl::NDRange(derivate_cols), cl::NullRange));
}

template <typename ArrowType>
void MultivariateSCVScore::scv(const cl::Buffer& training_data,
                                                 const unsigned int training_rows,
                                                 const unsigned int training_cols,
                                                 const unsigned int index_offset,
                                                 const unsigned int length,
                                                 typename ArrowType::c_type gaussian_const_2H_2G,
                                                 typename ArrowType::c_type gaussian_const_H_2G,
                                                 typename ArrowType::c_type gaussian_const_G,
                                                 const cl::Buffer& cholesky_2H_2G,
                                                 const cl::Buffer& cholesky_H_2G,
                                                 cl::Buffer& cholesky_G,
                                                 cl::Buffer& tmp_diff_mat_2H_2G,
                                                 cl::Buffer& tmp_diff_mat_H_2G,
                                                 cl::Buffer& tmp_diff_mat_G,
                                                 cl::Buffer& result) {                                             
    auto& opencl = OpenCLConfig::get();
    auto& k_compute_dot_product = opencl.kernel(OpenCL_kernel_traits<ArrowType>::compute_dot_product);

    auto& k_triangular_substract_mat = opencl.kernel(OpenCL_kernel_traits<ArrowType>::triangular_substract_mat);
    k_triangular_substract_mat.setArg(0, training_data);
    k_triangular_substract_mat.setArg(1, training_rows);
    k_triangular_substract_mat.setArg(2, training_cols);
    k_triangular_substract_mat.setArg(3, index_offset);
    k_triangular_substract_mat.setArg(4, length);
    k_triangular_substract_mat.setArg(5, tmp_diff_mat_G);
    auto& queue = opencl.queue();
    RAISE_ENQUEUEKERNEL_ERROR(queue.enqueueNDRangeKernel(
        k_triangular_substract_mat, cl::NullRange, cl::NDRange(length * training_cols), cl::NullRange));

    queue.enqueueCopyBuffer(tmp_diff_mat_G, tmp_diff_mat_2H_2G, 0, 0, length);
    queue.enqueueCopyBuffer(tmp_diff_mat_G, tmp_diff_mat_H_2G, 0, 0, length);

    auto& k_solve = opencl.kernel(OpenCL_kernel_traits<ArrowType>::solve);
    k_solve.setArg(0, tmp_diff_mat_G);
    k_solve.setArg(1, length);
    k_solve.setArg(2, training_cols);
    k_solve.setArg(3, cholesky_G);
    RAISE_ENQUEUEKERNEL_ERROR(queue.enqueueNDRangeKernel(k_solve, cl::NullRange, cl::NDRange(length), cl::NullRange));

    k_solve.setArg(0, tmp_diff_mat_2H_2G);
    k_solve.setArg(1, length);
    k_solve.setArg(2, training_cols);
    k_solve.setArg(3, cholesky_2H_2G);
    RAISE_ENQUEUEKERNEL_ERROR(queue.enqueueNDRangeKernel(k_solve, cl::NullRange, cl::NDRange(length), cl::NullRange));

    k_solve.setArg(0, tmp_diff_mat_H_2G);
    k_solve.setArg(1, length);
    k_solve.setArg(2, training_cols);
    k_solve.setArg(3, cholesky_H_2G);
    RAISE_ENQUEUEKERNEL_ERROR(queue.enqueueNDRangeKernel(k_solve, cl::NullRange, cl::NDRange(length), cl::NullRange));

    auto& k_square = opencl.kernel(OpenCL_kernel_traits<ArrowType>::square);
    k_square.setArg(0, tmp_diff_mat_G);
    RAISE_ENQUEUEKERNEL_ERROR(
        queue.enqueueNDRangeKernel(k_square, cl::NullRange, cl::NDRange(length * training_cols), cl::NullRange));

    k_square.setArg(0, tmp_diff_mat_2H_2G);
    RAISE_ENQUEUEKERNEL_ERROR(
        queue.enqueueNDRangeKernel(k_square, cl::NullRange, cl::NDRange(length * training_cols), cl::NullRange));
    k_square.setArg(0, tmp_diff_mat_H_2G);
    RAISE_ENQUEUEKERNEL_ERROR(
        queue.enqueueNDRangeKernel(k_square, cl::NullRange, cl::NDRange(length * training_cols), cl::NullRange));

    auto& k_sum_scv_mat = opencl.kernel(OpenCL_kernel_traits<ArrowType>::sum_scv_mat);
    k_sum_scv_mat.setArg(0, tmp_diff_mat_2H_2G);
    k_sum_scv_mat.setArg(1, tmp_diff_mat_H_2G);
    k_sum_scv_mat.setArg(2, tmp_diff_mat_G);
    k_sum_scv_mat.setArg(3, training_cols);
    k_sum_scv_mat.setArg(4, gaussian_const_2H_2G);
    k_sum_scv_mat.setArg(5, gaussian_const_H_2G);
    k_sum_scv_mat.setArg(6, gaussian_const_G);
    k_sum_scv_mat.setArg(7, result);

    RAISE_ENQUEUEKERNEL_ERROR(
        queue.enqueueNDRangeKernel(k_sum_scv_mat, cl::NullRange, cl::NDRange(length), cl::NullRange));

}

template <typename ArrowType, bool contains_null>
cl::Buffer SCVScorer::_copy_training_data(const DataFrame& df, const std::vector<std::string>& variables) const {
    auto training_data = df.to_eigen<false, ArrowType, contains_null>(variables);
    auto& opencl = OpenCLConfig::get();
    return opencl.copy_to_buffer(training_data->data(), training_data->rows() * variables.size());
}

cl::Buffer SCVScorer::_copy_training_data(const DataFrame& df, const std::vector<std::string>& variables) const {
    bool contains_null = df.null_count(variables) > 0;
    switch (m_training_type->id()) {
        case Type::DOUBLE: {
            if (contains_null)
                return _copy_training_data<arrow::DoubleType, true>(df, variables);
            else
                return _copy_training_data<arrow::DoubleType, false>(df, variables);
            break;
        }
        case Type::FLOAT: {
            if (contains_null)
                return _copy_training_data<arrow::FloatType, true>(df, variables);
            else
                return _copy_training_data<arrow::FloatType, true>(df, variables);
            break;
        }
        default:
            throw std::invalid_argument("Wrong data type to score SCV. [double] or [float] data is expected.");
    }
}

template <typename ArrowType>
std::vector<ArrowType> SCVScorer::calculateCovariance(const Eigen::Matrix<ArrowType, Eigen::Dynamic, Eigen::Dynamic>& data) const{
    // Number of observations
    int n = data.rows();

    // Calculate the mean of the data
    Eigen::Matrix<ArrowType, Eigen::Dynamic, 1> mean = data.colwise().mean();

    // Subtract the mean from the data
    Eigen::Matrix<ArrowType, Eigen::Dynamic, Eigen::Dynamic> centered = data.rowwise() - mean.transpose();

    // Calculate the covariance matrix
    Eigen::Matrix<ArrowType, Eigen::Dynamic, Eigen::Dynamic> covarianceMatrix = (centered.adjoint() * centered) / ArrowType(n - 1);

    // Convert the covariance matrix to a std::vector
    std::vector<ArrowType> covariance(covarianceMatrix.data(), covarianceMatrix.data() + covarianceMatrix.size());

    return covariance;
}

template <typename ArrowType>
std::vector<ArrowType> SCVScorer::kroneckerProduct(const std::vector<ArrowType>& a, const std::vector<ArrowType>& b) const {
    std::vector<ArrowType> result(a.size() * b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b.size(); ++j) {
            result[i * b.size() + j] = a[i] * b[j];
        }
    }
    return result;
}

unsigned int SCVScorer::transform_and_permute(unsigned int i, unsigned int r, unsigned int s, unsigned int p, const unsigned int dimension) const
 {
    std::vector<unsigned int> vec(r);

    // Transform i into a vector
    for (unsigned int j = 1; j <= r; ++j) {
        vec[j - 1] = std::floor((i) / (std::pow(dimension, j - 1))) - dimension * std::floor((i) / (std::pow(dimension, j))) + 1;
    }

    // Swap the s-th and k-th elements
    std::swap(vec[s - 1], vec[p - 1]);

    // Calculate the return value
    unsigned int result = 1;
    for (unsigned int j = 0; j < r; ++j) {
        result += (vec[j] - 1) * std::pow(dimension, j );
    }

    return result;
}

template <typename ArrowType>
std::vector<ArrowType> SCVScorer::divideVectorByConstant(const std::vector<ArrowType>& vec, ArrowType constant) const {
    std::vector<ArrowType> result = vec;
    std::transform(result.begin(), result.end(), result.begin(),
                   [constant](ArrowType value) { return value / constant; });
    return result;
}


template <typename ArrowType, bool contains_null>
cl::Buffer SCVScorer::psi_nr_8_impl(const DataFrame& df, const std::vector<std::string>& variables, const unsigned int dimension) const {

    auto training_data = df.to_eigen<false, ArrowType, contains_null>(variables);
    auto covariance = calculateCovariance<typename ArrowType::c_type>(*training_data);
    std::vector<typename ArrowType::c_type> copy = covariance;

    for (int i = 0; i < 3; ++i) {  
        covariance = kroneckerProduct<typename ArrowType::c_type>(covariance, copy);
    }

    unsigned int psi_nr_length = (unsigned int)std::pow(dimension,8);        
                
    for (unsigned int k = 2; k <= 8; k++){
        std::vector<typename ArrowType::c_type> w_new(psi_nr_length, 0.0);
        for (unsigned int j = 1; j <= k; j++){
            for (unsigned int coordinate = 0; coordinate < psi_nr_length; coordinate++){

                unsigned int new_coordinate = transform_and_permute(coordinate, 8u, j, k, dimension);
                w_new[new_coordinate - 1] += covariance[coordinate];

            }
        }        
        covariance = divideVectorByConstant<typename ArrowType::c_type>(w_new, (typename ArrowType::c_type)k);
    }
    auto& opencl = OpenCLConfig::get();
    cl::Buffer psi_nr_8_buffer = opencl.copy_to_buffer(covariance.data(), static_cast<int>(covariance.size()));

    return psi_nr_8_buffer;

}

cl::Buffer SCVScorer::psi_nr_8(const DataFrame& df, const std::vector<std::string>& variables, const unsigned int dimension) const {
    bool contains_null = df.null_count(variables) > 0;
    cl::Buffer psi_8;
    switch (m_training_type->id()) {
        case Type::DOUBLE: {
            if (contains_null){
                psi_8 = psi_nr_8_impl<arrow::DoubleType, true>(df, variables, dimension);
                return psi_8;
            }
                
            else{
                psi_8 = psi_nr_8_impl<arrow::DoubleType, false>(df, variables, dimension);
                return psi_8;
            }
                
            break;
        }
        case Type::FLOAT: {
            if (contains_null){
                psi_8 = psi_nr_8_impl<arrow::FloatType, true>(df, variables, dimension);
                return psi_8;
            }

            else{
                psi_8 = psi_nr_8_impl<arrow::FloatType, false>(df, variables, dimension);
                return psi_8;
                }
                
            break;
        }
        default:
            throw std::invalid_argument("Wrong data type to score SCV. [double] or [float] data is expected.");
    }


}


template <typename ArrowType>
std::pair<cl::Buffer, typename ArrowType::c_type> SCVScorer::copy_diagonal_bandwidth(
    const Matrix<typename ArrowType::c_type, Dynamic, 1>& diagonal_sqrt_bandwidth) const {
    using CType = typename ArrowType::c_type;
    auto& opencl = OpenCLConfig::get();
    auto bw = opencl.copy_to_buffer(diagonal_sqrt_bandwidth.data(), d);

    auto lognorm_const = -diagonal_sqrt_bandwidth.array().log().sum() - 0.5 * d * std::log(2 * util::pi<CType>);
    return std::make_pair(bw, lognorm_const);
}

template <typename ArrowType>
std::tuple<cl::Buffer, cl::Buffer, typename ArrowType::c_type, cl::Buffer> SCVScorer::copy_unconstrained_bandwidth_mse(
    const Matrix<typename ArrowType::c_type, Dynamic, Dynamic>& bandwidth) const {
    
    using CType = typename ArrowType::c_type;
    auto llt_cov = bandwidth.llt();
    auto llt_matrix = llt_cov.matrixLLT();
    Eigen::Matrix<typename ArrowType::c_type, Dynamic, Dynamic> invert_bandwidth = bandwidth.inverse();

    auto& opencl = OpenCLConfig::get();
    auto cholesky = opencl.copy_to_buffer(llt_matrix.data(), d * d);
    auto inv_bandwidth = opencl.copy_to_buffer(invert_bandwidth.data(), d * d);
    auto vec_bandwidth = opencl.copy_to_buffer(bandwidth.data(), d * d);

    auto gaussian_const = -llt_matrix.diagonal().array().log().sum() - 0.5 * d * std::log(2 * util::pi<CType>);

    return std::make_tuple(cholesky, inv_bandwidth, gaussian_const, vec_bandwidth);
}

template <typename ArrowType>
std::tuple<cl::Buffer, cl::Buffer, cl::Buffer, typename ArrowType::c_type, typename ArrowType::c_type, typename ArrowType::c_type, typename ArrowType::c_type> SCVScorer::copy_unconstrained_bandwidth(
    const Matrix<typename ArrowType::c_type, Dynamic, Dynamic>& bandwidth_H, const Matrix<typename ArrowType::c_type, Dynamic, Dynamic>& bandwidth_G) const {
    using CType = typename ArrowType::c_type;
    Eigen::Matrix<typename ArrowType::c_type, Dynamic, Dynamic> G2 = bandwidth_G * 2.0;
    Eigen::Matrix<typename ArrowType::c_type, Dynamic, Dynamic> H2 = bandwidth_H * 2.0;
    auto H2_2G = H2 + G2;
    auto H_2G = bandwidth_H + G2;

    auto llt_cov_2H_2G = H2_2G.llt();
    auto llt_matrix_2H_2G  = llt_cov_2H_2G.matrixLLT();
    auto llt_cov_H_2G = H_2G.llt();
    auto llt_matrix_H_2G  = llt_cov_H_2G.matrixLLT();
    auto llt_cov_G = bandwidth_G.llt();
    auto llt_matrix_G  = llt_cov_G.matrixLLT();

    auto& opencl = OpenCLConfig::get();
    auto cholesky_2H_2G = opencl.copy_to_buffer(llt_matrix_2H_2G.data(), d * d);
    auto cholesky_H_2G = opencl.copy_to_buffer(llt_matrix_H_2G.data(), d * d);
    auto cholesky_G = opencl.copy_to_buffer(llt_matrix_G.data(), d * d);

    auto gaussian_const_2H_2G = -llt_matrix_2H_2G.diagonal().array().log().sum() - 0.5 * d * std::log(2 * util::pi<CType>);
    auto gaussian_const_H_2G = -llt_matrix_H_2G.diagonal().array().log().sum() - 0.5 * d * std::log(2 * util::pi<CType>);
    auto gaussian_const_G = -llt_matrix_G.diagonal().array().log().sum() - 0.5 * d * std::log(2 * util::pi<CType>);

    auto llt_cov = bandwidth_H.llt();
    auto llt_matrix  = llt_cov.matrixLLT();
    auto lognorm_const = -llt_matrix.diagonal().array().log().sum() - 0.5 * d * std::log(4 * util::pi<CType>);

    return std::make_tuple(cholesky_2H_2G, cholesky_H_2G, cholesky_G, gaussian_const_2H_2G, gaussian_const_H_2G, gaussian_const_G, lognorm_const);
}

template <typename ArrowType>
double SCVScorer::score_diagonal_impl(
    const Matrix<typename ArrowType::c_type, Dynamic, 1>& diagonal_sqrt_bandwidth) const {
    using CType = typename ArrowType::c_type;
    auto [bw, lognorm_H] = copy_diagonal_bandwidth<ArrowType>(diagonal_sqrt_bandwidth);
    auto lognorm_2H = lognorm_H - 0.5 * d * std::log(2.);

    auto& opencl = OpenCLConfig::get();

    auto n_distances = N * (N - 1) / 2;

    auto instances_per_iteration = std::min(static_cast<size_t>(1000000), n_distances);
    auto iterations =
        static_cast<int>(std::ceil(static_cast<double>(n_distances) / static_cast<double>(instances_per_iteration)));

    cl::Buffer sum2h = opencl.new_buffer<CType>(instances_per_iteration);
    opencl.fill_buffer<CType>(sum2h, 0., instances_per_iteration);
    cl::Buffer sumh = opencl.new_buffer<CType>(instances_per_iteration);
    opencl.fill_buffer<CType>(sumh, 0., instances_per_iteration);

    cl::Buffer temp_h = opencl.new_buffer<CType>(instances_per_iteration);

    for (auto i = 0; i < (iterations - 1); ++i) {
        ProductSCVScore::sum_triangular_scores<ArrowType>(m_training,
                                                          N,
                                                          d,
                                                          i * instances_per_iteration,
                                                          instances_per_iteration,
                                                          bw,
                                                          lognorm_2H,
                                                          lognorm_H,
                                                          temp_h,
                                                          sum2h,
                                                          sumh);
    }

    auto remaining = n_distances - (iterations - 1) * instances_per_iteration;

    ProductSCVScore::sum_triangular_scores<ArrowType>(m_training,
                                                      N,
                                                      d,
                                                      (iterations - 1) * instances_per_iteration,
                                                      remaining,
                                                      bw,
                                                      lognorm_2H,
                                                      lognorm_H,
                                                      temp_h,
                                                      sum2h,
                                                      sumh);

    auto b2h = opencl.sum1d<ArrowType>(sum2h, instances_per_iteration);
    auto bh = opencl.sum1d<ArrowType>(sumh, instances_per_iteration);

    CType s2h, sh;
    opencl.read_from_buffer(&s2h, b2h, 1);
    opencl.read_from_buffer(&sh, bh, 1);

    // Returns UCV 
    return std::exp(lognorm_2H) + 2 * s2h / N - 4 * sh / (N - 1);
}

template <typename ArrowType, typename SCVScore>
double SCVScorer::score_unconstrained_impl(
    const Matrix<typename ArrowType::c_type, Dynamic, Dynamic>& bandwidth_H, 
    const Matrix<typename ArrowType::c_type, Dynamic, Dynamic>& bandwidth_G) const {
    using CType = typename ArrowType::c_type;
    auto [cholesky_2H_2G, cholesky_H_2G, cholesky_G, gaussian_2H_2G, gaussian_H_2G, gaussian_G, lognorm_H] = copy_unconstrained_bandwidth<ArrowType>(bandwidth_H, bandwidth_G);
    auto gaussian_2G = gaussian_G - 0.5 * d * std::log(2.);
    auto& opencl = OpenCLConfig::get();

    auto n_distances = N * (N - 1) / 2;
    auto instances_per_iteration = std::min(static_cast<size_t>(1000000), n_distances);
    auto iterations =
        static_cast<int>(std::ceil(static_cast<double>(n_distances) / static_cast<double>(instances_per_iteration)));

    cl::Buffer tmp_mat_buffer_2H_2G;
    if constexpr (std::is_same_v<SCVScore, MultivariateSCVScore>) {
        tmp_mat_buffer_2H_2G = opencl.new_buffer<CType>(instances_per_iteration * d);
    }

    cl::Buffer tmp_mat_buffer_H_2G;
    if constexpr (std::is_same_v<SCVScore, MultivariateSCVScore>) {
        tmp_mat_buffer_H_2G = opencl.new_buffer<CType>(instances_per_iteration * d);
    }

    cl::Buffer tmp_mat_buffer_G;
    if constexpr (std::is_same_v<SCVScore, MultivariateSCVScore>) {
        tmp_mat_buffer_G = opencl.new_buffer<CType>(instances_per_iteration * d);
    }


    cl::Buffer scv = opencl.new_buffer<CType>(instances_per_iteration);
    opencl.fill_buffer<CType>(scv, 0., instances_per_iteration);


    for (auto i = 0; i < (iterations - 1); ++i) {
        SCVScore::template scv<ArrowType>(m_training,
                                                            N,
                                                            d,
                                                            i * instances_per_iteration,
                                                            instances_per_iteration,
                                                            gaussian_2H_2G,
                                                            gaussian_H_2G,
                                                            gaussian_2G,
                                                            cholesky_2H_2G,
                                                            cholesky_H_2G,
                                                            cholesky_G,
                                                            tmp_mat_buffer_2H_2G,
                                                            tmp_mat_buffer_H_2G,
                                                            tmp_mat_buffer_G,
                                                            scv);
    }

    auto remaining = n_distances - (iterations - 1) * instances_per_iteration;
    SCVScore::template scv<ArrowType>(m_training,
                                                            N,
                                                            d,
                                                            (iterations - 1) * instances_per_iteration,
                                                            remaining,
                                                            gaussian_2H_2G,
                                                            gaussian_H_2G,
                                                            gaussian_2G,
                                                            cholesky_2H_2G,
                                                            cholesky_H_2G,
                                                            cholesky_G,
                                                            tmp_mat_buffer_2H_2G,
                                                            tmp_mat_buffer_H_2G,
                                                            tmp_mat_buffer_G,
                                                            scv);

                                               
    CType scv_h;
    auto scv_sum = opencl.sum1d<ArrowType>(scv, instances_per_iteration);
    opencl.read_from_buffer(&scv_h, scv_sum, 1);

    // Returns SCV scaled by N
    return std::exp(lognorm_H) + 0.25 * scv_h / N ;
}


template <typename ArrowType, typename SCVScore>
double SCVScorer::score_mse_unconstrained_impl(
    const Matrix<typename ArrowType::c_type, Dynamic, Dynamic>& bandwidth, cl::Buffer psi_previous, unsigned int derivate_order) const {
    using CType = typename ArrowType::c_type;

    auto [cholesky, inv_bandwidth, guassian_H, vec_H] = copy_unconstrained_bandwidth_mse<ArrowType>(bandwidth);
    
    auto& opencl = OpenCLConfig::get();

    auto derivate_size = std::pow(d, derivate_order);

    cl::Buffer tmp_gaussian_density_values;
    if constexpr (std::is_same_v<SCVScore, MultivariateSCVScore>) {
        tmp_gaussian_density_values = opencl.new_buffer<CType>(1);
    }

    cl::Buffer zero_vector = opencl.new_buffer<CType>(d);
    opencl.fill_buffer<CType>(zero_vector, 0., d);

    cl::Buffer tmp_derivate_buffer;
    if constexpr (std::is_same_v<SCVScore, MultivariateSCVScore>) {
        tmp_derivate_buffer = opencl.new_buffer<CType>(derivate_size);
    }
    

    cl::Buffer tmp_result_kron_Idr_psi;
    if constexpr (std::is_same_v<SCVScore, MultivariateSCVScore>) {
        tmp_result_kron_Idr_psi = opencl.new_buffer<CType>(derivate_size);
        opencl.fill_buffer<CType>(tmp_result_kron_Idr_psi, 0., derivate_size);
    }

    cl::Buffer tmp_w_new;
    if constexpr (std::is_same_v<SCVScore, MultivariateSCVScore>) {
        tmp_w_new = opencl.new_buffer<CType>(derivate_size);
    }

    cl::Buffer tmp_u_k_1;
    if constexpr (std::is_same_v<SCVScore, MultivariateSCVScore>) {
        tmp_u_k_1 = opencl.new_buffer<CType>(derivate_size);
    }

    cl::Buffer tmp_u_k_2;
    if constexpr (std::is_same_v<SCVScore, MultivariateSCVScore>) {
        tmp_u_k_2 = opencl.new_buffer<CType>(derivate_size);
    }
    
    
    cl::Buffer mse = opencl.new_buffer<CType>(derivate_size);
    opencl.fill_buffer<CType>(mse, 0., derivate_size);

    
    SCVScore::template AB_criterion<ArrowType>(zero_vector,
                                                        N,
                                                        d,
                                                        derivate_order,
                                                        guassian_H,
                                                        cholesky,
                                                        vec_H,
                                                        tmp_gaussian_density_values,
                                                        inv_bandwidth,
                                                        tmp_w_new,
                                                        tmp_u_k_1,
                                                        tmp_u_k_2,
                                                        psi_previous,
                                                        tmp_derivate_buffer,
                                                        tmp_result_kron_Idr_psi,
                                                        mse);
    CType mse_h;

    auto mse_sum = opencl.sum1d<ArrowType>(mse, derivate_size);

    opencl.read_from_buffer(&mse_h, mse_sum, 1);


    // Convert CType to double if ArrowType is DoubleType
    if constexpr (std::is_same_v<ArrowType, arrow::DoubleType>) {
        return static_cast<double>(mse_h);
    }
    // Convert CType to double if ArrowType is FloatType
    else if constexpr (std::is_same_v<ArrowType, arrow::FloatType>) {
        return static_cast<double>(static_cast<float>(mse_h));
    }
    else {
        // Handle unsupported ArrowType
        throw std::runtime_error("Unsupported ArrowType");
    }

    return mse_h;
}



double SCVScorer::score_diagonal(const VectorXd& diagonal_bandwidth) const {
    if (d != static_cast<size_t>(diagonal_bandwidth.rows()))
        throw std::invalid_argument("Wrong dimension for bandwidth vector. it should be a " + std::to_string(d) +
                                    " vector.");

    switch (m_training_type->id()) {
        case Type::DOUBLE: {
            return score_diagonal_impl<arrow::DoubleType>(diagonal_bandwidth.cwiseSqrt());
        }
        case Type::FLOAT: {
            return score_diagonal_impl<arrow::FloatType>(diagonal_bandwidth.template cast<float>().cwiseSqrt());
        }
        default:
            throw std::runtime_error("Unreachable code");
    }
}

double SCVScorer::score_unconstrained(const MatrixXd& bandwidth, const MatrixXd& bandwidth_G) const {
    if (d != static_cast<size_t>(bandwidth.rows()) && d != static_cast<size_t>(bandwidth.cols()))
        throw std::invalid_argument("Wrong dimension for bandwidth matrix. it should be a " + std::to_string(d) + "x" +
                                    std::to_string(d) + " matrix.");

    switch (m_training_type->id()) {
        case Type::DOUBLE: {
            if (d == 1)
                return score_unconstrained_impl<arrow::DoubleType, UnivariateSCVScore>(bandwidth, bandwidth_G);
            else
                return score_unconstrained_impl<arrow::DoubleType, MultivariateSCVScore>(bandwidth, bandwidth_G);
        }
        case Type::FLOAT: {
            if (d == 1)
                return score_unconstrained_impl<arrow::FloatType, UnivariateSCVScore>(bandwidth.template cast<float>(), bandwidth_G.template cast<float>());
            else
                return score_unconstrained_impl<arrow::FloatType, MultivariateSCVScore>(
                    bandwidth.template cast<float>(), bandwidth_G.template cast<float>());
        }
        default:
            throw std::runtime_error("Unreachable code");
    }
}


double SCVScorer::score_unconstrained_mse(const MatrixXd& bandwidth,const cl::Buffer psi_previous, const unsigned int derivate_order) const {
    if (d != static_cast<size_t>(bandwidth.rows()) && d != static_cast<size_t>(bandwidth.cols()))
        throw std::invalid_argument("Wrong dimension for bandwidth matrix. it should be a " + std::to_string(d) + "x" +
                                    std::to_string(d) + " matrix.");

    switch (m_training_type->id()) {
        case Type::DOUBLE: {
            if (d == 1)
                return score_mse_unconstrained_impl<arrow::DoubleType, UnivariateSCVScore>(bandwidth, psi_previous, derivate_order);
            else
                return score_mse_unconstrained_impl<arrow::DoubleType, MultivariateSCVScore>(bandwidth, psi_previous, derivate_order);
        }
        case Type::FLOAT: {
            if (d == 1)
                return score_mse_unconstrained_impl<arrow::FloatType, UnivariateSCVScore>(
                    bandwidth.template cast<float>(), psi_previous, derivate_order);
            else
                return score_mse_unconstrained_impl<arrow::FloatType, MultivariateSCVScore>(
                    bandwidth.template cast<float>(), psi_previous, derivate_order);
        }
        default:
            throw std::runtime_error("Unreachable code");
    }
}


template <typename ArrowType>
std::tuple<cl::Buffer, cl::Buffer, typename ArrowType::c_type> SCVScorer::psi_r_utils(
    const Matrix<typename ArrowType::c_type, Dynamic, Dynamic>& bandwidth) const {
    using CType = typename ArrowType::c_type;
    Eigen::Matrix<typename ArrowType::c_type, Dynamic, Dynamic> bandwidth2 = 2 * bandwidth;
    auto llt_cov2 = bandwidth2.llt();
    auto llt_matrix2 = llt_cov2.matrixLLT();
    Eigen::Matrix<typename ArrowType::c_type, Dynamic, Dynamic> invert_bandwidth2 = bandwidth2.inverse();

    auto& opencl = OpenCLConfig::get();
    auto cholesky = opencl.copy_to_buffer(llt_matrix2.data(), d * d);
    auto inv_bandwidth = opencl.copy_to_buffer(invert_bandwidth2.data(), d * d);

    auto gaussian_const = -llt_matrix2.diagonal().array().log().sum() - 0.5 * d * std::log(2 * util::pi<CType>);

    return std::make_tuple(cholesky, inv_bandwidth, gaussian_const);
}

template <typename ArrowType, typename SCVScore>
cl::Buffer SCVScorer::psi_r_impl(const Matrix<typename ArrowType::c_type, Dynamic, Dynamic>& bandwidth, const unsigned int derivate_order) const {
    using CType = typename ArrowType::c_type;
    auto [cholesky, inv_bandwidth, guassian_2H] = psi_r_utils<ArrowType>(bandwidth);

    auto& opencl = OpenCLConfig::get();
    auto derivate_size = std::pow(d, derivate_order);
    auto n_distances = N * (N - 1) / 2;
    size_t instances_per_iteration;
    if (derivate_order > 4){
        instances_per_iteration = std::min(static_cast<size_t>(100), n_distances);
    }
    else {
        instances_per_iteration = std::min(static_cast<size_t>(1000), n_distances);
    }
    
    auto iterations =
        static_cast<int>(std::ceil(static_cast<double>(n_distances) / static_cast<double>(instances_per_iteration)));

    cl::Buffer tmp_psi_4 = opencl.new_buffer<CType>(derivate_size);
    opencl.fill_buffer<CType>(tmp_psi_4, 0., derivate_size);

    cl::Buffer tmp_gaussian_density_values;
    if constexpr (std::is_same_v<SCVScore, MultivariateSCVScore>) {
        tmp_gaussian_density_values = opencl.new_buffer<CType>(instances_per_iteration);
    }

    cl::Buffer tmp_mat_buffer;
    if constexpr (std::is_same_v<SCVScore, MultivariateSCVScore>) {
        tmp_mat_buffer = opencl.new_buffer<CType>(instances_per_iteration * d);
    }
    

    cl::Buffer tmp_derivate_buffer;
    tmp_derivate_buffer = opencl.new_buffer<CType>(instances_per_iteration * derivate_size);

    cl::Buffer tmp_w_new;
    if constexpr (std::is_same_v<SCVScore, MultivariateSCVScore>) {
        tmp_w_new = opencl.new_buffer<CType>(instances_per_iteration * derivate_size);
    }

    cl::Buffer tmp_u_k_1;
    if constexpr (std::is_same_v<SCVScore, MultivariateSCVScore>) {
        tmp_u_k_1 = opencl.new_buffer<CType>(instances_per_iteration * derivate_size);
    }

    cl::Buffer tmp_u_k_2;
    if constexpr (std::is_same_v<SCVScore, MultivariateSCVScore>) {
        tmp_u_k_2 = opencl.new_buffer<CType>(instances_per_iteration * derivate_size);
    }


    for (auto i = 0; i < (iterations - 1); ++i) {
        SCVScore::template Psi_r<ArrowType>(m_training,
                                                            N,
                                                            d,
                                                            i * instances_per_iteration,
                                                            instances_per_iteration,
                                                            derivate_order,
                                                            guassian_2H,
                                                            cholesky,
                                                            tmp_mat_buffer,
                                                            tmp_gaussian_density_values,
                                                            inv_bandwidth,
                                                            tmp_w_new,
                                                            tmp_u_k_1,
                                                            tmp_u_k_2,
                                                            tmp_derivate_buffer,
                                                            tmp_psi_4);
    }

    auto remaining = n_distances - (iterations - 1) * instances_per_iteration;

    SCVScore::template Psi_r<ArrowType>(m_training,
                                                            N,
                                                            d,
                                                            (iterations - 1) * instances_per_iteration,
                                                            remaining,
                                                            derivate_order,
                                                            guassian_2H,
                                                            cholesky,
                                                            tmp_mat_buffer,
                                                            tmp_gaussian_density_values,
                                                            inv_bandwidth,
                                                            tmp_w_new,
                                                            tmp_u_k_1,
                                                            tmp_u_k_2,
                                                            tmp_derivate_buffer,
                                                            tmp_psi_4);
   return tmp_psi_4;
}

cl::Buffer SCVScorer::psi_r(const MatrixXd& bandwidth, const unsigned int derivate_order) const {
    switch (m_training_type->id()) {
        case Type::DOUBLE: {
            if (d == 1)
                return psi_r_impl<arrow::DoubleType, UnivariateSCVScore>(bandwidth,  derivate_order);
            else
                return psi_r_impl<arrow::DoubleType, MultivariateSCVScore>(bandwidth, derivate_order);
        }
        case Type::FLOAT: {
            if (d == 1)
                return psi_r_impl<arrow::FloatType, UnivariateSCVScore>(
                    bandwidth.template cast<float>(), derivate_order);
            else
                return psi_r_impl<arrow::FloatType, MultivariateSCVScore>(
                    bandwidth.template cast<float>(), derivate_order);
        }
        default:
            throw std::runtime_error("Unreachable code");
    }


}


struct SCVOptimInfo {
    SCVScorer scv_scorer;
    double start_score;
    double start_determinant;
    cl::Buffer psi_previous;
    unsigned int derivate_order;
    MatrixXd G;
};

double wrap_scv_diag_optim(unsigned n, const double* x, double*, void* my_func_data) {
    using MapType = Eigen::Map<const VectorXd>;
    MapType xm(x, n);

    SCVOptimInfo& optim_info = *reinterpret_cast<SCVOptimInfo*>(my_func_data);

    auto det_sqrt = xm.prod();
    auto det = det_sqrt * det_sqrt;

    if (det <= util::machine_tol || det < 1e-3 * optim_info.start_determinant ||
        det > 1e3 * optim_info.start_determinant)
        return optim_info.start_score + 10e-8;

    auto score = optim_info.scv_scorer.score_diagonal(xm.array().square().matrix());

    if (std::abs(score) > 1e3 * std::abs(optim_info.start_score)) return optim_info.start_score + 10e-8;

    return score;
}

double wrap_scv_optim(unsigned n, const double* x, double*, void* my_func_data) {
    using MapType = Eigen::Map<const VectorXd>;
    MapType xm(x, n);

    auto sqrt = util::invvech_triangular(xm);
    auto H = sqrt * sqrt.transpose();

    SCVOptimInfo& optim_info = *reinterpret_cast<SCVOptimInfo*>(my_func_data);

    auto det = std::exp(2 * sqrt.diagonal().array().log().sum());

    // Avoid too small/large determinants returning the start score.
    // Package ks uses 1e10 as constant.
    if (det <= util::machine_tol || det < 1e-3 * optim_info.start_determinant ||
        det > 1e3 * optim_info.start_determinant || std::isnan(det))
        return optim_info.start_score + 10e-8;

    auto score = optim_info.scv_scorer.score_unconstrained(H, optim_info.G);

    // Avoid scores with too much difference.
    if (std::abs(score) > 1e3 * std::abs(optim_info.start_score)) return optim_info.start_score + 10e-8;

    return score;
}


double wrap_scv_mse_optim(unsigned n, const double* x, double*, void* my_func_data) {
    using MapType = Eigen::Map<const VectorXd>;
    MapType xm(x, n);

    auto sqrt = util::invvech_triangular(xm);
    auto H = sqrt * sqrt.transpose();

    SCVOptimInfo& optim_info = *reinterpret_cast<SCVOptimInfo*>(my_func_data);

    auto det = std::exp(2 * sqrt.diagonal().array().log().sum());

    // Avoid too small/large determinants returning the start score.
    // Package ks uses 1e10 as constant.
    if (det <= util::machine_tol || det < 1e-3 * optim_info.start_determinant ||
        det > 1e3 * optim_info.start_determinant || std::isnan(det))
        return optim_info.start_score + 10e-8;

    auto score = optim_info.scv_scorer.score_unconstrained_mse(H, optim_info.psi_previous, optim_info.derivate_order);

    // Avoid scores with too much difference.
    if (std::abs(score) > 1e3 * std::abs(optim_info.start_score)) return optim_info.start_score + 10e-8;

    return score;
}

VectorXd SCV::diag_bandwidth(const DataFrame& df, const std::vector<std::string>& variables) const {
    if (variables.empty()) return VectorXd(0);

    NormalReferenceRule nr;

    auto normal_bandwidth = nr.diag_bandwidth(df, variables);

    SCVScorer scv_scorer(df, variables);
    auto start_score = 1.0;
    //scv_scorer.score_unconstrained(normal_bandwidth); falta adaptar todo para product kde
    auto start_determinant = normal_bandwidth.prod();

    SCVOptimInfo optim_info{/*.scv_scorer = */ scv_scorer,
                            /*.start_score = */ start_score,
                            /*.start_determinant = */ start_determinant};

    auto start_bandwidth = normal_bandwidth.cwiseSqrt().eval();

    nlopt::opt opt(nlopt::LN_NELDERMEAD, start_bandwidth.rows());
    opt.set_min_objective(wrap_scv_diag_optim, &optim_info);
    opt.set_ftol_rel(1e-3);
    opt.set_xtol_rel(1e-3);
    std::vector<double> x(start_bandwidth.rows());
    std::copy(start_bandwidth.data(), start_bandwidth.data() + start_bandwidth.rows(), x.data());
    double minf;

    try {
        opt.optimize(x, minf);
    } catch (std::exception& e) {
        throw std::invalid_argument(std::string("Failed optimizing bandwidth: ") + e.what());
    }

    std::copy(x.data(), x.data() + x.size(), start_bandwidth.data());

    return start_bandwidth.array().square().matrix();
}

MatrixXd SCV::bandwidth(const DataFrame& df, const std::vector<std::string>& variables) const {
    if (variables.empty()) return MatrixXd(0, 0);

    NormalReferenceRule nr;

    auto normal_bandwidth = nr.bandwidth(df, variables);

    SCVScorer scv_scorer(df, variables);
    auto psi_previous = scv_scorer.psi_nr_8(df, variables, static_cast<unsigned int>(variables.size()));
    auto start_score = scv_scorer.score_unconstrained_mse(normal_bandwidth, psi_previous, 6u);
    auto start_determinant = normal_bandwidth.determinant();
    auto initial_G = normal_bandwidth;
    SCVOptimInfo optim_info{/*.scv_scorer = */ scv_scorer,
                            /*.start_score = */ start_score,
                            /*.start_determinant = */ start_determinant,
                            /*.psi_previous = */ psi_previous,
                            /*.derivate_order = */ 6,
                            /*.G = */ initial_G};
    LLT<Eigen::Ref<MatrixXd>> start_sqrt(normal_bandwidth);
    auto start_vech = util::vech(start_sqrt.matrixL());

    nlopt::opt opt_G6(nlopt::LN_NELDERMEAD, start_vech.rows());
    opt_G6.set_min_objective(wrap_scv_mse_optim, &optim_info);
    opt_G6.set_ftol_rel(1e-3);
    opt_G6.set_xtol_rel(1e-3);
    std::vector<double> x(start_vech.rows());
    std::copy(start_vech.data(), start_vech.data() + start_vech.rows(), x.data());
    double minf_G6;

    try {
        opt_G6.optimize(x, minf_G6);
    } catch (std::exception& e) {
        throw std::invalid_argument(std::string("Failed optimizing bandwidth: ") + e.what());
    }

    std::copy(x.data(), x.data() + x.size(), start_vech.data());

    auto sqrt = util::invvech_triangular(start_vech);
    auto G6 = sqrt * sqrt.transpose();
    optim_info.psi_previous = optim_info.scv_scorer.psi_r(G6, 6);
    optim_info.derivate_order = 4;
    optim_info.start_score = scv_scorer.score_unconstrained_mse(normal_bandwidth, optim_info.psi_previous, 4);

    
    optim_info.start_determinant = normal_bandwidth.determinant();


    start_vech = util::vech(start_sqrt.matrixL());

    nlopt::opt opt_G4(nlopt::LN_NELDERMEAD, start_vech.rows());
    opt_G4.set_min_objective(wrap_scv_mse_optim, &optim_info);
    opt_G4.set_ftol_rel(1e-3);
    opt_G4.set_xtol_rel(1e-3);
    x = std::vector<double>(start_vech.rows());
    std::copy(start_vech.data(), start_vech.data() + start_vech.rows(), x.data());
    double minf_G4;

    try {
        opt_G4.optimize(x, minf_G4);
    } catch (std::exception& e) {
        throw std::invalid_argument(std::string("Failed optimizing bandwidth: ") + e.what());
    }

    std::copy(x.data(), x.data() + x.size(), start_vech.data());

    sqrt = util::invvech_triangular(start_vech);
    auto G4 = sqrt * sqrt.transpose();
    optim_info.G = G4;
    optim_info.psi_previous = optim_info.scv_scorer.psi_r(G4, 4);
    optim_info.derivate_order = 4;
    optim_info.start_score = scv_scorer.score_unconstrained(normal_bandwidth, optim_info.G);

    optim_info.start_determinant = normal_bandwidth.determinant();


    start_vech = util::vech(start_sqrt.matrixL());

    nlopt::opt opt(nlopt::LN_NELDERMEAD, start_vech.rows());
    opt.set_min_objective(wrap_scv_optim, &optim_info);
    opt.set_ftol_rel(1e-3);
    opt.set_xtol_rel(1e-3);
    x = std::vector<double>(start_vech.rows());
    std::copy(start_vech.data(), start_vech.data() + start_vech.rows(), x.data());
    double minf;

    try {
        opt.optimize(x, minf);
    } catch (std::exception& e) {
        throw std::invalid_argument(std::string("Failed optimizing bandwidth: ") + e.what());
    }

    std::copy(x.data(), x.data() + x.size(), start_vech.data());

    sqrt = util::invvech_triangular(start_vech);
    auto H = sqrt * sqrt.transpose();

    return H;
}

}  // namespace kde