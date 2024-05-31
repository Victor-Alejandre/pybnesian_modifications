#include <kde/BCV.hpp>
#include <kde/NormalReferenceRule.hpp>
#include <util/math_constants.hpp>
#include <util/vech_ops.hpp>
#include <nlopt.hpp>
#include <vector>
#include <cmath>
#include <unordered_set>
#include <iostream>


using Eigen::LLT;
using opencl::OpenCLConfig, opencl::OpenCL_kernel_traits;

namespace kde {

class UnivariateBCVScore {
public:
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
                                                 cl::Buffer&,
                                                 const cl::Buffer& vec_inv_bandwidth,
                                                 cl::Buffer&,
                                                 cl::Buffer&,
                                                 cl::Buffer&,
                                                 cl::Buffer& diff_derivates,
                                                 cl::Buffer& psi_r) ;
    template <typename ArrowType>
    static void bcv(const cl::Buffer& vectorization_H,
                                                 cl::Buffer& psi_r,
                                                 const unsigned int training_cols,
                                                 cl::Buffer& result) ;
};

template <typename ArrowType>
void UnivariateBCVScore::Psi_r(const cl::Buffer& training_data,
                                                 const unsigned int training_rows,
                                                 const unsigned int training_cols,
                                                 const unsigned int index_offset,
                                                 const unsigned int length,
                                                 const unsigned int derivate_order,
                                                 typename ArrowType::c_type gaussian_const,
                                                 const cl::Buffer& cholesky,
                                                 cl::Buffer&,
                                                 cl::Buffer&,
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
void UnivariateBCVScore::bcv(const cl::Buffer& vectorization_H,
                                                 cl::Buffer& psi_r,
                                                 const unsigned int training_cols,
                                                 cl::Buffer& result)  {                                             
    auto& opencl = OpenCLConfig::get();
    auto& k_compute_dot_product = opencl.kernel(OpenCL_kernel_traits<ArrowType>::compute_dot_product);
    k_compute_dot_product.setArg(0, vectorization_H);
    k_compute_dot_product.setArg(1, psi_r);
    k_compute_dot_product.setArg(2, training_cols);
    k_compute_dot_product.setArg(3, result);
    auto& queue = opencl.queue();
    RAISE_ENQUEUEKERNEL_ERROR(queue.enqueueNDRangeKernel(
        k_compute_dot_product, cl::NullRange, cl::NDRange(std::pow(training_cols, 4)), cl::NullRange));

}

class ProductBCVScore {
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
void ProductBCVScore::sum_triangular_scores(const cl::Buffer& training_data,
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

class MultivariateBCVScore {
public:
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
                                                 cl::Buffer& gaussian_density_values,
                                                 const cl::Buffer& vec_inv_bandwidth,
                                                 cl::Buffer& w_new,
                                                 cl::Buffer& u_k_1,
                                                 cl::Buffer& u_k_2,
                                                 cl::Buffer& diff_derivates,
                                                 cl::Buffer& psi_r);
    template <typename ArrowType>
    static void bcv(const cl::Buffer& vectorization_H,
                                                 cl::Buffer& psi_r,
                                                 const unsigned int training_cols,
                                                 cl::Buffer& result)  ;
};

template <typename ArrowType>
void MultivariateBCVScore::Psi_r(const cl::Buffer& training_data,
                                                 const unsigned int training_rows,
                                                 const unsigned int training_cols,
                                                 const unsigned int index_offset,
                                                 const unsigned int length,
                                                 const unsigned int derivate_order,
                                                 typename ArrowType::c_type gaussian_const,
                                                 const cl::Buffer& cholesky,
                                                 cl::Buffer& tmp_diff_mat,
                                                 cl::Buffer& gaussian_density_values,
                                                 const cl::Buffer& vec_inv_bandwidth,
                                                 cl::Buffer& w_new,
                                                 cl::Buffer& u_k_1,
                                                 cl::Buffer& u_k_2,
                                                 cl::Buffer& diff_derivates,
                                                 cl::Buffer& psi_r) {
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
void MultivariateBCVScore::bcv(const cl::Buffer& vectorization_H,
                                                 cl::Buffer& psi_r,
                                                 const unsigned int training_cols,
                                                 cl::Buffer& result) {                                             
    auto& opencl = OpenCLConfig::get();
    auto& k_compute_dot_product = opencl.kernel(OpenCL_kernel_traits<ArrowType>::compute_dot_product);
    k_compute_dot_product.setArg(0, vectorization_H);
    k_compute_dot_product.setArg(1, psi_r);
    k_compute_dot_product.setArg(2, training_cols);
    k_compute_dot_product.setArg(3, result);
    auto& queue = opencl.queue();
    RAISE_ENQUEUEKERNEL_ERROR(queue.enqueueNDRangeKernel(
        k_compute_dot_product, cl::NullRange, cl::NDRange(std::pow(training_cols, 4)), cl::NullRange));
}

template <typename ArrowType, bool contains_null>
cl::Buffer BCVScorer::_copy_training_data(const DataFrame& df, const std::vector<std::string>& variables) const {
    auto training_data = df.to_eigen<false, ArrowType, contains_null>(variables);
    auto& opencl = OpenCLConfig::get();
    return opencl.copy_to_buffer(training_data->data(), training_data->rows() * variables.size());
}

cl::Buffer BCVScorer::_copy_training_data(const DataFrame& df, const std::vector<std::string>& variables) const {
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
            throw std::invalid_argument("Wrong data type to score BCV. [double] or [float] data is expected.");
    }
}




template <typename ArrowType>
std::pair<cl::Buffer, typename ArrowType::c_type> BCVScorer::copy_diagonal_bandwidth(
    const Matrix<typename ArrowType::c_type, Dynamic, 1>& diagonal_sqrt_bandwidth) const {
    using CType = typename ArrowType::c_type;
    auto& opencl = OpenCLConfig::get();
    auto bw = opencl.copy_to_buffer(diagonal_sqrt_bandwidth.data(), d);

    auto lognorm_const = -diagonal_sqrt_bandwidth.array().log().sum() - 0.5 * d * std::log(2 * util::pi<CType>);
    return std::make_pair(bw, lognorm_const);
}

template <typename ArrowType>
std::tuple<cl::Buffer, cl::Buffer, typename ArrowType::c_type, cl::Buffer, typename ArrowType::c_type> BCVScorer::copy_unconstrained_bandwidth(
    const Matrix<typename ArrowType::c_type, Dynamic, Dynamic>& bandwidth) const {
    using CType = typename ArrowType::c_type;
    Eigen::Matrix<typename ArrowType::c_type, Dynamic, Dynamic> bandwidth2 = 2 * bandwidth;
    auto llt_cov2 = bandwidth2.llt();
    auto llt_matrix2 = llt_cov2.matrixLLT();
    Eigen::Matrix<typename ArrowType::c_type, Dynamic, Dynamic> invert_bandwidth2 = bandwidth2.inverse();

    auto& opencl = OpenCLConfig::get();
    auto cholesky = opencl.copy_to_buffer(llt_matrix2.data(), d * d);
    auto inv_bandwidth = opencl.copy_to_buffer(invert_bandwidth2.data(), d * d);
    auto vec_bandwidth = opencl.copy_to_buffer(bandwidth.data(), d * d);

    auto gaussian_const = -llt_matrix2.diagonal().array().log().sum() - 0.5 * d * std::log(2 * util::pi<CType>);

    auto llt_cov = bandwidth.llt();
    auto llt_matrix = llt_cov.matrixLLT();
    auto lognorm_const = -llt_matrix.diagonal().array().log().sum() - 0.5 * d * std::log(4 * util::pi<CType>);

    return std::make_tuple(cholesky, inv_bandwidth, gaussian_const, vec_bandwidth, lognorm_const);
}

template <typename ArrowType>
double BCVScorer::score_diagonal_impl(
    const Matrix<typename ArrowType::c_type, Dynamic, 1>& diagonal_sqrt_bandwidth) const {
    using CType = typename ArrowType::c_type;
    auto [bw, lognorm_H] = copy_diagonal_bandwidth<ArrowType>(diagonal_sqrt_bandwidth);
    auto lognorm_2H = lognorm_H - 0.5 * d * std::log(2.);

    auto& opencl = OpenCLConfig::get();

    auto n_distances = N * (N - 1) / 2;

    auto instances_per_iteration = std::min(static_cast<size_t>(1000000), n_distances); //cambiar a 10^2
    auto iterations =
        static_cast<int>(std::ceil(static_cast<double>(n_distances) / static_cast<double>(instances_per_iteration)));

    cl::Buffer sum2h = opencl.new_buffer<CType>(instances_per_iteration);
    opencl.fill_buffer<CType>(sum2h, 0., instances_per_iteration);
    cl::Buffer sumh = opencl.new_buffer<CType>(instances_per_iteration);
    opencl.fill_buffer<CType>(sumh, 0., instances_per_iteration);

    cl::Buffer temp_h = opencl.new_buffer<CType>(instances_per_iteration);

    for (auto i = 0; i < (iterations - 1); ++i) {
        ProductBCVScore::sum_triangular_scores<ArrowType>(m_training,
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

    ProductBCVScore::sum_triangular_scores<ArrowType>(m_training,
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

    // Returns BCV 
    return std::exp(lognorm_2H) + 2 * s2h / N - 4 * sh / (N - 1);
}

template <typename ArrowType, typename BCVScore>
double BCVScorer::score_unconstrained_impl(
    const Matrix<typename ArrowType::c_type, Dynamic, Dynamic>& bandwidth) const {
    using CType = typename ArrowType::c_type;
    auto [cholesky, inv_bandwidth, guassian_2H, vec_H, lognorm_H] = copy_unconstrained_bandwidth<ArrowType>(bandwidth);

    auto& opencl = OpenCLConfig::get();

    auto n_distances = N * (N - 1) / 2;

    auto instances_per_iteration = std::min(static_cast<size_t>(10000), n_distances); 
    auto iterations =
        static_cast<int>(std::ceil(static_cast<double>(n_distances) / static_cast<double>(instances_per_iteration)));

    auto derivate_size = std::pow(d, 4);
    cl::Buffer tmp_psi_4 = opencl.new_buffer<CType>(derivate_size);
    opencl.fill_buffer<CType>(tmp_psi_4, 0., derivate_size);

    cl::Buffer tmp_gaussian_density_values;
    if constexpr (std::is_same_v<BCVScore, MultivariateBCVScore>) {
        tmp_gaussian_density_values = opencl.new_buffer<CType>(instances_per_iteration);
    }

    cl::Buffer tmp_mat_buffer;
    if constexpr (std::is_same_v<BCVScore, MultivariateBCVScore>) {
        tmp_mat_buffer = opencl.new_buffer<CType>(instances_per_iteration * d);
    }
    

    cl::Buffer tmp_derivate_buffer;
    tmp_derivate_buffer = opencl.new_buffer<CType>(instances_per_iteration * derivate_size);


    cl::Buffer tmp_w_new;
    if constexpr (std::is_same_v<BCVScore, MultivariateBCVScore>) {
        tmp_w_new = opencl.new_buffer<CType>(instances_per_iteration * derivate_size);
    }

    cl::Buffer tmp_u_k_1;
    if constexpr (std::is_same_v<BCVScore, MultivariateBCVScore>) {
        tmp_u_k_1 = opencl.new_buffer<CType>(instances_per_iteration * derivate_size);
    }

    cl::Buffer tmp_u_k_2;
    if constexpr (std::is_same_v<BCVScore, MultivariateBCVScore>) {
        tmp_u_k_2 = opencl.new_buffer<CType>(instances_per_iteration * derivate_size);
    }


    for (auto i = 0; i < (iterations - 1); ++i) {
        BCVScore::template Psi_r<ArrowType>(m_training,
                                                            N,
                                                            d,
                                                            i * instances_per_iteration,
                                                            instances_per_iteration,
                                                            4,
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

    BCVScore::template Psi_r<ArrowType>(m_training,
                                                            N,
                                                            d,
                                                            (iterations - 1) * instances_per_iteration,
                                                            remaining,
                                                            4,
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
                                                   
    cl::Buffer bcv;
    bcv = opencl.new_buffer<CType>(derivate_size);
    BCVScore::template bcv<ArrowType>(vec_H,
                                                  tmp_psi_4,
                                                  d,
                                                  bcv); 

                                               
    auto bcv_sum = opencl.sum1d<ArrowType>(bcv, derivate_size);
    CType bcv_h;
    opencl.read_from_buffer(&bcv_h, bcv_sum, 1);

    // Returns BCV scaled by N: N * BCV
    return std::exp(lognorm_H) + 0.25 * bcv_h / N ;
}

double BCVScorer::score_diagonal(const VectorXd& diagonal_bandwidth) const {
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

double BCVScorer::score_unconstrained(const MatrixXd& bandwidth) const {
    if (d != static_cast<size_t>(bandwidth.rows()) && d != static_cast<size_t>(bandwidth.cols()))
        throw std::invalid_argument("Wrong dimension for bandwidth matrix. it should be a " + std::to_string(d) + "x" +
                                    std::to_string(d) + " matrix.");

    switch (m_training_type->id()) {
        case Type::DOUBLE: {
            if (d == 1)
                return score_unconstrained_impl<arrow::DoubleType, UnivariateBCVScore>(bandwidth);
            else
                return score_unconstrained_impl<arrow::DoubleType, MultivariateBCVScore>(bandwidth);
        }
        case Type::FLOAT: {
            if (d == 1)
                return score_unconstrained_impl<arrow::FloatType, UnivariateBCVScore>(bandwidth.template cast<float>());
            else
                return score_unconstrained_impl<arrow::FloatType, MultivariateBCVScore>(
                    bandwidth.template cast<float>());
        }
        default:
            throw std::runtime_error("Unreachable code");
    }
}

struct BCVOptimInfo {
    BCVScorer bcv_scorer;
    double start_score;
    double start_determinant;
};

double wrap_bcv_diag_optim(unsigned n, const double* x, double*, void* my_func_data) {
    using MapType = Eigen::Map<const VectorXd>;
    MapType xm(x, n);

    BCVOptimInfo& optim_info = *reinterpret_cast<BCVOptimInfo*>(my_func_data);

    auto det_sqrt = xm.prod();
    auto det = det_sqrt * det_sqrt;

    if (det <= util::machine_tol || det < 1e-3 * optim_info.start_determinant ||
        det > 1e3 * optim_info.start_determinant)
        return optim_info.start_score + 10e-8;

    auto score = optim_info.bcv_scorer.score_diagonal(xm.array().square().matrix());

    if (std::abs(score) > 1e3 * std::abs(optim_info.start_score)) return optim_info.start_score + 10e-8;

    return score;
}

double wrap_bcv_optim(unsigned n, const double* x, double*, void* my_func_data) {
    using MapType = Eigen::Map<const VectorXd>;
    MapType xm(x, n);

    auto sqrt = util::invvech_triangular(xm);
    auto H = sqrt * sqrt.transpose();

    BCVOptimInfo& optim_info = *reinterpret_cast<BCVOptimInfo*>(my_func_data);

    auto det = std::exp(2 * sqrt.diagonal().array().log().sum());

    // Avoid too small/large determinants returning the start score.
    // Package ks uses 1e10 as constant.
    if (det <= util::machine_tol || det < 1e-3 * optim_info.start_determinant ||
        det > 1e3 * optim_info.start_determinant || std::isnan(det))
        return optim_info.start_score + 10e-8;

    auto score = optim_info.bcv_scorer.score_unconstrained(H);

    // Avoid scores with too much difference.
    if (std::abs(score) > 1e3 * std::abs(optim_info.start_score)) return optim_info.start_score + 10e-8;

    return score;
}

VectorXd BCV::diag_bandwidth(const DataFrame& df, const std::vector<std::string>& variables) const {
    if (variables.empty()) return VectorXd(0);

    NormalReferenceRule nr;

    auto normal_bandwidth = nr.diag_bandwidth(df, variables);

    BCVScorer bcv_scorer(df, variables);
    auto start_score = bcv_scorer.score_unconstrained(normal_bandwidth);
    auto start_determinant = normal_bandwidth.prod();

    BCVOptimInfo optim_info{/*.bcv_scorer = */ bcv_scorer,
                            /*.start_score = */ start_score,
                            /*.start_determinant = */ start_determinant};

    auto start_bandwidth = normal_bandwidth.cwiseSqrt().eval();

    nlopt::opt opt(nlopt::LN_NELDERMEAD, start_bandwidth.rows());
    opt.set_min_objective(wrap_bcv_diag_optim, &optim_info);
    opt.set_ftol_rel(1e-4);
    opt.set_xtol_rel(1e-4);
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

MatrixXd BCV::bandwidth(const DataFrame& df, const std::vector<std::string>& variables) const {
    if (variables.empty()) return MatrixXd(0, 0);
    NormalReferenceRule nr;

    auto normal_bandwidth = nr.bandwidth(df, variables);

    BCVScorer bcv_scorer(df, variables);
    auto start_score = bcv_scorer.score_unconstrained(normal_bandwidth);

    auto start_determinant = normal_bandwidth.determinant();
    BCVOptimInfo optim_info{/*.bcv_scorer = */ bcv_scorer,
                            /*.start_score = */ start_score,
                            /*.start_determinant = */ start_determinant};

    LLT<Eigen::Ref<MatrixXd>> start_sqrt(normal_bandwidth);
    auto start_vech = util::vech(start_sqrt.matrixL());


    nlopt::opt opt(nlopt::LN_NELDERMEAD, start_vech.rows());
    opt.set_min_objective(wrap_bcv_optim, &optim_info);
    opt.set_ftol_rel(1e-4);
    opt.set_xtol_rel(1e-4);
    std::vector<double> x(start_vech.rows());
    std::copy(start_vech.data(), start_vech.data() + start_vech.rows(), x.data());
    double minf;

    try {
        opt.optimize(x, minf);
    } catch (std::exception& e) {
        throw std::invalid_argument(std::string("Failed optimizing bandwidth: ") + e.what());
    }
    std::copy(x.data(), x.data() + x.size(), start_vech.data());

    auto sqrt = util::invvech_triangular(start_vech);
    auto H = sqrt * sqrt.transpose();

    return H;
}

}  // namespace kde