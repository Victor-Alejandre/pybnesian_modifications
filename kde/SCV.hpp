#ifndef PYBNESIAN_KDE_SCV_HPP
#define PYBNESIAN_KDE_SCV_HPP

#include <dataset/dataset.hpp>
#include <opencl/opencl_config.hpp>
#include <kde/BandwidthSelector.hpp>

using dataset::DataFrame;

namespace kde {

class SCVScorer {
public:
    SCVScorer(const DataFrame& df, const std::vector<std::string>& variables)
        : m_training_type(df.same_type(variables)),
          m_training(_copy_training_data(df, variables)),
          N(df.valid_rows(variables)),
          d(variables.size()) {}

    double score_diagonal(const VectorXd& diagonal_bandwidth) const;
    double score_unconstrained(const MatrixXd& bandwidth, const MatrixXd& bandwidth_G) const;
    double score_unconstrained_mse(const MatrixXd& bandwidth, cl::Buffer psi_previous, unsigned int derivate_order) const;
    cl::Buffer psi_nr_8(const DataFrame& df, const std::vector<std::string>& variables, const unsigned int dimension)  const;
    cl::Buffer psi_r(const MatrixXd& bandwidth, const unsigned int derivate_order) const;

private:
    template <typename ArrowType>
    double score_diagonal_impl(const Matrix<typename ArrowType::c_type, Dynamic, 1>& diagonal_sqrt_bandwidth) const;
    template <typename ArrowType, typename KDEType>
    double score_unconstrained_impl(const Matrix<typename ArrowType::c_type, Dynamic, Dynamic>& bandwidth_H, 
    const Matrix<typename ArrowType::c_type, Dynamic, Dynamic>& bandwidth_G) const; 
    template <typename ArrowType, typename KDEType>
    double score_mse_unconstrained_impl(
    const Matrix<typename ArrowType::c_type, Dynamic, Dynamic>& bandwidth, cl::Buffer psi_previous, unsigned int derivate_order) const;
    template <typename ArrowType>
    std::pair<cl::Buffer, typename ArrowType::c_type> copy_diagonal_bandwidth(
        const Matrix<typename ArrowType::c_type, Dynamic, 1>& diagonal_bandwidth) const;

    template <typename ArrowType>
    std::tuple<cl::Buffer, cl::Buffer, typename ArrowType::c_type, cl::Buffer> copy_unconstrained_bandwidth_mse(
        const Matrix<typename ArrowType::c_type, Dynamic, Dynamic>& bandwidth) const;

    template <typename ArrowType, bool contains_null>
    cl::Buffer _copy_training_data(const DataFrame& df, const std::vector<std::string>& variables) const;
    cl::Buffer _copy_training_data(const DataFrame& df, const std::vector<std::string>& variables) const;
    template <typename ArrowType>
    std::tuple<cl::Buffer, cl::Buffer, typename ArrowType::c_type> psi_r_utils(
    const Matrix<typename ArrowType::c_type, Dynamic, Dynamic>& bandwidth) const;
    template <typename ArrowType, bool contains_null>
    cl::Buffer psi_nr_8_impl(const DataFrame& df, const std::vector<std::string>& variables, const unsigned int dimension) const;
    template <typename ArrowType, typename KDEType>
    cl::Buffer psi_r_impl(const Matrix<typename ArrowType::c_type, Dynamic, Dynamic>& bandwidth, const unsigned int derivate_order) const;

    template <typename ArrowType>
    std::tuple<cl::Buffer, cl::Buffer, cl::Buffer, typename ArrowType::c_type, typename ArrowType::c_type, typename ArrowType::c_type, typename ArrowType::c_type> copy_unconstrained_bandwidth(
    const Matrix<typename ArrowType::c_type, Dynamic, Dynamic>& bandwidth_H, const Matrix<typename ArrowType::c_type, Dynamic, Dynamic>& bandwidth_G) const;
    template <typename ArrowType>
    std::vector<ArrowType> calculateCovariance(const Eigen::Matrix<ArrowType, Eigen::Dynamic, Eigen::Dynamic>& data) const; 

    template <typename ArrowType>
    std::vector<ArrowType> kroneckerProduct(const std::vector<ArrowType>& a, const std::vector<ArrowType>& b) const;
    unsigned int transform_and_permute(unsigned int i, unsigned int r, unsigned int s, unsigned int k, const unsigned int dimension) const; 
    template <typename ArrowType>
    std::vector<ArrowType> divideVectorByConstant(const std::vector<ArrowType>& vec, ArrowType constant) const;


    std::shared_ptr<arrow::DataType> m_training_type;
    cl::Buffer m_training;
    size_t N;
    size_t d;
};

class SCV : public BandwidthSelector {
public:
    VectorXd diag_bandwidth(const DataFrame& df, const std::vector<std::string>& variables) const override;
    MatrixXd bandwidth(const DataFrame& df, const std::vector<std::string>& variables) const override;

    std::string ToString() const override { return "SCV"; }

    py::tuple __getstate__() const override { return py::make_tuple(); }
    static std::shared_ptr<SCV> __setstate__(py::tuple&) { return std::make_shared<SCV>(); }
};

}  // namespace kde

#endif  // PYBNESIAN_KDE_SCV_HPP