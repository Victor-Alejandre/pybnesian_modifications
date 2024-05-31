#ifndef PYBNESIAN_KDE_BCV_HPP
#define PYBNESIAN_KDE_BCV_HPP

#include <dataset/dataset.hpp>
#include <opencl/opencl_config.hpp>
#include <kde/BandwidthSelector.hpp>

using dataset::DataFrame;

namespace kde {

class BCVScorer {
public:
    BCVScorer(const DataFrame& df, const std::vector<std::string>& variables)
        : m_training_type(df.same_type(variables)),
          m_training(_copy_training_data(df, variables)),
          N(df.valid_rows(variables)),
          d(variables.size()) {}

    double score_diagonal(const VectorXd& diagonal_bandwidth) const;
    double score_unconstrained(const MatrixXd& bandwidth) const;

private:
    template <typename ArrowType>
    double score_diagonal_impl(const Matrix<typename ArrowType::c_type, Dynamic, 1>& diagonal_sqrt_bandwidth) const;
    template <typename ArrowType, typename KDEType>
    double score_unconstrained_impl(const Matrix<typename ArrowType::c_type, Dynamic, Dynamic>& bandwidth) const;

    template <typename ArrowType>
    std::pair<cl::Buffer, typename ArrowType::c_type> copy_diagonal_bandwidth(
        const Matrix<typename ArrowType::c_type, Dynamic, 1>& diagonal_bandwidth) const;

    template <typename ArrowType>
    std::tuple<cl::Buffer, cl::Buffer, typename ArrowType::c_type, cl::Buffer, typename ArrowType::c_type> copy_unconstrained_bandwidth(
        const Matrix<typename ArrowType::c_type, Dynamic, Dynamic>& bandwidth) const;

    template <typename ArrowType, bool contains_null>
    cl::Buffer _copy_training_data(const DataFrame& df, const std::vector<std::string>& variables) const;
    cl::Buffer _copy_training_data(const DataFrame& df, const std::vector<std::string>& variables) const;

    std::shared_ptr<arrow::DataType> m_training_type;
    cl::Buffer m_training;
    size_t N;
    size_t d;
};

class BCV : public BandwidthSelector {
public:
    VectorXd diag_bandwidth(const DataFrame& df, const std::vector<std::string>& variables) const override;
    MatrixXd bandwidth(const DataFrame& df, const std::vector<std::string>& variables) const override;

    std::string ToString() const override { return "BCV"; }

    py::tuple __getstate__() const override { return py::make_tuple(); }
    static std::shared_ptr<BCV> __setstate__(py::tuple&) { return std::make_shared<BCV>(); }
};

}  // namespace kde

#endif  // PYBNESIAN_KDE_BCV_HPP