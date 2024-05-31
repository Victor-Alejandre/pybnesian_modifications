#include <learning/scores/holdout_likelihood.hpp>
#include <models/BayesianNetwork.hpp>
#include <kde/BandwidthSelector.hpp>

using kde::BandwidthSelector;
using models::BayesianNetworkType;

namespace learning::scores {

double HoldoutLikelihood::local_score(const BayesianNetworkBase& model,
                                      const std::string& variable,
                                      const std::vector<std::string>& evidence) const {
    return local_score(model, model.underlying_node_type(m_holdout.training_data(), variable), variable, evidence);
}

double HoldoutLikelihood::local_score(const BayesianNetworkBase& model,
                                      const std::shared_ptr<FactorType>& variable_type,
                                      const std::string& variable,
                                      const std::vector<std::string>& evidence) const {
    auto [args, kwargs] = m_arguments.args(variable, variable_type);

    auto cpd = variable_type->new_factor(model, variable, evidence, args, kwargs);
    cpd->fit(training_data());
    return cpd->slogl(test_data());
}

double HoldoutLikelihood::local_score_wbm(const BayesianNetworkBase& model,
                                const std::shared_ptr<BandwidthSelector> m_bselector,
                                const std::string& variable,
                                const std::vector<std::string>& evidence) const {
    return local_score_wbm(model, model.underlying_node_type(m_holdout.training_data(), variable), m_bselector, variable, evidence);
}

double HoldoutLikelihood::local_score_wbm(const BayesianNetworkBase& model,
                               const std::shared_ptr<FactorType>& variable_type,                            
                               const std::shared_ptr<BandwidthSelector> m_bselector, 
                               const std::string& variable,
                               const std::vector<std::string>& evidence) const {
    auto [args, kwargs] = m_arguments.args(variable, variable_type);
    if (*variable_type == CKDEType::get_ref()){
            auto cpd = variable_type->new_factor_wbm(model, variable, evidence, m_bselector, args, kwargs);
            cpd->fit(training_data());
            return cpd->slogl(test_data());           
    } else {
           auto cpd = variable_type->new_factor(model, variable, evidence, args, kwargs); 
           cpd->fit(training_data());
           return cpd->slogl(test_data());
    }    
    
}

}  // namespace learning::scores