#pragma once
#include <cmath>



class AgeDependentFriendSampler
{
    const std::vector<size_t> csv_idxes;
    std::vector<std::vector<AliasSampler>> samplers;
    const double alpha, beta;

    double phi(double a);
    std::vector<double> h;
    double g(size_t a1, size_t a2);

 public:
    AgeDependentFriendSampler(std::vector<size_t>&& _csv_idxes, const std::vector<size_t>& ages, const std::vector<size_t>& genders, const std::vector<double>& social_competences, double _alpha = 0.75, double _beta = 1.6);
    size_t gen(size_t age, size_t gender);
};




MOCOS_FORCE_INLINE size_t AgeDependentFriendSampler::gen(size_t age, size_t gender)
{
    return csv_idxes[samplers[gender][age].gen()];
}

MOCOS_FORCE_INLINE double AgeDependentFriendSampler::phi(double a)
{
    if(a <= 20.0)
        return a;
    return 20.0 + pow(a - 20.0, alpha);
}


MOCOS_FORCE_INLINE double AgeDependentFriendSampler::g(size_t a1, size_t a2)
{
    double da1 = static_cast<double>(a1);
    double da2 = static_cast<double>(a2);

    // g's nominator:
    double ret = h[a2] * exp(-0.08*(da2));
    
    // Actually, the formula for nominator is:
    // double ret = h[a1] * h[a2] * exp(-0.08*(da1 + da2));
    // but h[a1] and da1 in exp can be factored out as only relative weights matter

    // And the denominator:
    ret = ret / (1.0 + 0.2 * pow(std::abs(phi(da1) - phi(da2)), beta));

    return ret;
}

AgeDependentFriendSampler::AgeDependentFriendSampler(std::vector<size_t>&& _csv_idxes, const std::vector<size_t>& ages, const std::vector<size_t>& genders, const std::vector<double>& social_competences, double _alpha, double _beta) :
csv_idxes(std::move(_csv_idxes)), // Dunno if cppyy can elide the copy needed here. This is ugly but makes *sure* we avoid the expensive copy.
alpha(_alpha),
beta(_beta)
{
    std::vector<double> empty_v;
    std::vector<size_t> ages_hist;
    ages_hist.resize(150);

    for(size_t age : ages)
        ages_hist[age]++;

    h.reserve(ages_hist.size());
    const double dsize = static_cast<double>(ages.size());
    for(auto it = ages_hist.begin(); it != ages_hist.end(); ++it)
        h.push_back(static_cast<double>(*it) / dsize);


    samplers.resize(2);
    for(size_t gender : {0, 1})
        for(size_t age = 0; age < ages_hist.size(); age++)
        {
            std::vector<double> probs_t;
            probs_t.reserve(_csv_idxes.size());

            if(0 == ages_hist[age])
            {
                samplers[gender].emplace_back(empty_v);
                continue;
            }

            for(size_t idx = 0; idx < ages.size(); idx++)
                probs_t.push_back(g(age, ages[idx]) * social_competences[idx] * (gender == genders[idx] ? 1.2 : 0.8));
            // The reminder: social competence of a1, scaling constant must be taken into account while sampling amount of contacts, 
            // that is, deciding how many times to call the gen() function

            samplers[gender].emplace_back(probs_t);
        }


}
