#pragma once
#include <cmath>
#include "population.h"

namespace mocos_cpp
{

class AgeDependentFriendSampler
{
    Population& population;
    std::vector<std::vector<AliasSampler>> samplers;
    const double alpha, beta;

    double phi(double a);
    std::vector<double> h;
    double g(size_t a1, size_t a2);

 public:
    AgeDependentFriendSampler(Population& pop, double _alpha = 0.75, double _beta = 1.6);
    Person& gen(size_t age, size_t gender);
};




MOCOS_FORCE_INLINE Person& AgeDependentFriendSampler::gen(size_t age, size_t gender)
{
    return population[samplers[gender][age].gen()];
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

AgeDependentFriendSampler::AgeDependentFriendSampler(Population& pop, double _alpha, double _beta) :
population(pop),
alpha(_alpha),
beta(_beta)
{
    std::vector<double> empty_v;
    std::vector<size_t> ages_hist;
    ages_hist.resize(150);

    for(Person& p : population)
        ages_hist[p.age]++;

    h.reserve(ages_hist.size());
    const double dsize = static_cast<double>(population.size());
    for(auto it = ages_hist.begin(); it != ages_hist.end(); ++it)
        h.push_back(static_cast<double>(*it) / dsize);


    samplers.resize(2);
    for(size_t gender : {0, 1})
        for(size_t age = 0; age < ages_hist.size(); age++)
        {
            std::vector<double> probs_t;
            probs_t.reserve(population.size());

            if(0 == ages_hist[age])
            {
                samplers[gender].emplace_back(empty_v);
                continue;
            }

            for(Person& p : population)
                probs_t.push_back(g(age, p.age) * p.social_competence * (gender == p.gender ? 1.2 : 0.8));
            // The reminder: social competence of a1, scaling constant must be taken into account while sampling amount of contacts, 
            // that is, deciding how many times to call the gen() function

            samplers[gender].emplace_back(probs_t);
        }


}

} // namespace mocos_cpp
