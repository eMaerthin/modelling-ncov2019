#pragma once

#include <algorithm>
#include <iostream>
#include "weighted_sampling.h"

static double inline sum(const std::vector<double>& V)
{
    double ret = 0.0;
    for(std::vector<double>::const_iterator it = V.cbegin(); it != V.cend(); ++it)
        ret += *it;
    return ret;
}

Sampler::Sampler(const std::vector<double>& _probabilities, size_t to_sample, double _precision, double _beta_bias) :
probabilities(_probabilities),
idx(-1),
to_sample_left(to_sample),
precision(_precision > 0.0 ? _precision : sum(_probabilities)),
beta_bias(_beta_bias),
confs_prob(0.0),
chasing_prob(0.0)
{}


bool Sampler::advanceToNextConfiguration()
{
    /* This function will be used mainly in very small, tight loops, therefore it makes sense to
     * aggressively inline it, despite its seemingly large body.
     */
    while(true)
    {
        double curr_conf_prob_left, current_prob;

        if(to_sample_left <= 0)
            return false;

        if(confs_prob < chasing_prob)
        {
            // Beta was last
            current_count = 1;
            to_sample_left--;
            idx++;
            current_prob = probabilities[idx];
            confs_prob += current_prob;
            while(confs_prob <= chasing_prob)
            {
                idx++;
                current_prob = probabilities[idx];
                confs_prob += current_prob;
            }
            if(to_sample_left <= 0)
                return true;
            curr_conf_prob_left = confs_prob - chasing_prob;
        }
        else
        {
            // Binomial was last
            current_count = 0;
            idx++;
            current_prob = probabilities[idx];
            confs_prob += current_prob;
            curr_conf_prob_left = current_prob;
        }

        double prob_left_to_1 = precision - chasing_prob;
        double expected_confs = curr_conf_prob_left * to_sample_left / prob_left_to_1;

        if(expected_confs <= beta_bias)
        {
            // Beta mode: we keep making beta jumps until we leave the current configuration
            chasing_prob += rdvariate_beta_1_b(to_sample_left) * prob_left_to_1;
            while(chasing_prob <= confs_prob)
            {
                current_count++;
                to_sample_left--;
                if(to_sample_left == 0)
                    return true;
                prob_left_to_1 = precision - chasing_prob;
                chasing_prob += rdvariate_beta_1_b(to_sample_left) * prob_left_to_1;
            }
            if(current_count > 0)
                return true;
        }
        else
        {
            // Binomial mode: a single binomial step
            size_t rbin = rdvariate_binom(to_sample_left, curr_conf_prob_left/prob_left_to_1);
            current_count += rbin;
            to_sample_left -= rbin;
            chasing_prob = confs_prob;
            if(current_count > 0)
                return true;
        }
    };
}


AliasSampler::AliasSampler(const std::vector<double>& weights, double precalculated_sum) : 
len(weights.size()),
alias_probs(new double[len]),
alias_idxes(new size_t[len]),
UIgen(std::uniform_int_distribution<size_t>(0, len-1))
{
    const size_t n = weights.size();

    if(n == 0)
        return; // Calling gen() will result in undefined behaviour

    if(0.0 == precalculated_sum)
        precalculated_sum = sum(weights);

    double scaling_factor = static_cast<double>(n) / precalculated_sum;

    // TODO: replace with arrays if moar performance is ever needed
    std::vector<std::pair<size_t, double> > large;
    std::vector<std::pair<size_t, double> > small;
    large.reserve(n);
    small.reserve(n);

    for(size_t idx = 0; idx < n; idx++)
    {
        double np_i = weights[idx] * scaling_factor;

        if(np_i >= 1.0)
            large.emplace_back(idx, np_i);
        else
            small.emplace_back(idx, np_i);
    }

    while(small.size() > 0 and large.size() > 0)
    {
        auto& [small_idx, small_p] = small[small.size()-1];
        auto& [large_idx, large_p] = large[large.size()-1];

        alias_probs[small_idx] = small_p;
        alias_idxes[small_idx] = large_idx;

        large_p = (large_p + small_p) - 1.0;

        if(large_p < 1.0)
        {
            small_idx = large_idx;
            small_p = large_p;
            large.pop_back();
        }
        else
            small.pop_back();
    }

    for(auto pair : large)
        alias_probs[pair.first] = 2.0;

    for(auto pair : small)
        alias_probs[pair.first] = 2.0;
}

void AliasSampler::print()
{
    for(size_t idx =0; idx < len; idx++)
    {
        std::cout << idx << '\t' << alias_probs[idx] << '\t';
        if(alias_probs[idx] < 1.0)
            std::cout << alias_idxes[idx];
        std::cout << std::endl;
    }
}

NonReplacingSampler::NonReplacingSampler(size_t _n) :
n(_n),
last_idx(_n)
{}


size_t NonReplacingSampler::next()
{
    last_idx--;
    std::uniform_int_distribution<size_t> unif(0, last_idx);
    size_t rnd = unif(random_gen);

    size_t ret = get(rnd);

    set(rnd, get(last_idx));

    return ret;
}


std::vector<size_t> ShuffledSample(const std::vector<double>& probabilities, size_t to_sample)
{
    std::vector<size_t> ret;
    ret.reserve(to_sample);

    Sampler S(probabilities, to_sample);

    while(S.advanceToNextConfiguration())
    {
        size_t count = S.count();
        size_t index = S.index();
        for(size_t ii = 0; ii<count; ii++)
            ret.push_back(index);
    }

    std::shuffle(ret.begin(), ret.end(), random_gen);

    return ret;
}
