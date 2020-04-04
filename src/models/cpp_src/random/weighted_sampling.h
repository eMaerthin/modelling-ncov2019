#pragma once

#include <cstddef>
#include <vector>
#include <unordered_map>
#include <memory>
#include "platform.h"

class Sampler final
{
    const std::vector<double>& probabilities;
    size_t idx;
    size_t to_sample_left;
    const double precision;
    const double beta_bias;
    double confs_prob;
    double chasing_prob;
    size_t current_count;

 public:
    Sampler(const std::vector<double>& probabilities, size_t to_sample, double precalculated_sum = 0.0, double beta_bias = 5.0);

    MOCOS_FORCE_INLINE size_t count() const { return current_count; };

    MOCOS_FORCE_INLINE size_t index() const { return idx; };

    bool advanceToNextConfiguration();
};


class AliasSampler final
{
    const size_t len;
    const std::shared_ptr<double[]> alias_probs; // Not unique_ptr because we want this to be copy-constructible
    const std::shared_ptr<size_t[]> alias_idxes; // As above
    std::uniform_int_distribution<size_t> UIgen;

public:
    AliasSampler(const std::vector<double>& weights, double precalculated_sum = 0.0);

    size_t gen()
    {
        size_t idx = UIgen(random_gen);
        if(alias_probs[idx] >= 1.0)
            return idx;
        if(stdunif(random_gen) < alias_probs[idx])
            return idx;
        else
            return alias_idxes[idx];
    }

    void print();
};


class NonReplacingSampler final
{
    const size_t n;
    size_t last_idx;
    std::unordered_map<size_t, size_t> replacements;

    MOCOS_FORCE_INLINE size_t get(size_t idx) { return replacements.count(idx) > 0 ? replacements[idx] : idx; };
    MOCOS_FORCE_INLINE size_t set(size_t idx, size_t towhat) { return replacements[idx] = towhat; };

 public:
    NonReplacingSampler(size_t _n);

    size_t next();
};

std::vector<size_t> ShuffledSample(const std::vector<double>& probabilities, size_t to_sample);

std::vector<double> OnesVector(size_t size)
{
    std::vector<double> ret;
    ret.reserve(size);
    for(size_t ii=0; ii<size; ii++)
        ret.push_back(1.0);
    return ret;
}
