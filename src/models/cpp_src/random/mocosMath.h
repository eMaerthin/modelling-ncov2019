#pragma once

#include <random>

extern std::mt19937 random_gen;
extern std::uniform_real_distribution<double> stdunif;

inline double rdvariate_beta_1_b(double b, std::mt19937& rgen = random_gen)
#if !MOCOS_HIDE_IMPLEMENTATIONS
{
    return 1.0 - pow(stdunif(rgen), 1.0/b);
}
#else
;
#endif

#include "btrd.h"

inline size_t rdvariate_binom(size_t tries, double succ_prob, std::mt19937& rgen = random_gen)
#if !MOCOS_HIDE_IMPLEMENTATIONS
{
    if (succ_prob >= 1.0)
        return tries;
    return boost_binomial_distribution_variate(tries, succ_prob, rgen);
}
#else
;
#endif

void mocos_seed(std::mt19937::result_type seed);


int randint(int a, int b)
{
    return std::uniform_int_distribution<decltype(a)>(a, b)(random_gen);
}

double rand_stdunif()
{
    return stdunif(random_gen);
}

double rand_lognormal(double mean, double stdev)
{
    return std::lognormal_distribution<decltype(mean)>(mean, stdev)(random_gen);
}

double rand_exponential_beta(double beta)
// Exponential random variate, following Numpy's ridiculous parametrization by beta=1.0/lambda
{
    return std::exponential_distribution<decltype(beta)>(1.0/beta)(random_gen);
}

int rand_poisson(double lambda)
{
    return std::poisson_distribution<int>(lambda)(random_gen);
}

double rand_uniform(double low, double high)
{
    return std::uniform_real_distribution<double>(low, high)(random_gen);
}

double rand_gamma(double alpha, double beta)
{
    return std::gamma_distribution<double>(alpha, beta)(random_gen);
}
