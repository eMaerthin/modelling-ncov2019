using Random

struct AliasSampler{IndexType <: Int, ProbabilityParamType <: Real}
    # Some alias_indices will be left uninitialized after the
    # constructor finishes. This is not a bug!
    alias_indices::Vector{IndexType}
    nonalias_probs::Vector{ProbabilityParamType}
end

function AliasSampler{IndexType, ProbabilityParamType}(
                weights::Vector{ProbabilityParamType}
            ) where{IndexType <: Int, ProbabilityParamType <: Real}

    n = length(weights)
    s = sum(weights)
    scf = n/s
    sc_probabilities = (weights*scf)::Vector{ProbabilityParamType}

    smalls = Vector{IndexType}(undef, n)
    smalls_idx = 0
    larges = Vector{IndexType}(undef, n)
    larges_idx = 0

    for idx in 1:n
        if sc_probabilities[idx] >= 1.0
            larges_idx += 1
            larges[larges_idx] = idx
        else
            smalls_idx += 1
            smalls[smalls_idx] = idx
        end
    end

    aliases = Vector{IndexType}(undef, n)

    while larges_idx > 0 && smalls_idx > 0
        sm_p_idx = smalls[smalls_idx]
        lg_p_idx = larges[larges_idx]
        aliases[sm_p_idx] = lg_p_idx

        # This is slightly better numerically than the more obvious: sc_probabilities[lg_p_idx] += sc_probabilities[sm_p_idx] - 1.0
        sc_probabilities[lg_p_idx] -= 1.0
        sc_probabilities[lg_p_idx] += sc_probabilities[sm_p_idx] 

        if sc_probabilities[lg_p_idx] < 1.0
            smalls[smalls_idx] = lg_p_idx
            larges_idx -= 1
        else
            smalls_idx -= 1
        end
    end

    while larges_idx > 0
        sc_probabilities[larges[larges_idx]] = 2.0
        larges_idx -= 1
    end

    while smalls_idx > 0
        sc_probabilities[smalls[smalls_idx]] = 2.0
        smalls_idx -= 1
    end

    return AliasSampler(aliases, sc_probabilities)
    
end


function a_sample(
                        alias_sampler::AliasSampler{IndexType, ProbabilityParamType},
                        rng=Random.GLOBAL_RNG
                 )::IndexType where{IndexType <: Int, ProbabilityParamType <: Real}

    idx = rand(rng, 1:length(alias_sampler.alias_indices))
    if alias_sampler.nonalias_probs[idx] >= 1.0
        return idx
    end
    if rand(rng) < alias_sampler.nonalias_probs[idx]
        return idx
    else
        return alias_sampler.alias_indices[idx]
    end
end
