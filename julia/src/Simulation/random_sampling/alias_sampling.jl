using Random

struct AliasSampler{ProbabilityParamType <: Real}
    # Some alias_indices will be left uninitialized after the
    # constructor finishes. This is not a bug!
    alias_indices::Vector{Int64}
    nonalias_probs::Vector{ProbabilityParamType}
end

function AliasSampler(weights::Vector{ProbabilityParamType}) where {ProbabilityParamType <: Real}
    n = length(weights)
    s = sum(weights)
    scf = n/s
    sc_probabilities = weights*scf

    smalls = Vector{Int64}(undef, n)
    smalls_idx = 0
    larges = Vector{Int64}(undef, n)
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

    aliases = Vector{Int64}(undef, n)
    nonalias_probs = Vector{ProbabilityParamType}(undef, n)

    while larges_idx > 0 && smalls_idx > 0
        sm_p_idx = smalls[smalls_idx]
        lg_p_idx = larges[larges_idx]
        nonalias_probs[sm_p_idx] = sc_probabilities[sm_p_idx]
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
        nonalias_probs[larges[larges_idx]] = 2.0
        larges_idx -= 1
    end

    while smalls_idx > 0
        nonalias_probs[smalls[smalls_idx]] = 2.0
        smalls_idx -= 1
    end

    return AliasSampler(aliases, nonalias_probs)
    
end


function a_sample(alias_sampler::AliasSampler, rng=Random.GLOBAL_RNG)::Int64
    # Please tell me that the compiler optimizes it into something that works in O(1) and the range doesn't actually get built in memory...
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
