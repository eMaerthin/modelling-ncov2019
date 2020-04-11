struct AliasSampler
    alias_indices::Vector{Int64}
    nonalias_probs::Vector{Float64}
end

function AliasSampler(weights::Vector{Float64})
    probabilities = normalize(weights)
    n = length(weights)

    smalls = Vector{Int64}(undefined, n)
    smalls_idx = 0
    larges = Vector{Int64}(undefined, n)
    larges_idx = 0

    avg_prob = 1.0/n

    for idx in 1:n
        if probabilities[idx] >= avg_prob
            larges_idx += 1
            larges[larges_idx] = idx
        else
            smalls_idx += 1
            smalls[smalls_idx] = idx
        end
    end

    aliases = Vector{Int64}(undefined)
    nonalias_probs = Vector{Float64}(undefined)

    while larges_idx > 0 && smalls_idx > 0
        sm_p_idx = smalls[smalls_idx]
        lg_p_idx = larges[larges_idx]
        nonalias_probs[sm_p_idx] = probabilities[sm_p_idx]
        aliases[sm_p_idx] = lg_p_idx

        # This is slightly better numerically than the more obvious: probabilities[lg_p_idx] += probabilities[sm_p_idx] - 1.0
        probabilities[lg_p_idx] -= 1.0
        probabilities[lg_p_idx] += probabilities[sm_p_idx] 

        if probabilities[lg_p_idx] < 1.0
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


function sample(alias_sampler::AliasSampler, rng=Random.GLOBAL_RNG <: AbstractRNG)::Int64
    # Please tell me that the compiler optimizes it into something that works in O(1) and the range doesn't actually get built in memory...
    idx = rand(rng, 1:length(alias_sampler.alias_idxes))
    if nonalias_probs[idx] >= 1.0
        return idx
    end
    if rand(rng) < nonalias_probs[idx]
        return idx
    else
        return alias_indices[idx]
    end
end
