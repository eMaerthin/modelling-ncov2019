using DataFrames

include("alias_sampling.jl")

struct FriendshipSampler
    categories_selectors::Vector{AliasSampler}
    category_samplers::Vector{AliasSampler}
end

function FriendshipSampler(population::DataFrame, alpha::Float64 = 0.75, beta::Float64 = 1.6)

    max_age = 200

    categories = [Vector{Int64}() for _ in 1:(2*max_age)]

    function to_idx(age::Int64, gender::Bool)
        if gender
            return age+max_age+1
        else
            return age+1
        end
    end

    function to_age_gender(idx::Int64)
        if idx <= max_age
            return idx-1, false
        else
            return idx-max_age-1, true
        end
    end


    function phi(age::Int64)::Float64
        fla = Float64(a)
        if a <= 20
            return fla
        end
        return 20.0 + (fla - 20.0)^alpha
    end

    for ii in 1:nrow(population)
        push!(categories[to_idx(population.age[ii], population.gender[ii])], ii)
    end

    H = [(length(categories[to_idx(ii, false)]) + length(categories[to_idx(ii, true)])) / nrow(population) for ii in 1:max_age]

    function g(age1::Int64, age2::Int64)::Float64
        nom = H[age1] * H[age2] * exp( -0.08 * Float64(age1+age2) )
        denom = 1.0 + 0.2 * abs(phi(age1) - phi(age2)) ^ beta
        return nom/denom
    end

    categories_selectors = Vector{AliasSampler}()

    for idx in 1:(2*max_age)
        age, gender = to_age_gender(idx)
        P = [begin age2, gender2 = to_age_gender(idx2); g(age, age2) * (gender == gender2 ? 1.2 : 0.8); end for idx2 in 1:(2*max_age)]
        push!(categories_selectors, AliasSampler(P))
    end

    category_samplers = [AliasSampler([population.social_competence[person_id] for person_id in category]) for category in categories]

    return FriendshipSampler(categories_selectors, category_samplers)

end
