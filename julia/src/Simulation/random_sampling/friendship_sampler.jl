using DataFrames

include("alias_sampling.jl")

struct FriendshipSampler
    categories_selectors::Vector{AliasSampler}
    category_samplers::Vector{AliasSampler}
    categories::Vector{Vector{Int64}}
end

const max_age = 120::Int64

function to_idx(age::Int8, gender::Bool)::Int64
    if gender
        return Int64(age)+max_age+2
    else
        return Int64(age)+1
    end
end

function to_age_gender(idx::Int64)::Tuple{Int8, Bool}
    if idx <= max_age+1
        return Int8(idx-1), false
    else
        return Int8(idx-max_age-2), true
    end
end


function FriendshipSampler(population::DataFrame, alpha::Float64 = 0.75, beta::Float64 = 1.6)::FriendshipSampler
    categories = [Vector{Int64}() for _ in 1:(2*max_age+2)]

    function phi(age::Int8)::Float32
        fla = Float32(age)
        if age <= 20
            return fla
        end
        return 20.0 + (fla - 20.0)^alpha
    end

    for ii in 1:nrow(population)
        push!(categories[to_idx(population.age[ii], population.gender[ii])], ii)
    end

    H = [Float32(length(categories[to_idx(ii, false)]) + length(categories[to_idx(ii, true)])) / Float32(nrow(population)) for ii::Int8 in 0:max_age]
    function h(age::Int8)
        return H[age+1]
    end

    function g(age1::Int8, age2::Int8)::Float32
        nom = h(age1) * h(age2) * exp( -0.08 * Float32(age1)+Float32(age2) )
        denom = 1.0 + 0.2 * abs(phi(age1) - phi(age2)) ^ beta
        return nom/denom
    end

    categories_selectors = Vector{AliasSampler}()

    for idx in 1:length(categories)
        age, gender = to_age_gender(idx)
        P = [
                begin
                    age2, gender2 = to_age_gender(idx2)
                    g(age, age2) * (gender == gender2 ? Float32(1.2) : Float32(0.8))
                end
                for idx2 in 1:length(categories)
            ]
        push!(categories_selectors, AliasSampler(P))
    end

    category_samplers = [AliasSampler([population.social_competence[person_id]::Float32 for person_id in category]) for category in categories]

    return FriendshipSampler(categories_selectors, category_samplers, categories)

end

function friend_sample(fs::FriendshipSampler, age::Int8, gender::Bool, rng=Random.GLOBAL_RNG)
    category = a_sample(fs.categories_selectors[to_idx(age, gender)], rng)
    return fs.categories[category][a_sample(fs.category_samplers[category], rng)]
end
