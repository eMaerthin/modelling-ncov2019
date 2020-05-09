include("../random_sampling/alias_sampling.jl")

const SocialCompetence = Normed{UInt16,16}

struct FriendshipSampler
    categories_selectors::Vector{AliasSampler}
    category_samplers::Vector{AliasSampler}
    categories::Vector{Vector{Int64}}
end

const max_age = 120

@inline to_idx(age::Integer, gender::Bool) = gender ? Int(age+max_age+2) : Int(age+1)
@inline to_age_gender(idx::Integer) = (idx <= max_age+1) ? (Int(idx-1), false) : (Int(idx-max_age-2), true)
@inline friendship_phi(age::Integer, alpha::Real) = age <= 20 ? float(age) : float(20+(age-20)^alpha)

@inline function friendship_g(age1::Integer, age2::Integer, H::AbstractVector{T} where T<:Real, alpha::Real, beta::Real)
    nom = H[age1+1] * H[age2+1] * exp( -0.08 * (age1 + age2) )
    denom = 1.0 + 0.2 * abs(friendship_phi(age1, alpha) - friendship_phi(age2, alpha)) ^ beta
    return nom/denom
end

function FriendshipSampler(
  ages::AbstractVector{Age},
  genders::AbstractVector{Bool},
  social_competences::AbstractVector{SocialCompetence},
  alpha::Real = 0.75, 
  beta::Real = 1.6)

  categories = [Vector{Int}() for _ in 1:(2*max_age+2)]

  num_individuals = length(ages)
  @assert num_individuals == length(genders) == length(social_competences)

  for ii in 1:num_individuals
    push!(categories[to_idx(ages[ii], genders[ii])], ii)
  end

  H = [Float32((length(categories[to_idx(ii, false)]) + length(categories[to_idx(ii, true)])) / num_individuals) for ii::Int8 in 0:max_age]
  categories_selectors = Vector{AliasSampler}(undef, length(categories))

  for idx in 1:length(categories)
    age, gender = to_age_gender(idx)
    P = Vector{Float32}(undef, length(categories))
    for idx2 in 1:length(categories)
      age2, gender2 = to_age_gender(idx2)
      P[idx2] = friendship_g(age, age2, H, alpha, beta) * (gender == gender2 ? Float32(1.2) : Float32(0.8))
    end
    categories_selectors[idx] = AliasSampler(P)
  end

  category_samplers = [AliasSampler([social_competences[person_id]::Float32 for person_id in category]) for category in categories]

  return FriendshipSampler(categories_selectors, category_samplers, categories)
end

FriendshipSampler(population::DataFrame, alpha::Real = 0.75, beta::Real = 1.6) = 
    FriendshipSampler(population.ages, population.genders, population.social_competences, alpha, beta)


function friend_sample(fs::FriendshipSampler, age::Integer, gender::Bool, rng=Random.GLOBAL_RNG)
    category = a_sample(fs.categories_selectors[to_idx(age, gender)], rng)
    return fs.categories[category][a_sample(fs.category_samplers[category], rng)]
end

struct FriendshipKernelParams
  kernel_constant::Real
  social_competences::Vector{SocialCompetence}
  friendship_sampler::FriendshipSampler
end

function FriendshipKernelParams(
  kernel_constant::Real, 
  ages::AbstractVector{Age},
  genders::AbstractVector{Bool},
  social_competences::Vector{SocialCompetence}) #concrete type for direct storage

  @assert all(0 .<= ages .<= max_age)

  FriendshipKernelParams(
    kernel_constant, 
    social_competences, 
    FriendshipSampler(
      ages, 
      genders, 
      social_competences))
end
socialcompetence(p::FriendshipKernelParams, person_id::Integer) = p.social_competences[person_id]
