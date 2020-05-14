include("../random_sampling/alias_sampling.jl")

const SocialCompetence = Normed{UInt16,16}
const CategoryIdx = UInt8
const InCategoryIdx = PersonIdx
const CategoryProb = Float32

const FriendshipCategorySelector = AliasSampler{CategoryIdx, CategoryProb}
const FriendshipCategorySampler = AliasSampler{InCategoryIdx, CategoryProb}

struct FriendshipSampler
    categories_selectors::Vector{FriendshipCategorySelector}
    category_samplers::Vector{FriendshipCategorySampler}
    categories::Vector{Vector{PersonIdx}}
end

const max_age = 120
const num_friendship_categories = 2*max_age+2

@inline friendship_to_idx(age::Integer, gender::Bool) = gender ? UInt(age+max_age+2) : UInt(age+1)
@inline friendship_to_age_gender(idx::Integer) = (idx <= max_age+1) ? (UInt(idx-1), false) : (UInt(idx-max_age-2), true)
@inline friendship_phi(age::Integer, alpha::Real) = age <= 20 ? float(age) : float(20+(age-20.0)^alpha)

@inline function friendship_g(age1::Integer, age2::Integer, H::AbstractVector{T} where T<:Real, alpha::Real, beta::Real)
    nom = H[age1+1] * H[age2+1] * exp( -0.08 * (age1 + age2) )
    denom = 1.0 + 0.2 * abs(friendship_phi(age1, alpha) - friendship_phi(age2, alpha)) ^ beta
    return nom/denom
end

function FriendshipSampler(
  ages::AbstractVector{T} where T<:Integer, 
  genders::AbstractVector{Bool}, 
  social_competences::AbstractVector{T} where T <: Real, 
  alpha::Real = 0.75, 
  beta::Real = 1.6)
    
  num_individuals = length(ages)
  @assert num_individuals == length(genders) == length(social_competences)
  
  categories = [Vector{PersonIdx}() for _ in 1:num_friendship_categories]
  for ii in 1:num_individuals
      push!(categories[friendship_to_idx(ages[ii], genders[ii])], ii)
  end

  H = Vector{Rational{PersonIdx}}(undef, max_age+1)
  for age in 0:max_age
    category_f = categories[friendship_to_idx(age, false)]
    category_t = categories[friendship_to_idx(age, true)]
    H[age+1] = (length(category_f) + length(category_t)) // num_individuals #should fit despite being Rational{Int}
  end

  categories_selectors = Vector{FriendshipCategorySelector}(undef, num_friendship_categories)

  for idx in 1:num_friendship_categories
      age, gender = friendship_to_age_gender(idx)
      P = Vector{CategoryProb}(undef, num_friendship_categories) # can this be moved outside the loop and re-used?
      for idx2 in 1:num_friendship_categories
          age2, gender2 = friendship_to_age_gender(idx2)
          P[idx2] = friendship_g(age, age2, H, alpha, beta) * (gender == gender2 ? 1.2 : 0.8)
      end
      categories_selectors[idx] = FriendshipCategorySelector(P)
  end

  category_samplers = Vector{FriendshipCategorySampler}(undef, num_friendship_categories)
  for cat_idx in 1:num_friendship_categories
      category = categories[cat_idx]
      alias_sampler_par = Vector{CategoryProb}(undef, length(category))
      for pers_idx in 1:length(category)
          person_id = category[pers_idx]
          category_samplers[cat_idx][pers_idx] = social_competences[person_id]
      end
  end

  return FriendshipSampler(categories_selectors, category_samplers, categories)
end

FriendshipSampler(population::DataFrame, alpha::Real = 0.75, beta::Real = 1.6) = 
    FriendshipSampler(population.age, population.gender, population.social_competence, alpha, beta)


function friend_sample(fs::FriendshipSampler, age::Integer, gender::Bool, rng=Random.GLOBAL_RNG)
    category = a_sample(fs.categories_selectors[friendship_to_idx(age, gender)], rng)
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
