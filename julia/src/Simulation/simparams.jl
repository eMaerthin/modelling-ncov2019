using Random
using Distributions
include("params/households.jl")
include("params/progression.jl")
include("params/hospital.jl")
include("params/phonetracking.jl")

include("random_sampling/friendship_sampler.jl")

struct Person
  age::Int8
  gender::Bool
  social_competence::Float32
end
struct SimParams 
  household_ptrs::Vector{Tuple{UInt32,UInt32}}  # (i1,i2) where i1 and i2 are the indices of first and last member of the household

  people::Vector{Person}

  progressions::Vector{Progression} # not sure if progressions should be there
    
  hospital_kernel_params::Union{Nothing, HospitalInfectionParams}  # nothing if hospital kernel not active  
    
  constant_kernel_param::Float64
  household_kernel_param::Float64
  friendship_kernel_param::Float64

  friendship_kernel_sampler::FriendshipSampler

  hospital_detections::Bool
  mild_detection_prob::Float64

  backward_tracking_prob::Float32
  backward_detection_delay::TimeDiff
  
  forward_tracking_prob::Float32
  forward_detection_delay::TimeDiff
  
  quarantine_length::Float32
  testing_time::TimeDiff

  phone_tracking_params::Union{Nothing, PhoneTrackingParams}
end

num_individuals(params::SimParams) = length(params.household_ptrs)
progressionof(params::SimParams, person_id::Integer) = params.progressions[person_id]
severityof(params::SimParams, person_id::Integer) = progressionof(params, person_id).severity
householdof(params::SimParams, person_id::Integer) = UnitRange(params.household_ptrs[person_id]...)
ishealthcare(params::SimParams, person_id::Integer) = 
  nothing!=params.hospital_kernel_params && ishealthcare(params.hospital_kernel_params, person_id)
uses_phone_tracking(params::SimParams, person_id::Integer) =
  nothing!=params.phone_tracking_params && uses_phone_tracking(params.phone_tracking_params, person_id)

function load_params(rng=MersenneTwister(0);
        population::Union{AbstractString,DataFrame},
        kwargs...
        )
        
  individuals_df::DataFrame = isa(population, AbstractString) ? load_individuals(population) : population
  
  num_individuals = individuals_df |> nrow

  dist_incubation_time = LogNormal(1.3669786931887833, 0.5045104580676582)
  dist_symptom_onset_time = Gamma(0.8738003969079596, 2.9148873266517685)
  dist_hospitalization_time = Gamma(1.1765988120148885, 2.6664347368236787)
  dist_mild_recovery_time = Uniform(11, 17)
  dist_severe_recovery_time = Uniform(4*7, 8*7)
  dist_death_time = LogNormal(2.610727719719777, 0.44476420066780653)
  
  progressions = Vector{Simulation.Progression}(undef, num_individuals);
  resample!(rng, progressions, individuals_df.age,
    dist_incubation_time, 
    dist_symptom_onset_time, 
    dist_hospitalization_time,
    dist_mild_recovery_time,
    dist_severe_recovery_time,
    dist_death_time
  )
  
  make_params(rng, individuals_df=individuals_df, progressions=progressions; kwargs...)
end

function make_household_ptrs!(
  ptrs::AbstractVector{Tuple{Ti,Ti}},
  household_indices::AbstractVector{T} where T<:Integer
  ) where Ti<:Integer

  @assert length(ptrs) == length(household_indices)  

  ptrarr = reshape(reinterpret(Ti, ptrs), 2, :) # tuples as 2-by-N array
  headptrs = view(ptrarr, 1, :)
  tailptrs = view(ptrarr, 2, :)
  
  groupptrs!(headptrs, tailptrs, household_indices)
  ptrs
end

make_household_ptrs(household_indices::AbstractVector{T} where T<:Real) = 
  make_household_ptrs!(Vector{Tuple{Int32, Int32}}(undef, length(household_indices)), household_indices) 

#make_household_ptrs(household_indices) = collect( zip(groupptrs(household_indices)...))


function make_params(
  rng::AbstractRNG=MersenneTwister(0);
  individuals_df::DataFrame,
  progressions::AbstractArray{Progression},
        
  hospital_kernel_param::Float64=0.0,
  healthcare_detection_prob::Float64=0.8,
  healthcare_detection_delay::Float64=1.0,
  
  constant_kernel_param::Float64=1.0,
  household_kernel_param::Float64=1.0,
        friendship_kernel_param::Float64=1.0,
  
  hospital_detections::Bool=true,
  mild_detection_prob::Float64=0.0,
  
  backward_tracking_prob::Float64=1.0,
  backward_detection_delay::Float64=1.0,
  
  forward_tracking_prob::Float64=1.0,
  forward_detection_delay::Float64=1.0,
  
  quarantine_length::Float64=14.0,
  testing_time::Float64=1.0

  phone_tracking_usage::Real=0.0,
  phone_detection_delay::Real=0.25
)

  sort!(individuals_df, :household_index)

  num_individuals = individuals_df |> nrow
    
  @assert num_individuals == length(progressions)

  household_ptrs = make_household_ptrs(individuals_df.household_index)

  population::Vector{Person} = [Person(individuals_df.age[idx], individuals_df.gender[idx], individuals_df.social_competence[idx]) for idx in 1:nrow(individuals_df)]
  friendship_sampler = FriendshipSampler(individuals_df)

  hospital_kernel_params =  if 0 == hospital_kernel_param; nothing
                            elseif 0.0 < hospital_kernel_param; 
                              HospitalInfectionParams(
                                individuals_df.ishealthcare,
                                (UInt32(1):UInt32(num_individuals))[individuals_df.ishealthcare],
                                hospital_kernel_param,
                                healthcare_detection_prob,
                                healthcare_detection_delay
                              )
                            else error("hospital_kernel_param must be postive or 0, got $hospital_kernel_param")
                            end
  phone_tracking_params = if 0 == phone_tracking_usage; nothing
                      elseif 0.0 < phone_tracking_usage <= 1.0
                        PhoneTrackingParams(rng, num_individuals, phone_tracking_usage, phone_detection_delay)
                      else error("tracking_app_usage must be nonnegative, got $phone_tracking_usage")
                      end
  

  params = SimParams(
    household_ptrs,
    population,
    progressions,
    
    hospital_kernel_params,
    
    constant_kernel_param,   
    household_kernel_param,
    friendship_kernel_param,
    friendship_sampler,
    
    hospital_detections,
    mild_detection_prob,
    
    backward_tracking_prob,
    backward_detection_delay,
    
    forward_tracking_prob,
    forward_detection_delay,
    
    quarantine_length, # quarantine length
    testing_time, # testing time

    phone_tracking_params
  )
  params
end

function saveparams(dict, p::SimParams)
  dict["constant/kernel_param"] = p.constant_kernel_param
  dict["household/kernel_param"] = p.household_kernel_param
  dict["quarantine/duration"] = p.quarantine_length
  dict["detections/hospital"] = p.hospital_detections
  dict["detections/mild"] = p.mild_detection_prob

  dict["tracking/backward_prob"] = p.backward_tracking_prob
  dict["tracking/backward_detection_delay"] = p.backward_detection_delay
  dict["tracking/forward_tracking_prob"] = p.forward_tracking_prob
  dict["tracking/forward_detection_delay"] = p.forward_detection_delay
  dict["tracking/testing_time"] = p.testing_time

  saveparams(dict, p.progressions, "progressions/")
  nothing==p.hospital_kernel_params || saveparams(dict, p.hospital_kernel_params, "hospital/")
  nothing==p.phone_tracking_params || saveparams(dict, p.phone_tracking_params, "phone_tracking/")
  nothing
end
