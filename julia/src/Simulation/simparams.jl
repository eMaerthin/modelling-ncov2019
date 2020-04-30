using Random
using Distributions

include("random_sampling/friendship_sampler.jl")

#struct RunParams
#  seed::Int
#
#  constant_kernel_param::Float64
#  household_kernel_param::Float64
#
#  hospital_detections::Bool
#
#  backward_tracking_prob::Float32
#  backward_detection_delay::TimeDiff
#
#  forward_tracking_prob::Float32
#  forward_detection_delay::TimeDiff
#
#  quarantine_length::Float32
#  testing_time::TimeDiff
#end
#
#struct PopulationParams
#  household_ptrs::Vector{Tuple{UInt32,UInt32}}  # (i1,i2) where i1 and i2 are the indices of first and last member of the household
#  
#end
struct HospitalInfectionParams
  ishealthcare::BitVector
  hospital_staff_ids::Vector{UInt32}
  kernel_constant::Float64
  
  healthcare_detection_prob::Float64
  healthcare_detection_delay::Float64
end

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

end

progressionof(params::SimParams, person_id::Integer) = params.progressions[person_id]
severityof(params::SimParams, person_id::Integer) = progressionof(params, person_id).severity
householdof(params::SimParams, person_id::Integer) = UnitRange(params.household_ptrs[person_id]...)
ishealthcare(params::SimParams, person_id::Integer)::Bool = nothing == params.hospital_kernel_params ? false : params.hospital_kernel_params.ishealthcare[person_id]
num_individuals(params::SimParams) = length(params.household_ptrs)


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
  resample_progressions!(rng, progressions, individuals_df.age,
    dist_incubation_time, 
    dist_symptom_onset_time, 
    dist_hospitalization_time,
    dist_mild_recovery_time,
    dist_severe_recovery_time,
    dist_death_time
  )
  
  make_params(rng, individuals_df=individuals_df, progressions=progressions; kwargs...)
end


make_household_ptrs(household_indices) = collect( zip(groupptrs(household_indices)...))

function make_params(rng::AbstractRNG=MersenneTwister(0);
        individuals_df::DataFrame,
        progressions::AbstractArray{Progression},
        
        ishealthcare::Union{Nothing, BitVector}=nothing,
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
        )
  sort!(individuals_df, :household_index)

  num_individuals = individuals_df |> nrow
    
  @assert num_individuals == length(progressions)

  household_ptrs = make_household_ptrs(individuals_df.household_index)

  population::Vector{Person} = [Person(individuals_df.age[idx], individuals_df.gender[idx], individuals_df.social_competence[idx]) for idx in 1:nrow(individuals_df)]
  friendship_sampler = FriendshipSampler(individuals_df)

  hospital_kernel_params =  if(ishealthcare==nothing || hospital_kernel_param==0.0); nothing
                            elseif 0.0 < hospital_kernel_param; 
                              HospitalInfectionParams(
                                ishealthcare,
                                (UInt32(1):UInt32(num_individuals))[individuals_df.ishealthcare],
                                hospital_kernel_param,
                                healthcare_detection_prob,
                                healthcate_detection_delay
                              )
                            else error("hospital_kernel_param must be postive, got $hospital_kernel_param")
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
    testing_time # testing time
  )
  params
end
