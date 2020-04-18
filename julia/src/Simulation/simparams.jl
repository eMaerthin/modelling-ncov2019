using Random
using Distributions

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

#
 

#
struct Progression
    severity::Severity
    # times given with respect to the infection time
    incubation_time::TimeDiff
    mild_symptoms_time::Union{Missing,TimeDiff}
    severe_symptoms_time::Union{Missing,TimeDiff}
    #critical_symptoms_time::Float32
    recovery_time::Union{Missing,TimeDiff}
    death_time::Union{Missing, TimeDiff}
    #Progression(severity::Severity, incubation_time::Real, mild_time::Real, severe_time::Real, recovery_time) = incubation < mild_time < severe_time < recovery_time
end
Progression() = Progression(Asymptomatic, missing, missing, missing, missing, missing)

# Asymptotic not allowed as yet (some asserts prevent them)
const severity_dists = [
  Categorical([0,  0.852,  0.144,  0.004]),
  Categorical([0,  0.848,  0.144,  0.008]),
  Categorical([0,  0.832,  0.141,  0.027]),
  Categorical([0,  0.793,  0.134,  0.073]),
  Categorical([0,  0.716,  0.121,  0.163]),
  Categorical([0,  0.596,  0.102,  0.302]),
]

#const severity_dists_hacked = [
#  Categorical([0,  0.746,  0.250,  0.004]),
#  Categorical([0,  0.742,  0.250,  0.008]),
#  Categorical([0,  0.723,  0.250,  0.027]),
#  Categorical([0,  0.677,  0.250,  0.073]),
#  Categorical([0,  0.587,  0.250,  0.163]),
#  Categorical([0,  0.448,  0.250,  0.302]),
#]

function sample_severity(rng::AbstractRNG, age::Real)
  dist = severity_dists[1]
  if age < 0;       error("age should be non-negative")
  elseif age < 40;  dist = severity_dists[1]
  elseif age < 50;  dist = severity_dists[2]    
  elseif age < 60;  dist = severity_dists[3]
  elseif age < 70;  dist = severity_dists[4]
  elseif age < 80;  dist = severity_dists[5]
  else;             dist = severity_dists[6]
  end
  severity_int = rand(rng, dist)
  severity = rand(rng, dist) |> Severity
end


@inline function sample_progression(rng::AbstractRNG, age::Real, 
    dist_incubation, 
    dist_symptom_onset, 
    dist_hospitalization,         
    dist_mild_recovery,
    dist_severe_recovery
  )

  severity = sample_severity(rng, age)
 
  mild_symptoms_time = missing
  severe_symptoms_time = missing
  recovery_time = missing
  death_time = missing
    
  incubation_time = rand(rng, dist_incubation)
   
  mild_symptoms_time = incubation_time + rand(rng, dist_symptom_onset)
  
  severe_symptoms_time = missing 
  recovery_time = missing
  death_time = missing
  if (severity==Severe) || (severity==Critical)
    severe_symptoms_time = incubation_time + rand(rng, dist_hospitalization)
    if severe_symptoms_time <= mild_symptoms_time
      mild_symptoms_time = missing
    end
    recovery_time = severe_symptoms_time + rand(rng, dist_severe_recovery)
  else
    recovery_time = mild_symptoms_time + rand(rng, dist_mild_recovery)
  end
    
  Progression(
    severity,
    incubation_time |> TimeDiff,
    ismissing(mild_symptoms_time) ? missing : TimeDiff(mild_symptoms_time),
    ismissing(severe_symptoms_time) ? missing : TimeDiff(severe_symptoms_time),
    ismissing(recovery_time) ? missing : TimeDiff(recovery_time),
    ismissing(death_time) ? missing : TimeDiff(death_time)
  )
end

function resample_progressions!(rng::AbstractRNG, 
                                progressions, ages::AbstractArray{T} where T <: Real,       
                                dist_incubation_time, 
                                dist_symptom_onset_time, 
                                dist_hospitalization_time,
                                dist_mild_recovery_time,
                                dist_severe_recovery_time
                                )

  for i in 1:length(ages)
    progressions[i] = Simulation.sample_progression(rng, ages[i],         
      dist_incubation_time, 
      dist_symptom_onset_time, 
      dist_hospitalization_time,
      dist_mild_recovery_time,
      dist_severe_recovery_time)
  end
  progressions
end

struct SimParams 
  household_ptrs::Vector{Tuple{UInt32,UInt32}}  # (i1,i2) where i1 and i2 are the indices of first and last member of the household
    
  progressions::Vector{Progression}
    
  constant_kernel_param::Float64
  household_kernel_param::Float64

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
  
  progressions = Vector{Simulation.Progression}(undef, num_individuals);
  resample_progressions!(rng, progressions, individuals_df.age,
    dist_incubation_time, 
    dist_symptom_onset_time, 
    dist_hospitalization_time,
    dist_mild_recovery_time,
    dist_severe_recovery_time
  )
  
  make_params(rng, individuals_df=individuals_df, progressions=progressions; kwargs...)
end

make_household_ptrs(household_indices) = collect( zip(groupptrs(household_indices)...))

function make_params(rng::AbstractRNG=MersenneTwister(0);
        individuals_df::DataFrame,
        progressions::AbstractArray{Progression},

        constant_kernel_param::Float64=1.0,
        household_kernel_param::Float64=1.0,
        
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
  
  params = SimParams(
    household_ptrs,
    progressions,        
    
    constant_kernel_param,   
    household_kernel_param,
    
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