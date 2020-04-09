using Random
using Distributions


struct Progression
    severity::Severity
    # times given with respect to the infection time
    incubation_time::Union{Missing,TimeDiff}
    mild_symptoms_time::Union{Missing,TimeDiff}
    severe_symptoms_time::Union{Missing,TimeDiff}
    #critical_symptoms_time::Float32
    recovery_time::Union{Missing,TimeDiff}
    death_time::Union{Missing, TimeDiff}
    #Progression(severity::Severity, incubation_time::Real, mild_time::Real, severe_time::Real, recovery_time) = incubation < mild_time < severe_time < recovery_time
end



function sample_progression(rng::AbstractRNG, dist_severity, dist_incubation, dist_symptom_onset, dist_hospitalization)
    severity = rand(rng, dist_severity) |> Severity
    
    incubation_time = rand(rng, dist_incubation)
    mild_symptoms_time = incubation_time + rand(rng, dist_symptom_onset)
    severe_symptoms_time = missing 
    recovery_time = missing
    death_time = missing
    if (severity==Severe) || (severity==Critical)
      severe_symptoms_time = incubation_time + rand(dist_hospitalization)
      if severe_symptoms_time <= mild_symptoms_time
        mild_symptoms_time = missing
      end
      recovery_time = severe_symptoms_time + 14
    else
      recovery_time = mild_symptoms_time + 14
    end
    
    Progression(
        severity,
        incubation_time,
        mild_symptoms_time,
        severe_symptoms_time,
        recovery_time,
        death_time
    )
end

struct SimParams 
  household_ptrs::Vector{Tuple{UInt32,UInt32}}  # (i1,i2) where i1 and i2 are the indices of first and last member of the household
    
  progressions::Vector{Progression}
    
  constant_kernel_param::Float64
  household_kernel_param::Float64

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