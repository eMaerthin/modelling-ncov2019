using Random
using Distributions


struct Progression
    severity::Severity
    # times given with respect to the infection time
    incubation_time::Float32
    symptom_onset_time::Float32
    hospitalization_time::Float32
    #critical_symptoms_time::Float32
    #recovery_time::Float32
    #death_time::Float32
end

function sample_progression(rng::AbstractRNG, dist_severity, dist_incubation, dist_symptom_onset, dist_hospitalization)
    severity = rand(rng, dist_severity) |> Severity
    
    incubation_time = rand(rng, dist_incubation)
    symptom_onset_time = incubation_time + rand(rng, dist_symptom_onset)
    hospitalization_time = NaN    
    
    if (severity==Severe) || (severity==Critical)
        hospitalization_time = incubation_time + rand(dist_hospitalization)
        if hospitalization_time < symptom_onset_time
            symptom_onset_time = NaN
        end
    end
    
    Progression(
        severity,
        incubation_time,
        symptom_onset_time,
        hospitalization_time
    )
end

struct SimParams 
  household_ptrs::Vector{Tuple{UInt32,UInt32}}  # (i1,i2) where i1 and i2 are the indices of first and last member of the household
    
  progressions::Vector{Progression}
    
  constant_kernel_param::Float64

  backward_tracking_prob::Float32
  backward_detection_delay::Float32
  
  forward_tracking_prob::Float32
  forward_detection_delay::Float32
  
  quarantine_length::Float32
  testing_time::Float32
end

progressionof(params::SimParams, person_id::Integer) = params.progressions[person_id]
severityof(params::SimParams, person_id::Integer) = progressionof(params, person_id).severity
householdof(params::SimParams, person_id::Integer) = UnitRange(params.household_ptrs[person_id]...)