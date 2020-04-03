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
    household_ptrs::Vector{Tuple{Int32,Int32}}  # to be decided what it is - (list of pointers to the first member?)
    
    progressions::Vector{Progression}
    
    constant_kernel_param::Float64

end

progressionof(params::SimParams, person_id::Integer) = progressions[person_id]
severityof(params::SimParams, person_id::Integer) = progressionof(params, params_id).severity
householdof(params::SimParams, person_id::Integer) = UnitRange(household_ptrs[person_id]...)