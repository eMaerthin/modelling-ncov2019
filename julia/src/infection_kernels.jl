function enqueue_transmissions!(state::SimState, ::Type{Val{ConstantKernelContact}}, source_id::Integer, params::SimParams)
    t0 = incubation_times[source_id]
    t1 = symptom_onset_times[source_id]
    t2 = hospitalization_times[source_id]
    
    start_time = t0
    end_time = isnan(t1) ? t2 : t1
        
    time_dist = Uniform(start_time, end_time)
    
    total_infection_rate = (end_time - start_time) * params.constant_kernel_param
    num_infections = random(Poisson(total_infection_rate))
    
    if num_infections == 0
        return
    end
    
    num_individuals = size(params.individual_df, 1)
    selected_individuals = sample(num_individuals-1, num_infecitons) # exclude the source itself
    
    for subject_id in selected_individuals
        if subject_id >= source_id # restore the indexing
            subject_id +=1 
        end
        
        if Healthy == subjecthealt(subject_id) 
            infection_time = rand(time_dist)
            push!(state.queue, TransmissionEvent(infection_time, source_id, subject_id, ConstantKernelContact))
        end
    end
end