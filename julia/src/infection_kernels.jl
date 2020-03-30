function enqueue_transmissions!(state::SimState, ::Type{Val{ConstantKernelContact}}, source_id::Integer, params::SimParams)
  progression = params.progressions[source_id]
  
  t0 = progression.incubation_time
  t1 = progression.symptom_onset_time
  t2 = progression.hospitalization_time
    
  start_time = t0
  end_time = isnan(t1) ? t2 : t1
      
  time_dist = Uniform(start_time, end_time)
    
  total_infection_rate = (end_time - start_time) * params.constant_kernel_param

  num_infections = rand(state.rng, Poisson(total_infection_rate))
    
  if num_infections == 0
    return
  end
    
  num_individuals = size(params.progressions, 1)
  selected_individuals = sample(state.rng, 1:(num_individuals-1), num_infections) # exclude the source itself
    
  for subject_id in selected_individuals
    if subject_id >= source_id # restore the indexing
      subject_id +=1 
    end
        
    if Healthy == health(state, subject_id) 
      infection_time = rand(state.rng, time_dist)
      push!(state.queue, TransmissionEvent(state.time+infection_time, subject_id, source_id, ConstantKernelContact))
    end
  end
end