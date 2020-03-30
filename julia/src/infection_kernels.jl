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

function enqueue_transmissions!(state::SimState, ::Type{Val{HouseholdContact}}, source_id::Integer, params::SimParams)
  progression = params.progressions[source_id]
  
  t0 = progression.incubation_time
  t1 = progression.symptom_onset_time
  t2 = progression.hospitalization_time
  recovery_time = t0 + 14
    
  start_time = t0
  end_time = isnan(t2) ? recovery_time : t2
      
  time_dist = Uniform(start_time, end_time)
    
  total_infection_rate = (end_time - start_time) * params.constant_kernel_param
  household_head_ptr, household_tail_ptr = params.household_ptrs[source_id]
  
  if household_head_ptr == household_tail_ptr
    return
  end
  
  num_infections = min(rand(state.rng, Poisson(total_infection_rate)), household_tail_ptr-household_head_ptr) #shouldn't it be a binomial dist?

  if 0 == num_infections
    return
  end
  selected_individuals = sample(state.rng, UnitRange(household_head_ptr,household_tail_ptr-Int32(1))) # exclude the source itself
    
  for subject_id in selected_individuals
    if subject_id >= source_id
      subject_id += 1 # restore the indexing
    end
    
    if Healthy == health(state, subject_id)
      push!(state.queue, TransmissionEvent(
        state.time+rand(state.rng, time_dist),
        subject_id,
        source_id,
        HouseholdContact)
      )
    end
    
    
  end
  
end