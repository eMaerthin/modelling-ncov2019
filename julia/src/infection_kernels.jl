function enqueue_transmissions!(state::SimState, ::Type{Val{ConstantKernelContact}}, source_id::Integer, params::SimParams)
  progression = progressionof(params, source_id)
  
    
  start_time = progression.incubation_time
  end_time = isnan(progression.mild_symptoms_time) ? progression.severe_symptoms_time : progression.mild_symptoms_time
          
  time_dist = Uniform(state.time, state.time + end_time - start_time) # in global time reference frame
    
  total_infection_rate = (end_time - start_time) * params.constant_kernel_param

  num_infections = rand(state.rng, Poisson(total_infection_rate))
  #num_infections |> display
    
  if num_infections == 0
    return
  end
    
  num_individuals = size(params.progressions, 1)
  #selected_individuals = sample(state.rng, 1:(num_individuals-1), num_infections) # exclude the source itself
    
  for _ in 1:num_infections
    subject_id = sample(state.rng, 1:(num_individuals-1)) # exclude the source itself
  #for subject_id in selected_individuals
    if subject_id >= source_id # restore the indexing
      subject_id +=1 
    end
        
    if Healthy == health(state, subject_id) 
      infection_time = rand(state.rng, time_dist)
      push!(state.queue, TransmissionEvent(infection_time, subject_id, source_id, ConstantKernelContact))
    end
  end
end

function enqueue_transmissions!(state::SimState, ::Type{Val{HouseholdContact}}, source_id::Integer, params::SimParams)
  progression = progressionof(params, source_id)
    
  start_time = progression.incubation_time
  end_time = isnan(progression.severe_symptoms_time) ? progression.recovery_time : progression.severe_symptoms_time
      
  time_dist = Uniform(state.time, state.time + end_time - start_time) # in global time reference frame
  time_dist |> display  
    
  total_infection_rate = (end_time - start_time) * params.constant_kernel_param
  household_head_ptr, household_tail_ptr = params.household_ptrs[source_id]
  
  if household_head_ptr == household_tail_ptr
    return
  end
  
  num_infections = min(rand(state.rng, Poisson(total_infection_rate)), household_tail_ptr-household_head_ptr) #shouldn't it be a binomial dist?

  if 0 == num_infections
    return
  end
  #selected_individuals = sample(state.rng, UnitRange(household_head_ptr,household_tail_ptr-Int32(1))) # exclude the source itself
    
  #for subject_id in selected_individuals
  for _ in 1:num_infections
    subject_id = sample(state.rng, UnitRange(household_head_ptr, household_tail_ptr-UInt32(1))) # exclude the source itself
    if subject_id >= source_id
      subject_id += 1 # restore the indexing
    end
    
    if Healthy == health(state, subject_id)
      push!(state.queue, TransmissionEvent(
        rand(state.rng, time_dist),
        subject_id,
        source_id,
        HouseholdContact)
      )
    end
  end
  
end