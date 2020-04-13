function enqueue_transmissions!(state::SimState, ::Type{Val{ConstantKernelContact}}, source_id::Integer, params::SimParams)
  progression = progressionof(params, source_id)
  
    
  start_time = progression.incubation_time
              
  end_time =  if      !ismissing(progression.mild_symptoms_time);   progression.mild_symptoms_time 
              elseif  !ismissing(progression.severe_symptoms_time); progression.severe_symptoms_time
              elseif  !ismissing(progression.recovery_time);        progression.recovery_time
              else    error("no recovery nor symptoms time defined")
              end  
          
  total_infection_rate = (end_time - start_time) * params.constant_kernel_param

  num_infections = rand(state.rng, Poisson(total_infection_rate))
  #num_infections |> display
    
  if num_infections == 0
    return
  end
  @assert start_time != end_time "pathologicaly short time for infections there shouldn't be any infections but are $num_infections, progression=$progression"
  
  time_dist = Uniform(state.time, end_time - start_time + state.time) # in global time reference frame
    
  num_individuals = size(params.progressions, 1)
    
  for _ in 1:num_infections
    subject_id = sample(state.rng, 1:(num_individuals-1)) # exclude the source itself
    if subject_id >= source_id # restore the indexing
      subject_id +=1 
    end
        
    if Healthy == health(state, subject_id) 
      infection_time::TimePoint = rand(state.rng, time_dist) |> TimePoint
      @assert state.time <= infection_time <= (end_time-start_time + state.time)
      push!(state.queue, Event(Val(TransmissionEvent), infection_time, subject_id, source_id, ConstantKernelContact))
    end
  end
end

function enqueue_transmissions!(state::SimState, ::Type{Val{HouseholdContact}}, source_id::Integer, params::SimParams)
  progression = progressionof(params, source_id)
    
  start_time = progression.incubation_time
  end_time = ismissing(progression.severe_symptoms_time) ? progression.recovery_time : progression.severe_symptoms_time
  
  end_time =  if      !ismissing(progression.severe_symptoms_time); progression.severe_symptoms_time
              elseif  !ismissing(progression.recovery_time);        progression.recovery_time
              else    error("no recovery nor severe symptoms time defined")
              end
   
    
  total_infection_rate = (end_time - start_time) * params.constant_kernel_param
  household_head_ptr, household_tail_ptr = params.household_ptrs[source_id]
  
  if household_head_ptr == household_tail_ptr
    return
  end
  
  num_infections = min(
    rand(state.rng, Poisson(total_infection_rate)), 
    household_tail_ptr-household_head_ptr) #shouldn't it be a binomial dist?

  if 0 == num_infections
    return
  end

  @assert start_time != end_time "pathologicaly short time for infections there shouldn't be any infections but are $num_infections, progression=$progression"
  time_dist = Uniform(state.time, end_time - start_time + state.time) # in global time reference frame  
  
  selected_ids = state.sample_id_buf
  resize!(selected_ids, num_infections)
  
  sample!(state.rng, UnitRange(household_head_ptr, household_tail_ptr-UInt32(1)), selected_ids) # exclude the source itself
  
  infection_times = state.sample_time_buf
  resize!(infection_times, num_infections)
  
  infection_times = rand!(state.rng, time_dist, infection_times)
    
  for i in 1:num_infections
    subject_id = selected_ids[i]
    if subject_id >= source_id
      subject_id += UInt32(1) # restore the indexing
    end
    
    if Healthy == health(state, subject_id)
      infection_time = infection_times[i]
      @assert state.time <= infection_time <= (end_time - start_time + state.time)
      push!(state.queue, Event(Val(TransmissionEvent),
        infection_time,
        subject_id,
        source_id,
        HouseholdContact)
      )
    end
  end
  
#  for _ in 1:num_infections
#    subject_id = sample(state.rng, UnitRange(household_head_ptr, household_tail_ptr-UInt32(1))) # exclude the source itself
#    if subject_id >= source_id
#      subject_id += UInt32(1) # restore the indexing
#    end
#    
#    if Healthy == health(state, subject_id)
#      infection_time::TimePoint = rand(state.rng, time_dist) |> TimePoint
#      @assert state.time <= infection_time <= (end_time - start_time + state.time)
#      push!(state.queue, Event(Val(TransmissionEvent),
#        infection_time,
#        subject_id,
#        source_id,
#        HouseholdContact)
#      )
#    end
#  end
  
end

