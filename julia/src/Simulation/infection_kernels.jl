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
  household = householdof(params,source_id)
  
  if 1==length(household)
    return
  end
  
  progression = progressionof(params, source_id)
    
  start_time = progression.incubation_time
  end_time = ismissing(progression.severe_symptoms_time) ? progression.recovery_time : progression.severe_symptoms_time
  
  end_time =  if      !ismissing(progression.severe_symptoms_time); progression.severe_symptoms_time
              elseif  !ismissing(progression.recovery_time);        progression.recovery_time
              else    error("no recovery nor severe symptoms time defined")
              end
   
  max_time = time(state) - start_time + end_time 
  
  mean_infection_time = (length(household)-1) / params.household_kernel_param 
  time_dist = Exponential(mean_infection_time)
  
  for subject_id in household
    if subject_id == source_id || Healthy != health(state, subject_id)
      continue
    end
      
    infection_time = time(state) + rand(rng, time_dist)
    if infection_time > max_time
      continue
    end
      
    @assert time(state) <= infection_time <= (end_time - start_time + time(state))
    push!(state.queue, Event(Val(TransmissionEvent),
      infection_time,
      subject_id,
      source_id,
      HouseholdContact)
    )
  end
end

