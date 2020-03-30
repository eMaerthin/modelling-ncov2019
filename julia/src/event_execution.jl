function execute!(state::SimState, params::SimParams, event::OutsideInfectionEvent)
  if Healthy != subjecthealth(state, event)
    return 
  end
        
  setsubjecthealth!(state, event, Infected)  
    
  push!(state.infections, 0 => event)    
  push!(state.queue, 
    BecomeInfectiousEvent(time(event) + params.progressions[subject(event)].incubation_time, subject(event))
  )
end

function execute!(state::SimState, params::SimParams, event::TransmissionEvent)
  if Healthy != subjecthealth(state, event)
    return
  end
    
  if (StayingHome == subjecthealth(state, event) ) && (event.kind != HouseholdContact)
    return 
  end
  
  setsubjecthealth!(state, event, Infected)
      
  push!(state.infections, source(event) => event)
  push!(state.queue, 
    BecomeInfectiousEvent(time(event) + params.progressions[subject(event)].incubation_time, subject(event))
  )
end

function execute!(state::SimState, params::SimParams, event::BecomeInfectiousEvent)
  @assert Infected == subjecthealth(state, event)
    
  setsubjecthealth!(state, event, Infectious)
        
  subject_id = subject(event)
  progression = params.progressions[subject_id]
  t0 = progression.incubation_time
  t1 = progression.symptom_onset_time
  t2 = progression.hospitalization_time
    
  severity = progression.severity
        
  if Mild == severity || Asymptomatic == severity
    @assert !isnan(t1)
    push!(state.queue, StayHomeMildEvent(time(event) + t1 - t0, subject_id))
  elseif Severe == severity || Critical == severity
    if !isnan(t1)
      push!(state.queue, StayHomeToHospitalizeEvent(time(event) + t1 - t0, subject_id))
    else
      push!(state.queue, GoHospitalEvent(time(event) + t2 - t0, subject_id))
    end
  else
    @error "Unsupported severity $severity"
  end

  enqueue_transmissions!(state, Val{ConstantKernelContact}, event.subject_id, params)
  #enqueue_transmissions!(HouseholdContact, state, params, event.subject_id)
end

function execute!(state::SimState, params::SimParams, event::StayHomeMildEvent)
  @assert Infectious == subjecthealth(state, event)
  setsubjecthealth!(state, event, StayingHome)  
  push!(state.queue, 
    RecoveryEvent(time(event) + 14, subject(event))
  )
end

function execute!(state::SimState, params::SimParams, event::StayHomeToHospitalizeEvent)
  @assert Infectious == subjecthealth(state, event)
  setsubjecthealth!(state, event, StayingHome)
    
  progression = params.progressions[event.subject_id]
  t1 = progression.symptom_onset_time
  t2 = progression.hospitalization_time
    
  @assert t2 > t1  
  push!(state.queue, GoHospitalEvent(time(event)+t2-t1, subject(event)))
end

function execute!(state::SimState, params::SimParams, event::GoHospitalEvent)
  @assert (StayingHome == subjecthealth(state, event)) || (Infectious == subjecthealth(state, event) )
  setsubjecthealth!(state, event, Hospitalized)
    
  severity = params.progressions[subject(event)].severity
  
  if Severe == severity # TODO: use real data here
    push!(state.queue, 
      RecoveryEvent(time(event)+14, subject(event))
    )
  elseif Critical == severity
    push!(state.queue, 
      DeathEvent(time(event)+14, subject(event))
    )
  end
end

function execute!(state::SimState, params::SimParams, event::RecoveryEvent)
  @assert Recovered !== subjecthealth(state, event)
  @assert Dead !== subjecthealth(state, event)
  setsubjecthealth!(state, event, Recovered)
end

function execute!(state::SimState, params::SimParams, event::DeathEvent)
  @assert Recovered !== subjecthealth(state, event)
  @assert Dead !== subjecthealth(state, event)
  setsubjecthealth!(state, event, Dead)
end