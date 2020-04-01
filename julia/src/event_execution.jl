#
# transmissions
#
function execute!(state::SimState, params::SimParams, event::OutsideInfectionEvent)
  if Healthy != subjecthealth(state, event)
    return false
  end
        
  setsubjecthealth!(state, event, Infected)  
    
  push!(state.infections, 0 => event)    
  push!(state.queue, 
    BecomeInfectiousEvent(time(event) + params.progressions[subject(event)].incubation_time, subject(event))
  )
  true
end

function execute!(state::SimState, params::SimParams, event::TransmissionEvent)
  if Healthy != subjecthealth(state, event)
    return false
  end
    
  if (StayingHome == subjecthealth(state, event) ) && (event.kind != HouseholdContact)
    return 
  end
  
  setsubjecthealth!(state, event, Infected)
      
  push!(state.infections, source(event) => event)
  push!(state.queue, 
    BecomeInfectiousEvent(time(event) + params.progressions[subject(event)].incubation_time, subject(event))
  )
  true
end

#
# Disease progression
#

function execute!(state::SimState, params::SimParams, event::BecomeInfectiousEvent)
  @assert Incubating == subjecthealth(state, event)
    
  setsubjecthealth!(state, event, Infectious)
        
  subject_id = subject(event)
  progression = params.progressions[subject_id]
  t0 = progression.incubation_time
  t1 = progression.symptom_onset_time
  t2 = progression.hospitalization_time
    
  severity = progression.severity
        
  if Mild == severity || Asymptomatic == severity # we treat all asymptomatic as if they were Mild cases
    @assert !isnan(t1)
    push!(state.queue, MildSymptomsEvent(time(event) + t1 - t0, subject_id))
  elseif Severe == severity || Critical == severity # and all critical as if they were Severe cases
    if !isnan(t1)
      push!(state.queue, MildSymptomsEvent(time(event) + t1 - t0, subject_id))
    else
      push!(state.queue, SevereSymptomsEvent(time(event) + t2 - t0, subject_id))
    end
  else
    @error "Unsupported severity $severity"
  end

  enqueue_transmissions!(state, Val{ConstantKernelContact}, event.subject_id, params)
  enqueue_transmissions!(state, Val{HouseholdContact}, event.subject_id, params)
  return true
end

function execute!(state::SimState, params::SimParams, event::MildSymptomsEvent)
  @assert Infectious == subjecthealth(state, event)
  setsubjecthealth!(state, event, MildSymptoms)
  
  progression = params.progressions[subject_id]
  @assert !isnan(progression.symptom_onset_time)
  

  if Severe == progression.severity || Critical == progression.severity
    @assert !isnan(progression.t2)
    push!(state.queue, SevereSymptomsEvent(time(event) + progression.symptom_onset_time - progression.incubation_time), subject(event))
  else
    @assert Mild == progression.severity
    push!(state.quque, RecoveryEvent(time(event) + 14, subject(event)))
  end

  push!(state.quque, HomeTreatmentEvent(time(event), subject(event))) #immediately
  return true
end

function execute!(state::SimState, params::SimParams, event::SevereSymptomsEvent)
  @assert Infectious == subjecthealth(state, event) || MildSymptoms == subjecthealth(state, event)
  setsubjecthealth!(state, event, SevereSymptomsEvent)
  
  #push!(state.queue, RecoveryEvent(time(event)+14, subject(event)))
  push!(state.queue, GoHospitalEvent(time(event), subject(event)))  # immediately
  
  if params.severe_detection_prob < rand(rng)
    quarantinehousehold!(state, params, subject(event), include_subject=false)  # quarantine everybody except the subject as he is staying in the hospital until recovery
    backtrackhousehold!(state, params, subject(event))
  end  
  return true
end




#
# freedom events
#

function execute!(state::SimState, params::SimParams, event::StayHomeTreatmentEvent)
  @assert Infectious == subjecthealth(state, event)
  setsubjectfreedom!(state, event, HomeTreatment)
  return true
end

function execute!(state::SimState, params::SimParams, event::GoHospitalEvent)
  @assert Severe == subjecthealth(state, event) || Critical == subjecthealth(state, event)
  # all hospitalized cases are detected
  push!(state.queue, DetectedEvent(time(event), subject(event))) # immediately
  return true
end

function execute!(state::SimState, params::SimParams, event::QuarantinedEvent)
  subject_id = subject(event)
  @assert Hospitalized != freedom(state, subject_id)
  is_already_quarantined = isquarantined(state, subject_id)
  quarantine_advance!(state, subject_id, +1)  # increase quarantine level
  push!(state.queue, QuarantineEndEvent(time(event)+params.quarantine_length))
  return !is_alrady_quarantined
end

function execute!(state::SimState, params::SimParams, event::QuarantineEndEvent)
  subject_id = subject(event)
  @assert Hospitalized != freedom(state, subject_id)
  
  quarantine_advance!(state, subject_id, -1)  # reduce quarantine level
  if isquarantined(state, subject_id)
    return false  # ignore event, the quarantine should last longer
  end
  
  return true
end


#
# Quarantine and backtracking helpers
#

function quarantinehousehold!(state::SimState, params::SimParams, subject_id::Integer; include_subject::Bool)
  household_members = UnitRange(params.households[subject_id]...)
  for member in household_memebers
    if include_subject || member!=subject_id
      push!(state.queue, StayHomeQuarantinedEvent(state.time), subject_id) #immediately
    end
  end
end

backtrackhousehold!(state::SimState, params::SimParams, subject_id::Integer) = for member in UnitRange(params.households[subject_id]...); backtrack!(state, params, memeber) end

function backtrack!(state::SimState, params::SimParams, subject_id::Integer)
  
end



#function execute!(state::SimState, params::SimParams, event::CriticalSymptomsEvent)
#  return false
#end




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

function execute!(state::SimState, params::SimParams, event::DetectedEvent)
  setdetected!(state, subject(event))
end

