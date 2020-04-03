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
  return true
end

function execute!(state::SimState, params::SimParams, event::TransmissionEvent)
  if Healthy != subjecthealth(state, event)
    return false
  end
    
  if (StayingHome == subjecthealth(state, event) ) && (event.kind != HouseholdContact)
    return false
  end
  
  setsubjecthealth!(state, event, Infected)
      
  push!(state.infections, source(event) => event)
  push!(state.queue, 
    BecomeInfectiousEvent(time(event) + params.progressions[subject(event)].incubation_time, subject(event))
  )
  return true
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
  
  return true
end

function execute!(state::SimState, params::SimParams, event::RecoveryEvent)
  @assert Recovered !== subjecthealth(state, event)
  @assert Dead !== subjecthealth(state, event)
  setsubjecthealth!(state, event, Recovered)
  return true
end

function execute!(state::SimState, params::SimParams, event::DeathEvent)
  @assert Recovered !== subjecthealth(state, event)
  @assert Dead !== subjecthealth(state, event)
  setsubjecthealth!(state, event, Dead)
  return true
end



#
# freedom events
#

function execute!(state::SimState, params::SimParams, event::QuarantinedEvent)
  subject_id = subject(event)
  freedom == freedom(state, subject_id)

  is_already_quarantined = isquarantined(state, subject_id)
  
  if Hospitalized == freedom
    @assert !is_already_quarantined
    return false
  end
  
  if Free == freedom
    @assert !is_already_quarantined
  elseif HomeQuarantine == freedom
    @assert is_already_quarantined
  else 
    @assert HomeTreatment == freedom
  end
  
  quarantine_advance!(state, subject_id, +1)  # increase quarantine level
  push!(state.queue, QuarantineEndEvent(time(event)+params.quarantine_length))
  return !is_alrady_quarantined
end

function execute!(state::SimState, params::SimParams, event::QuarantineEndEvent)
  subject_id = subject(event)
  
  if  Hospitalized != freedom(state, subject_id) 
    # false event as quarantine should be removed before hospitalization
    @assert isquarantined(state, subject_id)
    return false
  end
  
  quarantine_advance!(state, subject_id, -1)  # reduce quarantine level
  if isquarantined(state, subject_id)
    return false  # ignore event, the quarantine should last longer
  end
  
  return true
end


function execute!(state::SimState, params::SimParams, event::HomeTreatmentEvent)
  @assert MildSymptoms == subjecthealth(state, event)  
  freedom = subjectfreedom(state, event)
  @assert Hospitalized != freedom && HomeTreatment != freedom
  
  if Free == freedom
    setsubjectfreedom!(state, event, HomeTreatment)
  else
    @assert HomeQuarantine == freedom
    setsubjectfreedom!(state, event, HomeTreatment)
  end
  return true
end

function execute!(state::SimState, params::SimParams, event::GoHospitalEvent)
  @assert SevereSymptoms == subjecthealth(state, event)
  severity = severityof(params, subject(event))
  @assert Severe == severity || Critical == severity
  subject_id = subject(event)
  
  is_from_quarantine = isquarantined(state, subject_id)
  if is_from_quarantine  
    quarantine_cancel!(state, subject_id)
  end
  setfreedom(state, subject_id, Hospitalized)
  
  # all hospitalized cases are detected
  push!(state.queue, DetectedEvent(time(event), subject_id, is_from_quarantine)) # immediately
  return true
end

function execute!(state::SimState, params::SimParams, event::DetectedEvent)
  subject_id = subject(event)
  
  setdetected!(state, subject_id)
  
  #for member in householdof(params, subject_id)
  #  member_freedom = freedom(state, member)
  #  if Hospitalized != member_freedom
  #    push!(stat.queue, QuarantinedEvent(time(event), subject_id))  # quarantine household immediately
  #  end
  #  
  #  backtrack!()
  #end
  
  quarantinehousehold!(state, params, subject_id, include_subject=true)
  
  if !event.is_from_quarantine
    backtrackhousehold!(state, params, subject_id) 
  end
  
  return true
end

function execute!(state::SimState, params::SimParams, event::BackTrackedEvent)
  subject_id = subject(event)
  quarantinehousehold!(state, params, subject_id, include_subject=true)
  
  for member in householdof(params, subject_id)
    if detected(member)
      continue
    end
    
    member_health = health(state, member)
    @assert SevereSymptoms != member_health && CriticalSymptoms != member_health && Dead != member_health "patient should have already been infected at the hospital"
    
    if Infectious == member_health || MildSymptoms == member_health
      push!(state.queue, DetectedEvent(time(event) + params.testing_time, member, true))
    end
    
  end
end

#
# Quarantine and backtracking helpers
#

function quarantinehousehold!(state::SimState, params::SimParams, subject_id::Integer; include_subject::Bool)
  for member in householdof(subject_id)
    if !include_subject && (member == subject_id)
      continue
    end
    
    member_freedom = freedom(state, member)

    if (Hospitalized == member_freedom) || (Released == member_freedom)
      continue
    end
    @assert (Free == member_freedom) || (HomeTreatment == member_freedom) 

    push!(state.queue, QuarantinedEvent(state.time, subject_id)) #immediately
  end
end

backtrackhousehold!(state::SimState, params::SimParams, subject_id::Integerr; track_household_connections::Bool=false) = for member in householdof(params, subject_id); backtrack!(state, params, memeber, track_household_connections=track_household_connections) end

function backtrack!(state::SimState, params::SimParams, person_id::Integer; track_household_connections::Bool=false)
  current_time = state.global_time
  
  backward_id = infectedfrom(state, person_id)
  @assert !ismissing(backward_id) && 0 != source_id "infection source must have been infected"
    
  if !detected(state, person_id) # check if the source was not already deteced, hence already backtracked
    if params.backward_tracking_prob < rand(state.rng)    
      push!(state.queue, BackTrackedEvent(current_time + params.source_detection_delay, backward_id))
    end
  end
  
  # handle all outgoing infections
  for (source_id, infection) in forward_infections(state, subject_id)
    contact_kind = contactkind(event)
    @assert OutsideContact != contact_kind 
      
    @assert source_id==subject_id
    forward_id = subject(infection)

    health_state = health(state, forward_id)    
    @assert Healthy != health_state
    
    if !track_household_connections && (HouseholdContact == contact_kind)
      continue
    end

    if detected(state, forward_id)
      # the check for detected should be enough for avoiding loops and multiple checks of the same person
      # - in case there are multiple backtracing processes at the same time it does not affect the chances of being tested positively
      # - here we are looking for the contacts and all the contacts  already are or were infected
      # - only true infections are stored hence no one will be forward tracked twice anyway
      # it is just the chance of being detected which becomes larger if multiple backtrackings are leading to the same person
      continue
    end
    
    @assert SevereSymptoms != health_state && CriticalSymptoms != health_state && Dead != health_state "the patient should have already been detected at the hospital"
    
    # the probability that the contact is NOT tracked
    if params.forward_tracking_prob >= rand(state.rng)
      continue
    end
    # if it is found
    push!(state.queue, BackTrackedEvent(current_time + params.contact_detection_delay, forward_id))
  end
end



#function execute!(state::SimState, params::SimParams, event::CriticalSymptomsEvent)
#  return false
#end








