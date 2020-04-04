#
# transmissions
#
function execute!(state::SimState, params::SimParams, event::OutsideInfectionEvent)
  if Healthy != subjecthealth(state, event)
    return false
  end
        
  setsubjecthealth!(state, event, Incubating)  
    
  registerinfection!(state, event) 
  incubation_time = progressionof(params, subject(event)).incubation_time 
  push!(state.queue, 
    BecomeInfectiousEvent(time(event) + incubation_time, subject(event))
  )
  return true
end

function execute!(state::SimState, params::SimParams, event::TransmissionEvent)

  
  if Healthy != subjecthealth(state, event) # apparently the subject became already infected in some other way
    return false
  end
  
  source_health = sourcehealth(state, event)
  @assert source_health != Healthy && source_health != Recovered && source_health != Incubating
  @assert source_health != SevereSymptoms && source_health != CriticalSymptoms && source_health != Dead && source_health != Recovered "sampled time exceeds infectability time frame, subject is now in state $source_health, the event is $event"
    
  # the transmission events are queued in advace, therefore it might be the case that it can not be realized
  # for the transmission to happen both source and subject must be free or both must be staying at home in case
  # the infection takes place inside household

  subject_freedom = subjectfreedom(state, event)
  source_freedom = sourcefreedom(state, event)  

  @assert subject_freedom != HomeTreatment && subject_freedom != Hospitalized "a healthy subject should not be in HomeTreatment or Hospital"
  @assert source_freedom != Hospitalized

  # household contact conditions  
  # if it is an infection inside household and either source or subject are closed in home the event can not happen
  if (contactkind(event) != HouseholdContact) && ( (HomeTreatment == source_freedom) || (HomeQuarantine == source_freedom) || HomeQuarantine == subject_freedom) 
    return false
  end
  
  setsubjecthealth!(state, event, Incubating)
      
  registerinfection!(state, event)
  
  incubation_time = progressionof(params, subject(event)).incubation_time
  push!(state.queue, 
    BecomeInfectiousEvent(time(event) + incubation_time, subject(event))
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
 
    
  severity = progression.severity
  
  infected_time = time(event) - progression.incubation_time 
  if Asymptotic == severity
    @assert !isnan(progression.recovery_time)
    push!(state.quque, Recovered(infected_time + progression.recovery_time), subject_id)   
  elseif Mild == severity
    @assert !isnan(progression.mild_symptoms_time)
    push!(state.queue, MildSymptomsEvent(infected_time + progression.mild_symptoms_time, subject_id))
  elseif Severe == severity || Critical == severity # and all critical as if they were Severe cases
    if !isnan(progression.mild_symptoms_time)
      push!(state.queue, MildSymptomsEvent(infected_time + progression.mild_symptoms_time, subject_id))
    else
      push!(state.queue, SevereSymptomsEvent(infected_time + progression.severe_symptoms_time, subject_id))
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
  subject_id = subject(event)
  
  progression = progressionof(params, subject_id)
  @assert !isnan(progression.mild_symptoms_time)
  
  infection_time = time(event) - progression.mild_symptoms_time

  if Severe == progression.severity || Critical == progression.severity
    @assert !isnan(progression.severe_symptoms_time)
    @assert infection_time + progression.severe_symptoms_time > time(event)
    push!(state.queue, SevereSymptomsEvent(infection_time + progression.severe_symptoms_time, subject_id))
  else
    @assert infection_time + progression.recovery_time > time(event) "next event time $(infection_time + progression.recovery_time) is than current event $event"
    @assert (Mild == progression.severity) "unexpected severity $(progression.severity)"
    push!(state.queue, RecoveryEvent(infection_time + progression.recovery_time, subject_id))
  end

  push!(state.queue, HomeTreatmentEvent(time(event), subject_id)) #immediately
  return true
end

function execute!(state::SimState, params::SimParams, event::SevereSymptomsEvent)
  @assert Infectious == subjecthealth(state, event) || MildSymptoms == subjecthealth(state, event)
  setsubjecthealth!(state, event, SevereSymptoms)
  const subject_id = subject(event)
  const progression = progressionof(params, subject_id)

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
  setfreedom!(state, subject_id, Hospitalized)
  
  # all hospitalized cases are detected
  push!(state.queue, DetectedEvent(time(event), subject_id, is_from_quarantine)) # immediately
  return true
end

function execute!(state::SimState, params::SimParams, event::DetectedEvent)
  subject_id = subject(event)
  
  setdetected!(state, subject_id)
  
  quarantinehousehold!(state, params, subject_id, include_subject=true)
  
  #
  if !event.is_from_quarantine
    backtrackhousehold!(state, params, subject_id) 
  end
  
  return true
end

function execute!(state::SimState, params::SimParams, event::BackTrackedEvent)
  subject_id = subject(event)
  quarantinehousehold!(state, params, subject_id, include_subject=true)
  
  for member in householdof(params, subject_id)
    if isdetected(state, member)
      continue
    end
    
    member_health = health(state, member)
    @assert SevereSymptoms != member_health && CriticalSymptoms != member_health && Dead != member_health "patient should have already been infected at the hospital"
    
    if Infectious == member_health || MildSymptoms == member_health
      push!(state.queue, DetectedEvent(time(event) + params.testing_time, member, true))
    end
  end
  return true
end

function execute!(state::SimState, params::SimParams, event::QuarantinedEvent)
  subject_id = subject(event)
  freedom_state = freedom(state, subject_id)

  is_already_quarantined = isquarantined(state, subject_id)
  
  if Hospitalized == freedom_state
    @assert !is_already_quarantined "the quarantine should be lifted just before hospitalization"
    return false
  end
  
  if Free == freedom_state
    @assert !is_already_quarantined "quarantined cannot be free"
    setfreedom!(state, subject_id, HomeQuarantine)
  elseif HomeQuarantine == freedom_state
    @assert is_already_quarantined "person's state is quarantine therefore isquarantine should return true"
  else 
    @assert HomeTreatment == freedom_state "bad freedom status detected = $freedom_state"
  end
  
  quarantine_advance!(state, subject_id, +1)  # increase quarantine level
  push!(state.queue, QuarantineEndEvent(time(event)+params.quarantine_length, subject_id))

  return !is_already_quarantined
end

function execute!(state::SimState, params::SimParams, event::QuarantineEndEvent)
  subject_id = subject(event)
  subject_freedom = freedom(state, subject_id)
  if  (Hospitalized == subject_freedom) || (Released == subject_freedom)
    # false event as quarantine should be removed before hospitalization
    @assert !isquarantined(state, subject_id) "subject in state $subject_freedom detected in quarantine"
    return false
  end
  
  quarantine_advance!(state, subject_id, -1)  # reduce quarantine level
  if isquarantined(state, subject_id)
    return false  # ignore event, the quarantine should last longer
  end
  setfreedom!(state, subject_id, Free)
  
  return true
end

#
# Quarantine and backtracking helpers
#

function quarantinehousehold!(state::SimState, params::SimParams, subject_id::Integer; include_subject::Bool)
  for member in householdof(params, subject_id)
    if !include_subject && (member == subject_id)
      continue
    end
    
    member_freedom = freedom(state, member)

    if (Hospitalized == member_freedom) || (Released == member_freedom)
      continue
    end
    @assert (Free == member_freedom) || (HomeTreatment == member_freedom) || (HomeQuarantine == member_freedom)

    push!(state.queue, QuarantinedEvent(state.time, member)) #immediately
  end
end

backtrackhousehold!(state::SimState, params::SimParams, subject_id::Integer; track_household_connections::Bool=false) = for member in householdof(params, subject_id); backtrack!(state, params, member, track_household_connections=track_household_connections) end

function backtrack!(state::SimState, params::SimParams, person_id::Integer; track_household_connections::Bool=false)
  current_time = state.time
  
  backward_id = backwardinfection(state, person_id)
  if 0 != backward_id # cannont backtrack outside infections
    if !isdetected(state, person_id) # check if the source was not already deteced, hence already backtracked
      if rand(state.rng) < params.backward_tracking_prob     
        push!(state.queue, BackTrackedEvent(current_time + params.backward_detection_delay, backward_id))
      end
    end
  end
  
  # handle all outgoing infections
  for (source_id, infection) in forwardinfections(state, person_id)
    
    contact_kind = contactkind(infection)
    @assert OutsideContact != contact_kind 
      
    @assert source_id == person_id
    forward_id = subject(infection)

    health_state = health(state, forward_id)    
    @assert Healthy != health_state
    
    if !track_household_connections && (HouseholdContact == contact_kind)
      continue
    end

    if isdetected(state, forward_id)
      # the check for detected should be enough for avoiding loops and multiple checks of the same person
      # - in case there are multiple backtracing processes at the same time it does not affect the chances of being tested positively
      # - here we are looking for the contacts and all the contacts  already are or were infected
      # - only true infections are stored hence no one will be forward tracked twice anyway
      # it is just the chance of being detected which becomes larger if multiple backtrackings are leading to the same person
      continue
    end
    
    @assert SevereSymptoms != health_state && CriticalSymptoms != health_state && Dead != health_state "the patient should have already been detected at the hospital"
    
    # the probability that the contact is NOT tracked
    if rand(state.rng) >= params.forward_tracking_prob 
      continue
    end
    # if it is found
    push!(state.queue, BackTrackedEvent(current_time + params.forward_detection_delay, forward_id))
  end
end






