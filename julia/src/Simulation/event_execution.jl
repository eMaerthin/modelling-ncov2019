
#
# Dispatch to the right event handler
#

function execute!(state::SimState, params::SimParams, event::Event)::Bool 
  kind::EventKind = event.event_kind
  
  if     OutsideInfectionEvent==kind;           execute!(Val(OutsideInfectionEvent), state, params, event) 
  elseif TransmissionEvent==kind;               execute!(Val(TransmissionEvent), state, params, event)
  elseif BecomeInfectiousEvent==kind;           execute!(Val(BecomeInfectiousEvent), state, params, event)
  elseif MildSymptomsEvent==kind;               execute!(Val(MildSymptomsEvent), state, params, event)
  elseif SevereSymptomsEvent==kind;             execute!(Val(SevereSymptomsEvent), state, params, event)
  elseif RecoveryEvent==kind;                   execute!(Val(RecoveryEvent), state, params, event)
  elseif DeathEvent==kind;                      execute!(Val(DeathEvent), state, params, event)
  elseif HomeTreatmentEvent==kind;              execute!(Val(HomeTreatmentEvent), state, params, event)
  elseif HomeTreatmentSuccessEvent==kind;       execute!(Val(HomeTreatmentSuccessEvent), state, params, event)
  elseif GoHospitalEvent==kind;                 execute!(Val(GoHospitalEvent), state, params, event)
  elseif ReleasedEvent==kind;                   execute!(Val(ReleasedEvent), state, params, event)
  elseif DetectedOutsideQuarantineEvent==kind;  execute!(Val(DetectedOutsideQuarantineEvent), state, params, event)
  elseif DetectedFromQuarantineEvent==kind;     execute!(Val(DetectedFromQuarantineEvent), state, params, event)
  elseif BackTrackedEvent==kind;                execute!(Val(BackTrackedEvent), state, params, event)
  elseif QuarantinedEvent==kind;                execute!(Val(QuarantinedEvent), state, params, event)
  elseif QuarantineEndEvent==kind;              execute!(Val(QuarantineEndEvent), state, params, event)
  else error("unsupported event kind $kind")
  end
  return true
end


#
# transmissions
#
function execute!(::Val{OutsideInfectionEvent}, state::SimState, params::SimParams, event::Event)::Bool
  if Healthy != subjecthealth(state, event)
    return false
  end
        
  setsubjecthealth!(state, event, Incubating)  
    
  registerinfection!(state, event) 
  incubation_time = progressionof(params, subject(event)).incubation_time 
  
  event = Event(Val(BecomeInfectiousEvent), time(event) + incubation_time, subject(event))
  push!(state.queue, event)
  return true
end

function execute!(::Val{TransmissionEvent}, state::SimState, params::SimParams, event::Event)::Bool
  if Healthy != subjecthealth(state, event) # apparently the subject became already infected in some other way
    return false
  end
  
  source_health = sourcehealth(state, event)
  @assert source_health != Healthy && source_health != Recovered && source_health != Incubating
  @assert source_health != SevereSymptoms && source_health != CriticalSymptoms && source_health != Dead && source_health != Recovered "infection time exceeds infectability time frame, subject is now in state $source_health, the event is $event source progressions are $(progressionof(params, source(event)))"
    
  # the transmission events are queued in advace, therefore it might be the case that it can not be realized
  # for the transmission to happen both source and subject must be free or both must be staying at home in case
  # the infection takes place inside household

  subject_freedom = subjectfreedom(state, event)
  source_freedom = sourcefreedom(state, event)  

  @assert subject_freedom != HomeTreatment && subject_freedom != Hospitalized "a healthy subject should not be in HomeTreatment or Hospital"
  @assert source_freedom != Hospitalized && source_freedom != Released

  # household contact conditions  
  # if it is an infection inside household and either source or subject are closed in home the event can not happen
  if (contactkind(event) != HouseholdContact) && ( (HomeTreatment == source_freedom) || (HomeQuarantine == source_freedom) || (HomeQuarantine == subject_freedom) )
    return false
  end
  
  setsubjecthealth!(state, event, Incubating)
      
  registerinfection!(state, event)
  
  incubation_time = progressionof(params, subject(event)).incubation_time
  push!(state.queue, 
    Event(Val(BecomeInfectiousEvent), time(event) + incubation_time, subject(event))
  )
  return true
end

#
# Disease progression
#

function execute!(::Val{BecomeInfectiousEvent}, state::SimState, params::SimParams, event::Event)::Bool
  @assert Incubating == subjecthealth(state, event)
  subject_id = subject(event)
    
  sethealth!(state, subject_id, Infectious)
        

  progression = params.progressions[subject_id]
 
    
  severity = progression.severity
  
  infected_time = time(event) - progression.incubation_time 
  if Asymptomatic == severity
    @assert !isnan(progression.recovery_time)
    push!(state.quque, Event(Val(RecoveredEvent), infected_time + progression.recovery_time), subject_id)   
  elseif Mild == severity
    @assert !isnan(progression.mild_symptoms_time)
    push!(state.queue, Event(Val(MildSymptomsEvent), infected_time + progression.mild_symptoms_time, subject_id))
  elseif Severe == severity || Critical == severity # treat all critical as if they were Severe cases
    if !isnan(progression.mild_symptoms_time)
      push!(state.queue, Event(Val(MildSymptomsEvent), infected_time + progression.mild_symptoms_time, subject_id))
    else
      push!(state.queue, Event(Val(SevereSymptomsEvent), infected_time + progression.severe_symptoms_time, subject_id))
    end
  else
    @error "Unsupported severity $severity"
  end

  enqueue_transmissions!(state, Val{ConstantKernelContact}, event.subject_id, params)
  enqueue_transmissions!(state, Val{HouseholdContact}, event.subject_id, params)
  
  if UnderObservation == detected(state, subject_id)
    if isquarantined(state, subject_id)
      push!(state.queue, Event(Val(DetectedFromQuarantineEvent), time(event), subject_id), immediate=true) # immediately  
    else
      push!(state.queue, Event(Val(DetectedOutsideQuarantineEvent), time(event), subject_id), immediate=true) # immediately
    end
  end
  return true
end

function execute!(::Val{MildSymptomsEvent}, state::SimState, params::SimParams, event::Event)::Bool
  @assert Infectious == subjecthealth(state, event)
  setsubjecthealth!(state, event, MildSymptoms)
  subject_id = subject(event)
  
  progression = progressionof(params, subject_id)
  @assert !isnan(progression.mild_symptoms_time)
  
  infection_time = time(event) - progression.mild_symptoms_time

  if Severe == progression.severity || Critical == progression.severity
    @assert !isnan(progression.severe_symptoms_time)
    @assert infection_time + progression.severe_symptoms_time > time(event)
    push!(state.queue, Event(Val(SevereSymptomsEvent), infection_time + progression.severe_symptoms_time, subject_id))
  else
    @assert infection_time + progression.recovery_time > time(event) "next event time $(infection_time + progression.recovery_time) is than current event $event"
    @assert (Mild == progression.severity) "unexpected severity $(progression.severity)"
    push!(state.queue, Event(Val(RecoveryEvent), infection_time + progression.recovery_time, subject_id))
  end

  push!(state.queue, Event(Val(HomeTreatmentEvent), time(event), subject_id), immediate=true) #immediately
  return true
end

function execute!(::Val{SevereSymptomsEvent}, state::SimState, params::SimParams, event::Event)::Bool
  @assert Infectious == subjecthealth(state, event) || MildSymptoms == subjecthealth(state, event)
  setsubjecthealth!(state, event, SevereSymptoms)
  
  subject_id = subject(event)

  progression = progressionof(params, subject_id)

  #push!(state.queue, RecoveryEvent(time(event)+14, subject(event)))
  event = Event(Val(GoHospitalEvent), time(event), subject(event))
  push!(state.queue, event, immediate=true)  # immediately
  
  return true
end

function execute!(::Val{RecoveryEvent}, state::SimState, params::SimParams, event::Event)::Bool
  @assert Recovered !== subjecthealth(state, event)
  @assert Dead !== subjecthealth(state, event)
  setsubjecthealth!(state, event, Recovered)
  freedom_status = subjectfreedom(state, event)
  if Hospitalized == freedom_status
    push!(state.queue, Event(Val(ReleasedEvent), time(event), subject(event)))
  elseif HomeTreatment == freedom_status 
    push!(state.queue, Event(Val(HomeTreatmentSuccessEvent), time(event), subject(event)))
  end
  return true
end

function execute!(::Val{DeathEvent}, state::SimState, params::SimParams, event::Event)::Bool
  @assert Recovered !== subjecthealth(state, event)
  @assert Dead !== subjecthealth(state, event)
  setsubjecthealth!(state, event, Dead)
  push!(state.queue, Event(Val(ReleasedEvent), time(event), subject(event)))
  return true
end

#
# freedom events
#

function execute!(::Val{HomeTreatmentEvent}, state::SimState, params::SimParams, event::Event)::Bool
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

function execute!(::Val{HomeTreatmentSuccessEvent}, state::SimState, params::SimParams, event::Event)::Bool
  @assert Recovered == subjecthealth(state, event)
  @assert HomeTreatment == subjectfreedom(state, event) "subject $(subject(event)) was not in HomeTreatment but in $(subjectfreedom(state, event))"
  if isquarantined(state, subject(event))
    setfreedom!(state, subject(event), HomeQuarantine)
  else
    setfreedom!(state, subject(event), Free)
  end
  return true
end

function execute!(::Val{GoHospitalEvent}, state::SimState, params::SimParams, event::Event)::Bool
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
  if is_from_quarantine
    push!(state.queue, Event(Val(DetectedFromQuarantineEvent), time(event), subject_id), immediate=true) # immediately
  else
    push!(state.queue, Event(Val(DetectedOutsideQuarantineEvent), time(event), subject_id), immediate=true) # immediately
  end
  return true
end

function execute!(::Val{ReleasedEvent}, state::SimState, params::SimParams, event::Event)::Bool
  @assert Recovered == subjecthealth(state, event)
  setfreedom!(state, subject(event), Released)
  return true
end

#
# Detection events
#

function execute!(::Val{DetectedOutsideQuarantineEvent}, state::SimState, params::SimParams, event::Event)::Bool
  subject_id = subject(event)
  if isdetected(state, subject_id)
    return false
  end
  setdetected!(state, subject_id, Detected)
  quarantinehousehold!(state, params, subject_id, include_subject=true)
  trackhousehold!(state, params, subject_id, track_household_connections=true)
  return true
end

function execute!(::Val{DetectedFromQuarantineEvent}, state::SimState, params::SimParams, event::Event)::Bool
  subject_id = subject(event)
  if isdetected(state, subject_id)
    return false
  end
  setdetected!(state, subject_id, Detected)
  quarantinehousehold!(state, params, subject_id, include_subject=true) #make the quarantine longer
  return true
end

function execute!(::Val{BackTrackedEvent}, state::SimState, params::SimParams, event::Event)::Bool  
  quarantinehousehold!(state, params, subject(event), include_subject=true)
  
  for member in householdof(params, subject(event))
    if Undetected != detected(state, member) && UnderObservation != detected(state, member)
      @assert TestPending == detected(state, member) || Detected == detected(state, member)
      continue
    end
    setdetected!(state, member, TestPending)
    
    member_health = health(state, member)
    @assert SevereSymptoms != member_health && CriticalSymptoms != member_health && Dead != member_health "patient should have already been infected at the hospital"
    
    if Infectious == member_health || MildSymptoms == member_health
      push!(state.queue, Event(Val(DetectedFromQuarantineEvent), time(event) + params.testing_time, member))
    end
  end
  return true
end

function execute!(::Val{QuarantinedEvent}, state::SimState, params::SimParams, event::Event)::Bool
  subject_id = subject(event)
  freedom_state = freedom(state, subject_id)

  is_already_quarantined = isquarantined(state, subject_id)
  
  if Hospitalized == freedom_state || Released == freedom_state
    @assert !is_already_quarantined "the quarantine should be lifted just before hospitalization"
    return false
  end
  
  if event.extension 
    @assert Undetected != detected(state, subject_id)
    setfreedom!(state, subject_id, HomeQuarantine)
  elseif Free == freedom_state
    @assert !is_already_quarantined "quarantined cannot be free, but $subject_id is"
    setfreedom!(state, subject_id, HomeQuarantine)
    setdetected!(state, subject_id, UnderObservation)
  elseif HomeQuarantine == freedom_state
    @assert is_already_quarantined "person's $subject_id state is quarantine therefore isquarantine should return true"
    @assert Undetected != detected(state, subject_id) "the subject should be at least under observation"
  else 
    @assert HomeTreatment == freedom_state "bad freedom status detected = $freedom_state"
    setdetected!(state, subject_id, UnderObservation)
  end
  
  quarantine_advance!(state, subject_id, +1)  # increase quarantine level
  push!(state.queue, Event(Val(QuarantineEndEvent), time(event)+params.quarantine_length, subject_id))

  return !is_already_quarantined
end

function execute!(::Val{QuarantineEndEvent}, state::SimState, params::SimParams, event::Event)::Bool
  subject_id = subject(event)
  @assert Undetected != detected(state, subject_id) "subject $subject_id must be at least under observation since the quarantine is ending"
  
  subject_freedom = freedom(state, subject_id)
  
  if  (Hospitalized == subject_freedom) || (Released == subject_freedom)
    # false event as quarantine should have been removed before hospitalization
    @assert !isquarantined(state, subject_id) "subject in state $subject_freedom detected in quarantine"
    return false
  end
  
  quarantine_advance!(state, subject_id, -1)  # reduce quarantine level
  if isquarantined(state, subject_id)
    return false  # ignore event, the quarantine should last longer
  end

  subject_health = health(state, subject_id)
  
  if Infectious == subject_health || MildSymptoms == subject_health 
    quarantinehousehold!(state, params, subject_id, include_subject=true, extension=true)
    return false
  end
    
  @assert Healthy == subject_health || Incubating == subject_health || Recovered == subject_health "subject $subject_id must be in hospital hence not quarantined"
  setfreedom!(state, subject_id, Free)
    
  return true
end


#
# Quarantine and backtracking helpers
#

function quarantinehousehold!(state::SimState, params::SimParams, subject_id::Integer; include_subject::Bool, extension::Bool=false)
  for member in householdof(params, subject_id)
    if !include_subject && (member == subject_id)
      continue
    end
    
    member_freedom = freedom(state, member)

    if (Hospitalized == member_freedom) || (Released == member_freedom)
      continue
    end
    @assert (Free == member_freedom) || (HomeTreatment == member_freedom) || (HomeQuarantine == member_freedom)
    push!(state.queue, Event(Val(QuarantinedEvent), state.time, member, extension), immediate=true) #immediately
  end
  nothing
end

function trackhousehold!(state::SimState, params::SimParams, subject_id::Integer; track_household_connections::Bool)
  for member in householdof(params, subject_id)
     backtrack!(state, params, member, track_household_connections=track_household_connections) 
     forwardtrack!(state, params, member, track_household_connections=track_household_connections)
  end
  nothing
end
function backtrack!(state::SimState, params::SimParams, person_id::Integer; track_household_connections::Bool)
  current_time = state.time
  
  backward_id, contact_kind = backwardinfection(state, person_id)
  if NoContact == contact_kind 
    @assert 0 == backward_id
    return
  end
  
  if !track_household_connections && (HouseholdContact == contact_kind)
    return
  end
  
  if isdetected(state, person_id)
    return
  end
     
  if rand(state.rng) >= params.backward_tracking_prob 
    return
  end
     
  push!(state.queue, Event(Val(BackTrackedEvent), current_time + params.backward_detection_delay, backward_id))
  nothing
end

function forwardtrack!(state::SimState, params::SimParams, person_id::Integer; track_household_connections::Bool)
  current_time = state.time
  # handle all outgoing infections
  for infection in forwardinfections(state, person_id)
    
    contact_kind = contactkind(infection)
    @assert OutsideContact != contact_kind && NoContact != contact_kind
      
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
    
    @assert SevereSymptoms != health_state && CriticalSymptoms != health_state && Dead != health_state "the forward $forward_id should have already been detected at the hospital but his health is $health_state"
    freedom_state = freedom(state, forward_id)
    @assert Hospitalized != freedom_state && Released != freedom_state "the forward $forward_id should have already been detected at the hospital but he is $freedom_state"
    
    # the probability that the contact is NOT tracked
    if rand(state.rng) >= params.forward_tracking_prob 
      continue
    end
    # if it is found
    push!(state.queue, Event(Val(BackTrackedEvent), current_time + params.forward_detection_delay, forward_id))
  end
  nothing
end



