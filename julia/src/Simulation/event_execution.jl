
#
# Update time & ispatch to the right event handler
#
function execute!(state::SimState, params::SimParams, event::Event)::Bool 
  @assert state.time <= time(event)  "time for event $event was smaller than current time $(state.time)"
  state.time = time(event)
  
  event_kind = kind(event)
  was_executed = execute!(event_kind, state, params, event)
  if was_executed
    update!(state.stats, event)
  end
  was_executed
end

function execute!(kind::EventKind, state::SimState, params::SimParams, event::Event)::Bool
  if     OutsideInfectionEvent==kind;           return execute!(Val(OutsideInfectionEvent), state, params, event) 
  elseif TransmissionEvent==kind;               return execute!(Val(TransmissionEvent), state, params, event)
  elseif BecomeInfectiousEvent==kind;           return execute!(Val(BecomeInfectiousEvent), state, params, event)
  elseif MildSymptomsEvent==kind;               return execute!(Val(MildSymptomsEvent), state, params, event)
  elseif SevereSymptomsEvent==kind;             return execute!(Val(SevereSymptomsEvent), state, params, event)
  elseif RecoveryEvent==kind;                   return execute!(Val(RecoveryEvent), state, params, event)
  elseif DeathEvent==kind;                      return execute!(Val(DeathEvent), state, params, event)
  elseif HomeTreatmentEvent==kind;              return execute!(Val(HomeTreatmentEvent), state, params, event)
  elseif HomeTreatmentSuccessEvent==kind;       return execute!(Val(HomeTreatmentSuccessEvent), state, params, event)
  elseif GoHospitalEvent==kind;                 return execute!(Val(GoHospitalEvent), state, params, event)
  elseif ReleasedEvent==kind;                   return execute!(Val(ReleasedEvent), state, params, event)
  elseif DetectionEvent==kind;                  return execute!(Val(DetectionEvent), state, params, event)
  elseif TrackedEvent==kind;                    return execute!(Val(TrackedEvent), state, params, event)
  elseif QuarantinedEvent==kind;                return execute!(Val(QuarantinedEvent), state, params, event)
  elseif QuarantineEndEvent==kind;              return execute!(Val(QuarantineEndEvent), state, params, event)
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
  @assert !ismissing(incubation_time)
  event = Event(Val(BecomeInfectiousEvent), time(event) + incubation_time, subject(event))
  push!(state.queue, event)
  return true
end

function execute!(::Val{TransmissionEvent}, state::SimState, params::SimParams, event::Event)::Bool
  if Healthy != subjecthealth(state, event) # apparently the subject became already infected in some other way
    return false
  end
  
  source_health = sourcehealth(state, event)
  @assert contactkind(event)==HospitalContact || source_health ∉ SA[Healthy, Incubating, SevereSymptoms, CriticalSymptoms, Dead, Recovered] "infection time exceeds infectability time frame, source is now in state $source_health, the event is $event source progressions are $(progressionof(params, source(event)))"
    
  # the transmission events are queued in advace, therefore it might be the case that it can not be realized
  # for the transmission to happen both source and subject must be free or both must be staying at home in case
  # the infection takes place inside household

  subject_freedom = subjectfreedom(state, event)
  source_freedom = sourcefreedom(state, event)  

  @assert subject_freedom ∉ SA[HomeTreatment, Hospitalized] "a healthy subject should not be in HomeTreatment or Hospital"
  @assert contactkind(event) == HospitalContact || source_freedom ∉ SA[Hospitalized, Released]

  # household contact conditions  
  # if it is an infection inside household and either source or subject are closed in home the event can not happen
  if (contactkind(event) != HouseholdContact) && ((HomeTreatment == source_freedom) || (HomeQuarantine == source_freedom) || (HomeQuarantine == subject_freedom))
    return false
  end
  
  @assert contactkind(event) == HospitalContact || (contactkind(event) == HouseholdContact) || (!isquarantined(state, source(event)) && !isquarantined(state, subject(event))) "the event $event should not be executed because subject state is $(state.individuals[subject(event)]) and source state is $(state.individuals[source(event)])"
    
  if params.infection_modulation_function != nothing && !params.infection_modulation_function(state, params, event)::Bool
    return false
  end

  setsubjecthealth!(state, event, Incubating)
      
  registerinfection!(state, event)
  
  incubation_time = progressionof(params, subject(event)).incubation_time
  @assert !ismissing(incubation_time)
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
    @assert !ismissing(progression.recovery_time)
    push!(state.queue, Event(Val(RecoveryEvent), infected_time + progression.recovery_time, subject_id))  
  elseif Mild == severity
    @assert !ismissing(progression.mild_symptoms_time)
    push!(state.queue, Event(Val(MildSymptomsEvent), infected_time + progression.mild_symptoms_time, subject_id))
  elseif Severe == severity || Critical == severity # treat all critical as if they were Severe cases
    if !ismissing(progression.mild_symptoms_time)
      push!(state.queue, Event(Val(MildSymptomsEvent), infected_time + progression.mild_symptoms_time, subject_id))
    else
      push!(state.queue, Event(Val(SevereSymptomsEvent), infected_time + progression.severe_symptoms_time, subject_id))
    end
  else
    @error "Unsupported severity $severity"
  end

  enqueue_transmissions!(state, Val{ConstantKernelContact}, event.subject_id, params)
  enqueue_transmissions!(state, Val{HouseholdContact}, event.subject_id, params)
  enqueue_transmissions!(state, Val{FriendshipContact}, event.subject_id, params)
  # hospital transmissions are enqueued in GoHospitalEvent

  
  detectioncheck!(state, params, subject_id)

  if ishealthcare(params, subject_id) && rand(state.rng) < params.hospital_kernel_params.healthcare_detection_prob
    delay = params.hospital_kernel_params.healthcare_detection_delay
    push!(state.queue, Event(Val(DetectionEvent), time(event)+delay, subject_id, OutsideQuarantineDetction))
  elseif(rand(state.rng) < params.mild_detection_prob)
    push!(state.queue, Event(Val(DetectionEvent), time(event)+2, subject_id, OutsideQuarantineDetction))
  end
  return true  
end

function execute!(::Val{MildSymptomsEvent}, state::SimState, params::SimParams, event::Event)::Bool
  @assert Infectious == subjecthealth(state, event)
  setsubjecthealth!(state, event, MildSymptoms)
  subject_id = subject(event)
  
  progression = progressionof(params, subject_id)
  @assert !ismissing(progression.mild_symptoms_time)
  
  infection_time = time(event) - progression.mild_symptoms_time

  if Severe == progression.severity || Critical == progression.severity
    @assert !ismissing(progression.severe_symptoms_time)
    @assert infection_time + progression.severe_symptoms_time >= time(event) "$(infection_time + progression.severe_symptoms_time) !>= $(time(event))"
    push!(state.queue, Event(Val(SevereSymptomsEvent), infection_time + progression.severe_symptoms_time, subject_id))
  else
    @assert infection_time + progression.recovery_time > time(event) "next event time $(infection_time + progression.recovery_time) is than current event $event"
    @assert (Mild == progression.severity) "unexpected severity $(progression.severity)"
    push!(state.queue, Event(Val(RecoveryEvent), infection_time + progression.recovery_time, subject_id))
  end

  push!(state.queue, Event(Val(HomeTreatmentEvent), time(event), subject_id), immediate=true) #immediately
  
  detectioncheck!(state, params, subject_id)
  
  return true
end

function execute!(::Val{SevereSymptomsEvent}, state::SimState, params::SimParams, event::Event)::Bool
  @assert subjecthealth(state, event) in SA[Infectious, MildSymptoms]
  setsubjecthealth!(state, event, SevereSymptoms)
  
  subject_id = subject(event)

  progression = progressionof(params, subject_id)

  #push!(state.queue, RecoveryEvent(time(event)+14, subject(event)))
  event = Event(Val(GoHospitalEvent), time(event), subject(event))
  push!(state.queue, event, immediate=true)  # immediately
  
  return true
end

function execute!(::Val{RecoveryEvent}, state::SimState, params::SimParams, event::Event)::Bool
  @assert subjecthealth(state, event) ∉ SA[Recovered, Dead]
  
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
  @assert subjecthealth(state, event) ∉ SA[Recovered, Dead]
  
  setsubjecthealth!(state, event, Dead)
  push!(state.queue, Event(Val(ReleasedEvent), time(event), subject(event)))
  return true
end

#
# freedom events
#

function execute!(::Val{HomeTreatmentEvent}, state::SimState, params::SimParams, event::Event)::Bool
  @assert MildSymptoms == subjecthealth(state, event)  "subject $(subject(event)) does not have mild symptoms, its state is $(state.individuals[subject(event)]) progression is $(progressionof(params,subject(event)))"
  freedom = subjectfreedom(state, event)
  @assert freedom ∉ SA[Hospitalized, HomeTreatment]
  
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
  @assert severity in SA[Severe, Critical]
  subject_id = subject(event)
  
  is_from_quarantine = isquarantined(state, subject_id) 
  
  if is_from_quarantine  
    quarantine_cancel!(state, subject_id)
  end
  setfreedom!(state, subject_id, Hospitalized)
  
  enqueue_transmissions!(state, Val{HospitalContact}, event.subject_id, params)
  
  # all hospitalized cases are detected
  if !params.hospital_detections
    return true
  end
  
  if is_from_quarantine
    push!(state.queue, Event(Val(DetectionEvent), time(event), subject_id, FromQuarantineDetection), immediate=true) # immediately
  else
    push!(state.queue, Event(Val(DetectionEvent), time(event), subject_id, OutsideQuarantineDetction), immediate=true) # immediately
  end
  return true
end

function execute!(::Val{ReleasedEvent}, state::SimState, params::SimParams, event::Event)::Bool
  @assert subjecthealth(state, event) in SA[Dead, Recovered]
  setfreedom!(state, subject(event), Released)
  return true
end

#
# Detection events
#
function execute!(::Val{DetectionEvent}, state::SimState, params::SimParams, event::Event)::Bool
  subject_id = subject(event)
  detection_kind = detectionkind(event)
  if isdetected(state, subject_id)
    return false
  end
  setdetected!(state, subject_id, Detected)
  quarantinehousehold!(state, params, subject_id, include_subject=true)
  if FromQuarantineDetection !== detection_kind
    trackhousehold!(state, params, subject_id, track_household_connections=true)
  end
  return true
end

function execute!(::Val{TrackedEvent}, state::SimState, params::SimParams, event::Event)::Bool  
  quarantinehousehold!(state, params, subject(event), include_subject=true)
  
  for member in householdof(params, subject(event))
    if Undetected != detected(state, member) && UnderObservation != detected(state, member)
      @assert detected(state, member) in SA[TestPending, Detected]
      continue
    end
    
    if member == source(event) # avoid detection loops
      #@assert Detected == detected(state, member) "member was in state $(state.individual[member])"
      continue
    end
    
    setdetected!(state, member, TestPending)
    
    member_health = health(state, member)
    @assert member_health ∉ SA[SevereSymptoms, CriticalSymptoms, Dead] "patient should have already been infected at the hospital"
    
    if Infectious == member_health || MildSymptoms == member_health
      push!(state.queue, Event(Val(DetectionEvent), time(event) + params.testing_time, member, FromTrackingDetection))
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
    #setdetected!(state, subject_id, UnderObservation)
  elseif HomeQuarantine == freedom_state
    @assert is_already_quarantined "person's $subject_id state is quarantine therefore isquarantine should return true"
    @assert Undetected != detected(state, subject_id) "the subject should be at least under observation"
  else 
    @assert HomeTreatment == freedom_state "bad freedom status detected = $freedom_state"
  end

  detection_state = detected(state, subject_id)
  if Undetected == detection_state
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
    
  @assert subject_health in SA[Healthy, Incubating, Recovered] "subject $subject_id must be in hospital hence not quarantined state = $(individualstate(state, subject_id))"
  setfreedom!(state, subject_id, Free)
    
  return true
end

#
# Quarantine and backtracking helpers
#

function detectioncheck!(state::SimState, params::SimParams, person_id::Integer)
  now = state.time
  if Undetected != detected(state, person_id) 
    if isquarantined(state, person_id)
      push!(state.queue, Event(Val(DetectionEvent), now, person_id, FromQuarantineDetection), immediate=true) # immediately  
    else
      push!(state.queue, Event(Val(DetectionEvent), now, person_id, OutsideQuarantineDetction), immediate=true) # immediately
    end
  end 
end

function quarantinehousehold!(state::SimState, params::SimParams, subject_id::Integer; include_subject::Bool, extension::Bool=false)
  for member in householdof(params, subject_id)
    if !include_subject && (member == subject_id)
      continue
    end
    
    member_freedom = freedom(state, member)

    if (Hospitalized == member_freedom) || (Released == member_freedom)
      continue
    end
    @assert member_freedom in SA[Free, HomeTreatment, HomeQuarantine]
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
  current_time = time(state)
  
  event = backwardinfection(state, person_id)
  backward_id = source(event)
  
  if NoContact == contactkind(event) || OutsideContact == contactkind(event)
    @assert 0 == backward_id
    return
  end
  
  if !track_household_connections && (HouseholdContact == contactkind(event))
    return
  end
  
  @assert 0 != backward_id "backward_id=$backward_id infection=$event"
  if isdetected(state, backward_id)
    return
  end
     

  if uses_phone_tracking(params, person_id) && 
      uses_phone_tracking(params, backward_id) && 
      rand(state.rng) < params.phone_tracking_params.detection_prob
    push!(state.queue, Event(Val(TrackedEvent), current_time + params.phone_tracking_params.detection_delay, backward_id, person_id))
  elseif rand(state.rng) < params.backward_tracking_prob 
    push!(state.queue, Event(Val(TrackedEvent), current_time + params.backward_detection_delay, backward_id, person_id))
  end
  nothing
end

function forwardtrack!(state::SimState, params::SimParams, person_id::Integer; track_household_connections::Bool)
  current_time = time(state)

  # handle all outgoing infections
  for infection in forwardinfections(state, person_id)
    
    contact_kind = contactkind(infection)
    @assert contact_kind ∉ SA[OutsideContact, NoContact]
      
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
    
    # in very rare circumstances these asserts are actually not true - if the backtracking comes in exactly between hospitalization and detection
#    @assert health_state ∉ SA[SevereSymptoms, CriticalSymptoms, Dead] "the forward $forward_id should have already been detected at the hospital but his health is $health_state"
    freedom_state = freedom(state, forward_id)
#    @assert freedom_state ∉ SA[Hospitalized, Released] "the forward $forward_id should have already been detected at the hospital but he is $freedom_state"

    if uses_phone_tracking(params, person_id) && 
        uses_phone_tracking(params, forward_id) && 
        rand(state.rng) < params.phone_tracking_params.detection_prob
      push!(state.queue, Event(Val(TrackedEvent), current_time + params.phone_tracking_params.detection_delay, forward_id, person_id))
    elseif rand(state.rng) < params.forward_tracking_prob 
      push!(state.queue, Event(Val(TrackedEvent), current_time + params.forward_detection_delay, forward_id, person_id))
    end

  end
  nothing
end



