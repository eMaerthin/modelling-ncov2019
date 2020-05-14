@enum EventKind::UInt8 begin 

# the underlying values are also priorites in case the time is the same
# therefore the order of definition implies priority

  QuarantinedEvent
  DetectedFromQuarantineEvent  
  HomeTreatmentSuccessEvent
  QuarantineEndEvent

  GoHospitalEvent
  
  DetectedOutsideQuarantineEvent
  DetectedFromTrackingEvent

  TrackedEvent
  ReleasedEvent
  
  # the progression events have low priority to let the immediate actions execute
  BecomeInfectiousEvent 
  
  OutsideInfectionEvent 
  TransmissionEvent 
  
  MildSymptomsEvent 
  HomeTreatmentEvent
  
  SevereSymptomsEvent 
  CriticalSymptomsEvent 
  RecoveryEvent
  DeathEvent
  
  InvalidEvent # should not be executed
end

struct Event
  time::TimePoint
  subject_id::PersonIdx
  source_id::PersonIdx
  event_kind::EventKind
  contact_kind::ContactKind
  extension::Bool
  
  Event() = new(0.0, 0, 0, InvalidEvent, NoContact, false)
  Event(::Val{E}, time::Real, subject::Integer) where E = new(time, subject, 0, E, NoContact, false)
  Event(::Val{OutsideInfectionEvent}, time::Real, subject::Integer) = new(time, subject, 0, OutsideInfectionEvent, OutsideContact, false)
  Event(::Val{TransmissionEvent}, ::Real, ::Integer) = error("source and contact kind needed for transmission event")
  Event(::Val{TransmissionEvent}, time::Real, subject::Integer, source::Integer, contact_kind::ContactKind) = new(time, subject, source, TransmissionEvent, contact_kind, false)
  Event(::Val{QuarantinedEvent}, time::Real, subject::Integer, extension::Bool) = new(time, subject, 0, QuarantinedEvent, NoContact, extension)
  Event(::Val{TrackedEvent}, time::Real, subject::Integer) = error("source must be given for TrackedEvent")
  Event(::Val{TrackedEvent}, time::Real, subject::Integer, source::Integer) = new(time, subject, source, TrackedEvent, NoContact, false)
  
end

time(event::Event) = event.time
subject(event::Event) = event.subject_id
source(event::Event) = event.source_id
kind(event::Event) = event.event_kind
contactkind(event::Event) = event.contact_kind

function show(io::IO, event::Event)
  print(io, time(event), ":", kind(event), " ", subject(event))
  if TransmissionEvent == kind(event) || OutsideContact == kind(event)
    print(io, " <= ", source(event), " ", contactkind(event))
  elseif QuarantinedEvent == kind(event)
    print(io, " extension=", event.extension)
  elseif TrackedEvent == kind(event)
    print(io, " <= ", source(event))
  elseif InvalidEvent == kind(event)
    print(io, " <= ", source(event), " ", kind(event), " ", contactkind(event), " ", event.extension)
  end
end