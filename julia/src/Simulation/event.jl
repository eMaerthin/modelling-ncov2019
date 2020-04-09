@enum EventKind::UInt8 begin
  OutsideInfectionEvent 
  TransmissionEvent 
  BecomeInfectiousEvent 
  MildSymptomsEvent 
  SevereSymptomsEvent 
  CriticalSymptomsEvent 
  DeathEvent
  RecoveryEvent
  QuarantinedEvent
  QuarantineEndEvent
  HomeTreatmentEvent
  HomeTreatmentSuccessEvent
  GoHospitalEvent
  DetectedOutsideQuarantineEvent
  DetectedFromQuarantineEvent
  BackTrackedEvent
  ReleasedEvent
end

struct Event
  time::Float32
  subject_id::UInt32
  source_id::UInt32
  event_kind::EventKind
  contact_kind::ContactKind
  extension::Bool
  
  Event(::Val{E}, time::Real, subject::Integer) where E = new(time, subject, 0, E, NoContact, false)
  Event(::Val{OutsideInfectionEvent}, time::Real, subject::Integer) = new(time, subject, 0, OutsideInfectionEvent, OutsideContact, false)
  Event(::Val{TransmissionEvent}, time::Real, subject::Integer) = error("source and contact kind needed for transmission event")
  Event(::Val{TransmissionEvent}, time::Real, subject::Integer, source::Integer, contact_kind::ContactKind) = new(time, subject, source, TransmissionEvent, contact_kind, false)
  Event(::Val{QuarantinedEvent}, time::Real, subject::Integer, extension::Bool) = new(time, subject, 0, QuarantinedEvent, NoContact, extension)
  
end

time(event::Event) = event.time
subject(event::Event) = event.subject_id
source(event::Event) = event.source_id
kind(event::Event) = event.event_kind
contactkind(event::Event) = event.contact_kind

import Base.show

function show(io::IO, event::Event)
  print(io, time(event), ":", kind(event), " ", subject(event), " ")
  if TransmissionEvent == kind(event) || OutsideContact == kind(event)
    print(io, "<= ", source(event), " ", contactkind(event))
  elseif QuarantinedEvent == kind(event)
    print(io, "extension=", event.extension)
  end
end