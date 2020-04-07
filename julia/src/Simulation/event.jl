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
  
  Event(::Val{E}, time::Real, subject::Integer) where E = new(time, subject, 0, E, UnknownContact)
  Event(::Val{OutsideInfectionEvent}, time::Real, subject::Integer) = new(time, subject, 0, OutsideInfectionEvent, OutsideContact)
  Event(::Val{TransmissionEvent}, time::Real, subject::Integer) = error("source and contact kind needed for transmission event")
  Event(::Val{TransmissionEvent}, time::Real, subject::Integer, source::Integer, contact_kind::ContactKind) = new(time, subject, source, TransmissionEvent, contact_kind)
end

time(event::Event) = event.time
subject(event::Event) = event.subject_id
source(event::Event) = event.source_id
kind(event::Event) = event.event_kind
contactkind(event::Event) = event.contact_kind

