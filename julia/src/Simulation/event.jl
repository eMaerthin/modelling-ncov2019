@enum EventKind::UInt8 begin 

# the underlying values are also priorites in case the times are equal
# therefore the order of definition implies priority

  QuarantinedEvent
  DetectionEvent
  #DetectedFromQuarantineEvent  
  HomeTreatmentSuccessEvent
  QuarantineEndEvent

  GoHospitalEvent
  
  #DetectedOutsideQuarantineEvent
  #DetectedFromTrackingEvent

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
  time::TimePoint                 # 4 bytes
  subject_id::PersonIdx           # 4 bytes
  source_id::PersonIdx            # 4 bytes
  event_kind::EventKind           # 1 byte
  contact_kind::ContactKind       # 1 byte
  extension::Bool                 # 1 byte
  detection_kind::DetectionKind   # 1 byte
                                  # alignment = 4 bytes
  
  Event() = new(0.0, 0, 0, InvalidEvent, NoContact, false, NoDetection)
  Event(::Val{E}, time::Real, subject::Integer) where E = new(time, subject, 0, E, NoContact, false, NoDetection)
  Event(::Val{OutsideInfectionEvent}, time::Real, subject::Integer) = new(time, subject, 0, OutsideInfectionEvent, OutsideContact, false, NoDetection)
  Event(::Val{TransmissionEvent}, ::Real, ::Integer) = error("source and contact kind needed for transmission event")
  Event(::Val{TransmissionEvent}, time::Real, subject::Integer, source::Integer, contact_kind::ContactKind) = new(time, subject, source, TransmissionEvent, contact_kind, false, NoDetection)
  Event(::Val{QuarantinedEvent}, time::Real, subject::Integer, extension::Bool) = new(time, subject, 0, QuarantinedEvent, NoContact, extension, NoDetection)
  Event(::Val{TrackedEvent}, ::Real, ::Integer) = error("source must be given for TrackedEvent")
  Event(::Val{TrackedEvent}, time::Real, subject::Integer, source::Integer) = new(time, subject, source, TrackedEvent, NoContact, false, NoDetection)
  Event(::Val{DetectionEvent}, ::Real, ::Integer) = error("detection kind must be given for detection event")
  Event(::Val{DetectionEvent}, time::Real, subject::Integer, detectionkind::DetectionKind) = new(time, subject, 0, DetectionEvent, NoContact, false, detectionkind)
end

time(event::Event) = event.time
subject(event::Event) = event.subject_id
source(event::Event) = event.source_id
kind(event::Event) = event.event_kind
contactkind(event::Event) = event.contact_kind
detectionkind(event::Event) = event.detection_kind

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

isdetection(ek::Simulation.EventKind) = ek == DetectionEvent
isdetection(e::Event) = isdetection(kind(e))
istransmission(ek::Simulation.EventKind) = ek == TransmissionEvent || ek == OutsideInfectionEvent
istransmission(e::Event) = istransmission(kind(e))
isquarantine(ek::Simulation.EventKind) = ek == QuarantinedEvent || ek == QuarantineEndEvent
isquarantine(e::Event) = isquarantine(kind(e))
ishospitalization(ek::Simulation.EventKind) = ek == GoHospitalEvent
ishospitalization(e::Event) = ishospitalization(kind(e))
isdeath(ek::Simulation.EventKind) = ek == DeathEvent
isdeath(e::Event) = isdeath(kind(e))