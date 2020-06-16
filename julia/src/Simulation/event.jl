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
  extra::UInt8

  #contact_kind::ContactKind       # 1 byte
  #extension::Bool                 # 1 byte
  #detection_kind::DetectionKind   # 1 byte
                                  # alignment = 4 bytes
  
  Event() = new(0.0, 0, 0, InvalidEvent, 0)
  Event(::Val{E}, time::Real, subject::Integer) where E = new(time, subject, 0, E, 0)
  Event(::Val{OutsideInfectionEvent}, time::Real, subject::Integer) = new(time, subject, 0, OutsideInfectionEvent, UInt8(OutsideContact))
  Event(::Val{TransmissionEvent}, ::Real, ::Integer) = error("source and contact kind needed for transmission event")
  Event(::Val{TransmissionEvent}, time::Real, subject::Integer, source::Integer, contact_kind::ContactKind) = new(time, subject, source, TransmissionEvent, UInt8(contact_kind))
  Event(::Val{QuarantinedEvent}, time::Real, subject::Integer, extension::Bool) = new(time, subject, 0, QuarantinedEvent, UInt8(extension))
  Event(::Val{TrackedEvent}, ::Real, ::Integer) = error("source and tracking kind must be given for TrackedEvent")
  Event(::Val{TrackedEvent}, time::Real, subject::Integer, source::Integer, tracking_kind::TrackingKind) = new(time, subject, source, TrackedEvent, UInt8(tracking_kind))
  Event(::Val{DetectionEvent}, ::Real, ::Integer) = error("detection kind must be given for detection event")
  Event(::Val{DetectionEvent}, time::Real, subject::Integer, detectionkind::DetectionKind) = new(time, subject, 0, DetectionEvent, UInt8(detectionkind))
end

time(event::Event) = event.time
subject(event::Event) = event.subject_id
source(event::Event) = event.source_id
kind(event::Event) = event.event_kind

contactkind(event::Event) = istransmission(event) ? ContactKind(event.extra) : NoContact
detectionkind(event::Event) = isdetection(event) ? DetectionKind(event.extra) : NoDetection
extension(event::Event) = isquarantine(event) ? Bool(event.extra) : false
trackingkind(event::Event) = istracking(event) ? TrackingKind(event.extra) : NotTracked

function show(io::IO, event::Event)
  print(io, time(event), ":", kind(event), " ", subject(event))
  if TransmissionEvent == kind(event) || OutsideContact == kind(event)
    print(io, " <= ", source(event), " ", contactkind(event))
  elseif QuarantinedEvent == kind(event)
    print(io, " extension=", extension(event))
  elseif TrackedEvent == kind(event)
    print(io, " <= ", source(event))
  elseif InvalidEvent == kind(event)
    print(io, " <= ", source(event), " ", kind(event), " ", event.extra)
  end
end

isdetection(ek::EventKind) = ek == DetectionEvent
isdetection(e::Event) = isdetection(kind(e))
istransmission(ek::EventKind) = ek == TransmissionEvent || ek == OutsideInfectionEvent
istransmission(e::Event) = istransmission(kind(e))
isquarantine(ek::EventKind) = ek == QuarantinedEvent || ek == QuarantineEndEvent
isquarantine(e::Event) = isquarantine(kind(e))
istracking(ek::EventKind) = ek == TrackedEvent
istracking(e::Event) = istracking(kind(e))
ishospitalization(ek::EventKind) = ek == GoHospitalEvent
ishospitalization(e::Event) = ishospitalization(kind(e))
isdeath(ek::EventKind) = ek == DeathEvent
isdeath(e::Event) = isdeath(kind(e))