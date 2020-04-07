

abstract type AbstractEvent end

abstract type AbstractInfectionEvent <: AbstractEvent end

struct OutsideInfectionEvent <: AbstractInfectionEvent
  time::Float32
  subject_id::UInt32
end

struct TransmissionEvent <: AbstractInfectionEvent
  time::Float32
  subject_id::UInt32
  source_id::UInt32
  kind::ContactKind
end

abstract type DiseaseEvent <: AbstractEvent end

struct BecomeInfectiousEvent <: DiseaseEvent
  time::Float32
  subject_id::UInt32
end

struct MildSymptomsEvent <: DiseaseEvent
  time::Float32
  subject_id::UInt32
end

struct SevereSymptomsEvent <: DiseaseEvent
  time::Float32
  subject_id::UInt32
end

struct CriticalSymptomsEvent <: DiseaseEvent
  time::Float32
  subject_id::UInt32
end

struct DeathEvent <: DiseaseEvent
  time::Float32
  subject_id::UInt32
end

struct RecoveryEvent <: DiseaseEvent
  time::Float32
  subject_id::UInt32
end

abstract type FreedomEvent <: AbstractEvent end

struct QuarantinedEvent <: FreedomEvent
  time::Float32
  subject_id::UInt32
end

struct QuarantineEndEvent <: FreedomEvent
  time::Float32
  subject_id::UInt32
end

struct HomeTreatmentEvent <: FreedomEvent
  time::Float32
  subject_id::UInt32
end


struct GoHospitalEvent <: FreedomEvent
  time::Float32
  subject_id::UInt32
end



#
# Other events
#

struct DetectedEvent <: AbstractEvent
  time::Float32
  subject_id::UInt32
  is_from_quarantine::Bool
end

struct BackTrackedEvent <:AbstractEvent
  time::Float32
  subject_id::UInt32
end

TransmissionUnion = Union{
  OutsideInfectionEvent,
  TransmissionEvent
}

EventUnion = Union{
  OutsideInfectionEvent,
  TransmissionEvent,
  
  BecomeInfectiousEvent,
  MildSymptomsEvent,
  SevereSymptomsEvent,
  CriticalSymptomsEvent,
  DeathEvent,
  RecoveryEvent,
  
  QuarantinedEvent,
  QuarantineEndEvent,
  HomeTreatmentEvent,
  GoHospitalEvent,
  
  DetectedEvent,
  BackTrackedEvent
}

time(event::AbstractEvent)::Float32 = event.time 




subject(event::AbstractEvent)::UInt32 = event.subject_id

source(event::TransmissionEvent)::UInt32 = event.source_id
source(event::OutsideInfectionEvent)::UInt32 = 0

contactkind(event::TransmissionEvent)::ContactKind = event.kind
contactkind(event::OutsideInfectionEvent)::ContactKind = OutsideContact