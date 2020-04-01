import Base.isless

abstract type AbstractEvent end

abstract type AbstractInfectionEvent <: AbstractEvent end

struct OutsideInfectionEvent <: AbstractInfectionEvent
  time::Float32
  subject_id::Int32
end

struct TransmissionEvent <: AbstractInfectionEvent
  time::Float32
  subject_id::Int32
  source_id::Int32
  kind::ContactKind
end

abstract type DiseaseEvent <: AbstractEvent end

struct BecomeInfectiousEvent <: DiseaseEvent
  time::Float32
  subject_id::Int32
end

struct MildSymptomsEvent <: DiseaseEvent
  time::Float32
  subject_id::Int32
end

struct SevereSymptomsEvent <: DiseaseEvent
  time::Float32
  subject_id::Int32
end

struct CriticalSymptomsEvent <: DiseaseEvent
  time::Float32
  subject_id
end

struct DeathEvent <: DiseaseEvent
  time::Float32
  subject_id::Int32
end

struct RecoveryEvent <: DiseaseEvent
  time::Float32
  subject_id::Int32
end

abstract type FreedomEvent end

struct QuarantinedEvent <: FreedomEvent
  time::Float32
  subject_id::Int32
end

struct StayHomeTreatmentEvent <: FreedomEvent
  time::Float32
  subject_id::Int32
end

#struct StayHomeToHospitalizeEvent <: FreedomEvent
#  time::Float32
#  subject_id::Int32
#end

struct GoHospitalEvent <: FreedomEvent
  time::Float32
  subject_id::Int32
end

struct QuarantineEndEvent <: FreedomEvent
  time::Float32
  subject_id::Int32
end

#
# Other events
#

struct DetectedEvent <: AbstractEvent
  time::Float32
  subject_id::Int32
end

time(event::AbstractEvent) = event.time 



#isless(e1::AbstractEvent, e2::AbstractEvent) = time(e1) < time(e2)
#isless(x::AbstractInfectionEvent, y::AbstractInfectionEvent) = subject(x) < subject(y)
#isless(x::OutsideInfectionEvent, y::AbstractInfectionEvent) = true
#isless(x::AbstractInfectionEvent, y::OutsideInfectionEvent) = false
#isless(x::OutsideInfectionEvent, y::OutsideInfectionEvent) = false



subject(event::AbstractEvent) = event.subject_id
source(event::TransmissionEvent) = event.source_id
source(event::OutsideInfectionEvent) = missing