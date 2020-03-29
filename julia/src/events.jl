import Base.isless

abstract type AbstractEvent end

abstract type DiseaseEvent <: AbstractEvent end
abstract type AbstractInfectionEvent <: AbstractEvent end

struct BecomeInfectiousEvent <: DiseaseEvent
    time::Float32
    subject_id::Int32
end

struct StayHomeMildEvent <: DiseaseEvent
    time::Float32
    subject_id::Int32
end

struct StayHomeToHospitalizeEvent <: DiseaseEvent
    time::Float32
    subject_id::Int32
end

struct GoHospitalEvent <: DiseaseEvent
    time::Float32
    subject_id::Int32
end

struct DeathEvent <: DiseaseEvent
    time::Float32
    subject_id::Int32
end

struct RecoveryEvent <: DiseaseEvent
    time::Float32
    subject_id::Int32
end

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

time(event::AbstractEvent) = event.time 



#isless(e1::AbstractEvent, e2::AbstractEvent) = time(e1) < time(e2)
#isless(x::AbstractInfectionEvent, y::AbstractInfectionEvent) = subject(x) < subject(y)
#isless(x::OutsideInfectionEvent, y::AbstractInfectionEvent) = true
#isless(x::AbstractInfectionEvent, y::OutsideInfectionEvent) = false
#isless(x::OutsideInfectionEvent, y::OutsideInfectionEvent) = false



subject(event::AbstractEvent) = event.subject_id
source(event::TransmissionEvent) = event.source_id
source(event::OutsideInfectionEvent) = missing