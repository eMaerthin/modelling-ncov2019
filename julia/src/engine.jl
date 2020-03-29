
using DataFrames

abstract type AbstractEvent end
abstract type DiseaseEvent <: AbstractEvent end

struct OutsideInfectionEvent <: AbstractEvent
    time::Float32
    subject_id::Int32
end

struct BecomeInfectiousEvent <: DiseaseEvent
    time::Float32
    subject_id::Int32
end

struct StayHomeEvent <: DiseaseEvent
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

struct RecoverEvent <: DiseaseEvent
    time::Float32
    subject_id::Int32
end

struct TransmissionEvent <: AbstractEvent
    time::Float32
    subject_id::Int32
    source_id::Int32
    kind::ContactKind
end

# the constant part of the simulation
struct SimParams
    people_df::DataFrame       # the sampled population
    household_df::DataFrame    # sampled households
    progression_df::DataFrame  # sampled progression times


    max_affected::Int   # stop criteria
end


mutable struct SimState
    time::Float64
    queue::BinaryHeap{AbstractEvent} # TODO should be changed to union of all events once all events are implemented
    health_states::Vector{HealthState}
    detected::BitVector



    num_dead::Int
    num_affected::Int
    num_detected::Int
end

isactive(state::HealthState) = (state == Infectious) || (state == StayingHome)

time(event::AbstractEvent) = event.time
subject(event::AbstractEvent) = event.subject_id
source(event::TransmissionEvent) = event.source_id

health(state::SimState, person_id::Integer) = state.infection_status[person_id]

subjecthealth(state::SimState, event::AbstractEvent) = health(state, subject(event))
sourcehealth(state::SimState, event::TransmissionEvent) = health(state, source(event))

function execute!(state::SimState, params::SimParams, event::OutsideInfectionEvent)
    @assert Recovered !== subjecthealth(state, event)
    if Healthy == subjecthealth(state, event)
        state.infection_status[event.person_id] = Infected
    end
end

function execute!(state::SimState, params::SimParams, event::TransmissionEvent)
    if Healthy != subjecthealth(state, event) || !isactive(sourcehealth(event))
        return
    end

    if event.kind != HouseholdContact
        if sourcehealth(event) != StayingHome
            add_new_infection()
        end
    else
        add_new_infection()
    end
end

function execute!(state::SimState, params::SimParams, event::BecomeInfectiousEvent)
    @assert Infectious !== subjecthealth(state, event)
    @assert Recovered !== subjecthealth(state, event)
    state.infection_status[event.subject_id] = Infecitous

    add_potential_contacts!(HouseholdContact, state, params, event.subject_id)
    add_potential_contacts!(ConstantKernelContact, state, params, event.subject_id)
end

function execute!(state::SimState, params::SimParams, event::StayHomeEvent)
    if target_status == subjecthealth(state, event)
        state.infection_status[event.subject_id] = StayHome
    end
end

function execute!(state::SimState, params::SimParams, event::GoHospitalEvent)
    person_health = subjecthealth(state, event)
    if person_health == StayingHome || person_health == Infectious
        state.infection_status[event.subject_id] = Hospitalized
    end
end

function execute!(state::SimState, params::SimParams, event::RecoverEvent)
    @assert isactive(subjecthealth(event))

    state.infection_status[event.subject_id] = Recovered
end

function execute!(state::SimState, params::SimParams, event::DeathEvent)
    @assert Recovered !== subjecthealth(event)
    @assert Dead !== subjecthealth(event)

    state.infection_status[event.subject_id] = Dead
end

function simulate!(state, params)
    while true
        if isempty(queue)
            @info "Empty queue"
            break
        end

        event = pop!(state.event_queue)
        if state.affected_people >= params.max_affected
            @info "The outbreak reached a high number $(params.max_affected)"
            break
        else
            event.time >= params.max_time
            @info "Max time reached"
            break
        end

        state.global_time = event.time

        execute!(state, params, event)
    end
end
