


# the mutable part of the simulation

struct IndividualState #TODO change to immutable
  health::HealthState
  freedom::FreedomState
  detected::DetectionStatus
  quarantine_level::SafeUInt8
end

IndividualState() = IndividualState(
  Healthy,
  Free,
  Undetected,
  0
)

show(io::IO, s::IndividualState) = print(io, "(",s.health, ", ", s.freedom, ", ", s.detected, ", ", s.quarantine_level, ")")


mutable struct SimState
  rng::MersenneTwister

  time::TimePoint
  
  queue::EventQueue
  
  individuals::Vector{IndividualState}  
  
  forest::InfectionForest
  
  #infections::Vector{Vector{Event}}    
  #infections::SortedMultiDict{UInt32,Event}
  
  #infection_sources::Vector{Tuple{UInt32, ContactKind}}
  
    
  #num_dead::Int
  #num_affected::Int
  #num_detected::Int
    
  # buffers for rand
  sample_id_buf::Vector{UInt32}
  sample_time_buf::Vector{TimePoint}
    
    
  SimState(rng::AbstractRNG, num_individuals::Integer) = num_individuals <= 0 || num_individuals > typemax(UInt32) ? error("number of individuals must be positive and smaller than $(typemax(UInt32))") : 
    new(
      rng,
      
      0.0,
      EventQueue(),
      
      fill(IndividualState(), num_individuals),
      
      InfectionForest(num_individuals),
      
      # just the initial size, will be resized to meet the needs
      Vector{UInt32}(undef, 100),
      Vector{TimePoint}(undef, 100) 
    ) 
end

SimState(num_individuals::Integer; seed::Integer=0) = SimState(MersenneTwister(seed), num_individuals)

function reset!(state::SimState, rng::AbstractRNG)
  if isa(rng, AbstractRNG)
    state.rng = rng
  end
  state.time=0
  empty!(state.queue)
  reset!(state.forest)
  fill!(state.individuals, IndividualState())
  state
end

time(state) = state.time

reset!(state::SimState) = reset!(state::SimState, state.rng)
reset!(state::SimState, seed::Integer) = reset!(state, MersenneTwister(seed))

individualstate(state::SimState, person_id::Integer) = state.individuals[person_id]
health(state::SimState, person_id::Integer)::HealthState = state.individuals[person_id].health
freedom(state::SimState, person_id::Integer)::FreedomState = state.individuals[person_id].freedom

quarantine_level(state::SimState, person_id::Integer) = state.individuals[person_id].quarantine_level
isquarantined(state::SimState, person_id::Integer)::Bool = quarantine_level(state, person_id) != 0
detected(state::SimState, person_id::Integer)::DetectionStatus = state.individuals[person_id].detected 
isdetected(state::SimState, person_id::Integer)::Bool = (Detected == detected(state, person_id))

subjecthealth(state::SimState, event::Event)::HealthState = health(state, subject(event))
subjectfreedom(state::SimState, event::Event)::FreedomState = freedom(state, subject(event))

sourcehealth(state::SimState, event::Event)::HealthState = health(state, source(event))
sourcefreedom(state::SimState, event::Event)::FreedomState = freedom(state, source(event))

#forwardinfections(state::SimState, person_id::Integer) = inclusive(state.infections, searchequalrange(state.infections, person_id)...) |> values
forwardinfections(state::SimState, person_id::Integer)::Vector{Event} = forwardinfections(state.forest, person_id)
backwardinfection(state::SimState, person_id::Integer)::Event = backwardinfection(state.forest, person_id)


function sethealth!(state::SimState, person_id::Integer, new_health::HealthState)
  orig = state.individuals[person_id]
  @assert orig.health <= new_health
  state.individuals[person_id] = @set orig.health = new_health
  nothing
end

function setfreedom!(state::SimState, person_id::Integer, new_freedom::FreedomState)
  orig = state.individuals[person_id]
  state.individuals[person_id] = @set orig.freedom = new_freedom
  nothing
end

function setdetected!(state::SimState, person_id::Integer, new_detected::DetectionStatus)
  orig = state.individuals[person_id]
  @assert orig.detected <= new_detected
  state.individuals[person_id] = @set orig.detected = new_detected
  nothing
end

function quarantine_advance!(state::SimState, person_id::Integer, adv_val::Integer) 
  orig = state.individuals[person_id]
  if adv_val >= 0 
    state.individuals[person_id] = @set orig.quarantine_level = orig.quarantine_level+adv_val
  else
    state.individuals[person_id] = @set orig.quarantine_level = orig.quarantine_level-(-adv_val)    
  end  
  nothing
end

function quarantine_cancel!(state::SimState, person_id::Integer)
  orig = state.individuals[person_id]
  state.individuals[person_id] = @set orig.quarantine_level = 0 
  nothing
end

setsubjecthealth!(state::SimState, event::Event, health::HealthState) = sethealth!(state, subject(event), health)
setsubjectfreedom!(state::SimState, event::Event, freedom::FreedomState) = setfreedom!(state, subject(event), freedom)

registerinfection!(state::SimState, infection::Event) = push!(state.forest, infection)

function initialfeed!(state::SimState, num_initial_infections::Integer)
  @assert 0 == time(state)
  
  N = length(state.individuals)  
  individuals = 1:N
  
  for _ in 1:num_initial_infections
    person_id = sample(state.rng, individuals)
    event = Event(Val(OutsideInfectionEvent), 0.0, person_id)
    push!(state.queue, event)
  end
   
end