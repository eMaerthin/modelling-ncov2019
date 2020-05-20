struct TanhModulation
  weight_detected::Float64
  weight_deaths::Float64
  loc::Float64
  scale::Float64
  limit_value::Float64
end
TanhModulation(config::Dict{String,Any}) = TanhModulation(
  config["weight_detected"],
  config["weight_deaths"],
  config["loc"],
  config["scale"],
  config["limit_value"])

function (f::TanhModulation)(state::SimState, params::SimParams, event::Event)
  @assert kind(event) == TransmissionEvent

  ck = contactkind(event)
  if ConstantKernelContact !== ck && SporadicContact !== ck && FriendshipContact !== ck
    return true # do not affect other types of contact than "outer" ones
  end

  num_detected = numdetected(state) 
  num_deaths = numdead(state)
  fear = num_detected * f.weight_detected + num_deaths * f.weight_deaths
  x = (fear - f.loc) / f.scale
  scaling = ((1 - f.limit_value) / 2)
  base = (1 - (1 - f.limit_value) / 2)
  rand(state.rng) < tanh(x) * scaling + base 
end