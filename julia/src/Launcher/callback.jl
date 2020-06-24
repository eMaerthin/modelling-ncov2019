
struct DetectionCallback
    detection_times::Vector{OptTimePoint}
    detection_types::Vector{UInt8}

    tracking_times::Vector{OptTimePoint}
    tracking_sources::Vector{UInt32}
    tracking_types::Vector{UInt8}
    
    max_num_infected::UInt32
end
DetectionCallback(sz::Integer, max_num_infected::Integer=10^8) = DetectionCallback(
    Vector{OptTimePoint}(missing, sz),
    fill(UInt8(0), sz),
    Vector{OptTimePoint}(missing, sz),
    fill(UInt32(0), sz),
    fill(UInt8(0), sz),
    max_num_infected
)

function saveparams(dict, cb::DetectionCallback, prefix::AbstractString="") 
  dict[prefix*"detection_times"] = optreal2float32.(cb.detection_times)
  dict[prefix*"detection_types"] = cb.detection_types

  dict[prefix*"tracking_times"] = optreal2float32.(cb.tracking_times)
  dict[prefix*"tracking_sources"] = cb.tracking_sources
  dict[prefix*"tracking_types"] = cb.tracking_types
end

function reset!(cb::DetectionCallback)
  fill!(cb.detection_times, missing)
  fill!(cb.detection_types, 0)
  fill!(cb.tracking_sources, 0)
  fill!(cb.tracking_types, 0)
end

function (cb::DetectionCallback)(event::Simulation.Event, state::Simulation.SimState, params::Simulation.SimParams)
  eventkind = Simulation.kind(event)
  contactkind = Simulation.contactkind(event)
  subject = Simulation.subject(event)
  if Simulation.isdetection(eventkind)
    cb.detection_times[subject] = Simulation.time(event)
    cb.detection_types[subject] = Simulation.detectionkind(event) |> UInt8
  elseif Simulation.istracking(eventkind)
    cb.tracking_times[subject] = Simulation.time(event)
    cb.tracking_sources[subject] = Simulation.source(event)
    cb.tracking_types[subject] = Simulation.trackingkind(event) |> UInt8
  end
  return Simulation.numinfected(state.stats) < cb.max_num_infected
end

function save_infections_and_detections(path::AbstractString, simstate::Simulation.SimState, callback::DetectionCallback)
  f = jldopen(path, "w", compress=true)
  try
    Simulation.saveparams(f, simstate)
    saveparams(f, callback)
  finally
    close(f)
  end
  nothing
end