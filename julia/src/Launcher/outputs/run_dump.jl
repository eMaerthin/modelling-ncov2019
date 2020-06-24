mutable struct RunDump <: Output
  path_prefix::String
  num_trajectories::Int
  RunDump(path_prefix::AbstractString, ::Integer) = new(path_prefix, 0)
end

function pushtrajectory!(d::RunDump, state::Simulation.SimState, ::Simulation.SimParams, callback::DetectionCallback)
  d.num_trajectories += 1
  f = jldopen(d.path_prefix*"_$(d.num_trajectories).jld2", "w", compress=true)
  try
    Simulation.saveparams(f, state)
    saveparams(f, callback)
  finally
    close(f)
  end
  nothing
end