struct Summary <: Output
  filename::String
  last_infections::Vector{Float32}
  num_infections::Vector{UInt32}
  Summary(filename::AbstractString, num_trajectories::Integer) = new(filename, sizehint!(Vector{Float32}(), num_trajectories), sizehint!(Vector{UInt32}(), num_trajectories))
end

function beforetrajectories(s::Summary, params::Simulation.SimParams)
  sz = Simulation.numindividuals(params)
  sizehint!(s.last_infections, sz)
  sizehint!(s.num_infections, sz)
end

function pushtrajectory!(s::Summary, state::Simulation.SimState, ::Simulation.SimParams, ::DetectionCallback)
  push!(s.last_infections, maximum(Simulation.time, state.forest.inedges))
  push!(s.num_infections, count(Simulation.istransmission, state.forest.inedges))
  nothing
end

function aftertrajectories(s::Summary, ::Simulation.SimParams)
  file = jldopen(s.filename, "w")
  try 
    file["num_infections"] = s.num_infections
    file["last_infections"] = s.last_infections
  finally
    close(file)
  end
  nothing
end