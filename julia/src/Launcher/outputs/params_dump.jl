struct ParamsDump <: Output
  path::String
end
ParamsDump(path::AbstractString, ::Integer) = ParamsDump(path)

function beforetrajectories(d::ParamsDump, params::Simulation.SimState)
  @info "saving full parameters to $(d.path)"  
  file = jldopen(d.path, "w", compress=true)
  try 
    Simulation.saveparams(file, params)
  finally
    close(file)
  end
end
