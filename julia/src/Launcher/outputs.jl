abstract type Output end

beforetrajectories(::Output, ::Simulation.SimParams) = nothing
pushtrajectory!(::Output, ::Simulation.SimState, ::Simulation.SimParams, ::DetectionCallback) = nothing
aftertrajectories(::Output, ::Simulation.SimParams) = nothing

include("outputs/daily_trajectories.jl")
include("outputs/params_dump.jl")
include("outputs/run_dump.jl")
include("outputs/summary.jl")

const cmd_to_output = Dict{String,Type}(
  "output-summary" => Summary,
  "output-daily" => DailyTrajectories,
  "output-params-dump" => ParamsDump,
  "output-run-dump-prefix" => RunDump
)

function make_outputs(cmd_args::Dict{String,Any}, num_trajectories::Integer)
  outputs = Output[]
  for (opt, type) in cmd_to_output
    if cmd_args[opt] |> !isnothing
      push!(outputs, type(cmd_args[opt], num_trajectories))
    end
  end
  outputs
end