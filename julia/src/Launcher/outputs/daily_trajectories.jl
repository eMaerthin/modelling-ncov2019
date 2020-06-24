using StatsBase
using JLD2

daily(times::AbstractArray{T} where T, max_days::Integer) = fit(Histogram, times, 0:(max_days) ).weights
daily(times::AbstractVector{T} where T) = daily(times, maximum(times))

mutable struct DailyTrajectories <: Output
  file::JLD2.JLDFile
  last_trajectory::Int
end

DailyTrajectories(fname::AbstractString, ::Integer) = DailyTrajectories(JLD2.jldopen(fname, "w+", compress=true), 0)

function pushtrajectory!(d::DailyTrajectories, state::Simulation.SimState, params::Simulation.SimParams, cb::DetectionCallback)
  d.last_trajectory += 1
  trajectory_group = JLD2.Group(d.file, string(d.last_trajectory))
  save_daily_trajectories(trajectory_group, state, params, cb)
end

aftertrajectories(d::DailyTrajectories) = close(d.file)

function save_daily_trajectories(dict, state::Simulation.SimState, params::Simulation.SimParams, cb::DetectionCallback)
  max_days = Simulation.time(state) |> floor |> Int
  num_individuals = Simulation.numindividuals(state)
  
  infection_times = Vector{OptTimePoint}(missing, num_individuals)
  contact_kinds = Vector{Simulation.ContactKind}(undef, num_individuals)

  for i in 1:num_individuals
    event = Simulation.backwardinfection(state, i)
    kind = contactkind(event)

    contact_kinds[i] = kind
    infection_times[i] = ifelse(kind == Simulation.NoContact, time(event), missing)
  end
  hospitalization_progressions = getproperty.(params.progressions, :severe_symptoms_time)
  death_progressions = getproperty.(params.progressions, :death_time)

  dict["infection_times"] = daily(filter(!ismissing,infection_times), max_days)
  dict["detection_tiems"] = daily(filter(!ismissing, cb.detection_times), max_days)
  dict["daily_deaths"] = daily(filter(!ismissing, infection_times.+death_progressions), max_days)
  dict["daily_hospitalizations"] = daily(filter(!ismissing, infection_times.+hospitalization_progressions), max_days)
  for kind in instances(ContactKind)
    if kind != NoContact
      dict["daily_" * lowercase(string(kind))] = daily(infection_times[contact_kinds.==Int(kind)], max_days)
    end
  end
end