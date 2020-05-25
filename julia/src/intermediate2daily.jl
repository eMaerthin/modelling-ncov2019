using JLD2
using FileIO
using ProgressMeter
using StatsBase

trajectory(times::AbstractVector{T} where T) = filter(isfinite, times) |> sort!
daily(times::AbstractArray{T} where T, max_days::Integer) = fit(Histogram, times, 0:(max_days) ).weights
daily(times::AbstractVector{T} where T) = daily(times, maximum(times))
const daily_trajectory = daily ∘ trajectory

if length(ARGS) < 1
  error("give path to the grid tiles and number of trajectoirs(optional)")
end

output_path = ARGS[1]
num_paths = length(ARGS) == 2 ? parse(Int, ARGS[2]) : 100

output_file = jldopen(output_path*"/daily_trajectories.jld2", "w", compress=true)
try
  hospitalization_progressions, death_progressions = load(output_path*"/run_params.jld2", 
    "progressions/severe_symptoms_times",
    "progressions/death_times")

  @showprogress for i in 1:num_paths
    run_path = output_path*"/run_$i.jld2"

    if !isfile(run_path)
      @warn "could not load run file $run_path, stopping"
      break
    end

    try
      infection_times, detection_times = load(run_path, "infection_times", "detection_times")

      max_infection_time = filter(!isnan, infection_times) |> maximum
      max_detection_time = filter(!isnan, detection_times) |> maximum
      max_days = min(max_infection_time, max_detection_time) |> floor |> Int
      
      trajectory_group = JLD2.Group(output_file, string(i))
      
      trajectory_group["daily_infections"] = daily( infection_times |> trajectory, max_days)
      trajectory_group["daily_detections"] = daily( detection_times |> trajectory, max_days)
      trajectory_group["daily_deaths"] = daily(infection_times.+death_progressions |> trajectory, max_days)
      trajectory_group["daily_hospitalizations"] = daily(infection_times.+hospitalization_progressions |> trajectory, max_days)
  
    catch ex
      @error "problem processing file " run_path ex
    end

  end
catch ex
  @error ex
finally
  close(output_file)
end