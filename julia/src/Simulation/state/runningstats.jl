mutable struct RunningStats
  num_infected::Int
  num_detected::Int
  num_dead::Int
  RunningStats() = new(0, 0, 0)
end

numinfected(stats::RunningStats) = stats.num_infected
numdetected(stats::RunningStats) = stats.num_detected
numdead(stats::RunningStats) = stats.num_dead

function  update!(stats::RunningStats, event::Event)
  ek = kind(event)
  if isdetection(ek)
    stats.num_detected += 1
  elseif istransmission(ek)
    stats.num_infected += 1
  elseif isdeath(ek)
    stats.num_dead += 1
  end
  nothing
end
