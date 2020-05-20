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
  