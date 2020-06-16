@enum Severity::UInt8 Asymptomatic=1 Mild Severe Critical # 2bits

@enum HealthState::UInt8 Healthy Incubating Infectious MildSymptoms SevereSymptoms CriticalSymptoms Recovered Dead # 3 bits

@enum FreedomState::UInt8 Free HomeQuarantine HomeTreatment Hospitalized Released # 3 bits

@enum DetectionStatus::UInt8 Undetected UnderObservation TestPending Detected #2 bits

@enum ContactKind::UInt8 NoContact=0 HouseholdContact HospitalContact FriendshipContact SporadicContact ConstantKernelContact OutsideContact # 3 bits

@enum DetectionKind::UInt8 NoDetection=0 OutsideQuarantineDetction=1 FromQuarantineDetection FromTrackingDetection

@enum TrackingKind::UInt8 begin 
  NotTracked=0 
  MildCaseTracked #unused yet
  HospitalTracked #unused yet
  QuarantineTracked #unused yet
  ClassicalTracked 
  PhoneTracked 
end
