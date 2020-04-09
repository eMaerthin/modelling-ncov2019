@enum Severity::UInt8 Asymptomatic=1 Mild Severe Critical

@enum HealthState::UInt8 Healthy Incubating Infectious MildSymptoms SevereSymptoms CriticalSymptoms Recovered Dead

@enum FreedomState::UInt8 Free HomeQuarantine HomeTreatment Hospitalized Released

@enum DetectionStatus::UInt8 Undetected UnderObservation TestPending Detected

@enum ContactKind::UInt8 NoContact=0 HouseholdContact FriendshipContact SporadicContact ConstantKernelContact OutsideContact