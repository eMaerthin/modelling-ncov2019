@enum Severity::UInt32 Asymptomatic=1 Mild Severe Critical

@enum HealthState::UInt8 Healthy Incubating Infectious MildSymptoms SevereSymptoms CriticalSymptoms Recovered Dead

@enum FreedomState::UInt8 Free HomeQuarantine HomeTreatment Hospitalized Released

@enum ContactKind::UInt8 UnknownContact HouseholdContact FriendshipContact SporadicContact ConstantKernelContact OutsideContact

@enum DetectionStatus::UInt8 Undetected TestPending Detected