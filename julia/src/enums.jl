#@enum HealthState::UInt8 Healthy Infected Infectious StayingHome Hospitalized Recovered Dead HomeQuarantine
@enum Severity::UInt32 Asymptomatic=1 Mild Severe Critical

@enum HealthState::UInt8 Healthy Incubating Infectious MildSymptoms SevereSymptoms CriticalSymptoms Recovered Dead

@enum HospitalizationState::UInt8 Free HomeTreatment Hospitalized 

@enum ContactKind::UInt8 HouseholdContact FriendshipContact SporadicContact ConstantKernelContact