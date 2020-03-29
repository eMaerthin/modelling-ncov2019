function execute!(state::SimState, params::SimParams, event::OutsideInfectionEvent)
    if Healthy != subjecthealth(state, event)
        return 
    end
        
    state.infection_status[subject(event)] = Infected
    
    push!(state.infections, source(event) => event)    
    push!(state.queue, 
        BecomeInfectiousEvent(time(event) + params.incubation_times[subject(event)], subject(event))
    )
end

function execute!(state::SimState, params::SimParams, event::TransmissionEvent)
    if Healthy != subjecthealth(state, event)
        return
    end
    
    if (StayHome == subjecthealth(state, event) ) && (event.kind != HouseholdContact)
        return 
    end
        
    push!(state.infections, source(event) => event)
    push!(state.queue, 
        BecomeInfectiousEvent(time(event) + params.incubation_times[subject(event)], subject(event))
    )
end

function execute!(state::SimState, params::SimParams, event::BecomeInfectiousEvent)
    @assert Infected == subjecthealth(state, event)
    state.infection_status[subject(event)] = Infecitous
        
    subject_id = subject(event)
    t0 = params.incubation_times[subject_id]
    t1 = params.symptom_onset_times[subject_id]
    t2 = params.hospitalization_times[subject_id]
    
    severity = params.severity[subject(event)]
    
        
    if Mild == severity || Asymptomatic == severity
        @assert !isnan(t1)
        push!(state.queue, StayHomeMildEvent(time(event) + t1 - t0, subject_id))
    elseif Severe == severty || Critical == severity
        if !isnan(t1)
            push!(state.queue, StayHomeToHospitalizeEvent(time(event) + t1 - t0, subject_id))
        end
        push!(state.queue, GoHospitalEvent(time(event) + t2 - t0, subject_id))        
    else
        @error "Unsupported severity $severity"
    end

    enqueue_transmissions!(ConstantKernelContact, state, params, event.subject_id)
    enqueue_transmissions!(HouseholdContact, state, params, event.subject_id)
end

function execute!(state::SimState, params::SimParams, event::StayHomeMildEvent)
    if target_status == subjecthealth(state, event)
        state.infection_status[event.subject_id] = StayHome
    end
    
    push!(state.queue, RecoveryEvent(time(event) + 14))
        
end

function execute!(state::SimState, params::SimParams, event::StayHomeToHospitalizeEvent)
    if target_status == subjecthealth(state, event)
        state.infection_status[event.subject_id] = StayHome
    end
    
    t1 = params.symptom_onset_times[subject_id]
    t2 = params.hospitalization_times[subject_id]
    
    push!(state.queue, GoHospitalEvent(time(event)+t2-t1, subject_id))
end

function execute!(state::SimState, params::SimParams, event::GoHospitalEvent)
    person_health = subjecthealth(state, event)
    if person_health == StayingHome || person_health == Infectious
        state.infection_status[event.subject_id] = Hospitalized
    end
    
    severity = params.severity[subject(event)]
    #if Severe == severity
    #    push!(state.queue, RecoveryEvent(time(event)+14))
    #elseif Critical == severity
    #    push!(state.queue, DeathEvent(time(event)+14))
    #end
end

function execute!(state::SimState, params::SimParams, event::RecoveryEvent)
    @assert isactive(subjecthealth(event))
    
    state.infection_status[event.subject_id] = Recovered
end

function execute!(state::SimState, params::SimParams, event::DeathEvent)
    @assert Recovered !== subjecthealth(event)
    @assert Dead !== subjecthealth(event)

    state.infection_status[event.subject_id] = Dead
end