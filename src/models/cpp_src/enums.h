#pragma once

enum DetectionStatus
{
    NotDetected, Detected
};

enum QuarantineStatus
{
    NoQuarantine, Quarantine
};

enum CaseSeverity
{
    // Stuff *does* depend on this enum being continuous, and
    // having UNCALCULATED as the last item. Please don't break these
    // assumptions if adding more classes.
    Asymptomatic, Mild, Severe, Critical, UNCALCULATED
};
