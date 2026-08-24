---
status: accepted
---

# Normalize shared Boundary States

Behavior Documents store each synchronized sampled state once as a Boundary State and store each Evidence Unit only as the transition between adjacent boundaries. Call 2 may reuse an accepted BEFORE Boundary as fallible context while still seeing the shared images, but it does not emit a duplicate description; this prevents contradictory `previous AFTER` and `current BEFORE` records without changing Pi0.5's per-transition reader or resampler contract. Old flat endpoint artifacts fail closed and must be regenerated because choosing between conflicting duplicate descriptions is not safely automatable.
