# CMake file for PDE set of scripts.

# Document the current location
message("At the set of scripts.")

set (SCRIPTS
pde.py
)

set (MODULES
pde_energy.py
pde_phase.py
pde_input.py
pde_compute.py
pde_viz.py
pde_builder.py
pde_3d.py
pde_check.py
)

install (PROGRAMS ${SCRIPTS} DESTINATION bin)
install (FILES ${MODULES} DESTINATION bin)
