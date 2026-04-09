"""Shared pytest configuration for PDE tests.

Adds the src/scripts directory to sys.path so that
pde_energy, pde_phase, pde_compute, pde_input, and
pde_check are importable without installation.
"""

import os
import sys

_scripts_dir = os.path.join(
    os.path.dirname(__file__),
    '..', 'src', 'scripts')
_scripts_dir = os.path.abspath(_scripts_dir)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
