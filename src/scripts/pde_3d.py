#!/usr/bin/env python3
"""
3-D T-P-x phase diagram visualization for PDE.

PhaseDiagram3D
    Reorganises a FullGridWorker result (grid[i_T][i_P] of EqResult objects)
    into two-phase boundary surfaces suitable for 3-D rendering.

Viz3DWindow
    Non-modal QMainWindow wrapping a PyVista QtInteractor.  Displays the
    two-phase boundary surfaces as semi-transparent StructuredGrids.

Axis convention
    x-axis : composition (0 → 1)
    y-axis : temperature T (K), ascending
    z-axis : pressure P, ascending

Imports of pyvista and pyvistaqt are deferred to Viz3DWindow.__init__ so that
the rest of the application works normally when those packages are absent.
"""

import dataclasses

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QLabel, QMainWindow,
                                QPushButton, QSlider, QVBoxLayout, QWidget)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PhaseDiagram3D:
    """Pre-processed 3-D phase diagram data derived from a full T-P grid."""

    T_arr:              np.ndarray  # shape (N_T,) ascending
    P_arr:              np.ndarray  # shape (N_P,) ascending
    system:             object      # reference to the original System object
    two_phase_surfaces: list        # list of dicts (see from_grid docstring)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_grid(cls, grid, T_arr, P_arr, system):
        """Build a PhaseDiagram3D from a FullGridWorker result.

        Parameters
        ----------
        grid : list[list[EqResult]]
            grid[i_T][i_P] where i_T=0 corresponds to T_arr[0] (T_max when
            stored by MainWindow, i.e. T_arr is descending).
        T_arr : np.ndarray  shape (N_T,)  — as stored in MainWindow._T_arr (descending)
        P_arr : np.ndarray  shape (N_P,)  — as stored in MainWindow._P_arr (descending)
        system : System

        Returns
        -------
        PhaseDiagram3D with T_arr/P_arr sorted ascending and two_phase_surfaces
        filled.  Each surface dict has keys:
            'label'   : str            e.g. "liquid + solid"
            'phases'  : tuple[int]     phase indices into system.phases
            'x_left'  : ndarray (N_T, N_P) NaN where region absent
            'x_right' : ndarray (N_T, N_P) NaN where region absent
        """
        N_T = len(T_arr)
        N_P = len(P_arr)

        T_sorted = np.sort(T_arr)   # ascending
        P_sorted = np.sort(P_arr)   # ascending

        # Accumulate two-phase region surfaces keyed by phase-index tuple.
        # j_T = N_T-1-i_T maps descending i_T → ascending j_T (works because
        # linspace(T_max, T_min, N_T) is uniformly spaced).
        signatures = {}   # sig → {'phases': sig, 'x_left': ndarray, 'x_right': ndarray}

        for i_T in range(N_T):
            j_T = N_T - 1 - i_T
            for i_P in range(N_P):
                j_P = N_P - 1 - i_P
                result = grid[i_T][i_P]
                for region in result.two_phase_regions:
                    sig = tuple(region['phases'])
                    if sig not in signatures:
                        signatures[sig] = {
                            'phases':  sig,
                            'x_left':  np.full((N_T, N_P), np.nan),
                            'x_right': np.full((N_T, N_P), np.nan),
                        }
                    signatures[sig]['x_left'][j_T, j_P]  = region['x0']
                    signatures[sig]['x_right'][j_T, j_P] = region['x1']

        # Build human-readable labels from phase names.
        phases = system.phases
        surfaces = []
        for sig, data in signatures.items():
            if len(sig) >= 2:
                label = f'{phases[sig[0]].name} + {phases[sig[1]].name}'
            else:
                label = phases[sig[0]].name
            surfaces.append({
                'label':   label,
                'phases':  sig,
                'x_left':  data['x_left'],
                'x_right': data['x_right'],
            })

        return cls(T_arr=T_sorted, P_arr=P_sorted,
                   system=system, two_phase_surfaces=surfaces)

    # ------------------------------------------------------------------
    # PyVista export
    # ------------------------------------------------------------------

    def to_pyvista_surfaces(self):
        """Return a list of pyvista.StructuredGrid, two per two-phase region.

        Axis convention: x=composition, y=T (K), z=P.

        NaN points are left in place; VTK renders only cells whose all four
        corner points are finite (StructuredGrid blanking).
        """
        import pyvista as pv

        N_T = len(self.T_arr)
        N_P = len(self.P_arr)

        # meshgrid with indexing='ij': T_grid[j_T, j_P], P_grid[j_T, j_P]
        T_grid, P_grid = np.meshgrid(self.T_arr, self.P_arr, indexing='ij')

        surfaces = []
        for surf_data in self.two_phase_surfaces:
            for side in ('x_left', 'x_right'):
                x_2d = surf_data[side]   # shape (N_T, N_P)

                pts = np.column_stack([
                    x_2d.ravel(),
                    T_grid.ravel(),
                    P_grid.ravel(),
                ]).astype(float)

                sg = pv.StructuredGrid()
                sg.points = pts
                # PyVista dim order: [fast, mid, slow]
                # The ravel() above iterates P fastest (dim-1), T next (dim-0).
                sg.dimensions = [N_P, N_T, 1]
                sg.field_data['label'] = np.array([surf_data['label']])
                sg.field_data['side']  = np.array([side])
                surfaces.append(sg)

        return surfaces

    # ------------------------------------------------------------------
    # Stubs — reserved for future XDMF/HDF5 export
    # ------------------------------------------------------------------

    def to_pyvista_volume(self, n_x=500):
        """Dense label-grid as a RectilinearGrid (not yet implemented)."""
        raise NotImplementedError("Dense label-grid export not yet implemented")

    def to_xdmf(self, path, mode='surfaces'):
        """Write XDMF + HDF5 files for ParaView (not yet implemented)."""
        raise NotImplementedError("XDMF+HDF5 export not yet implemented")


# ---------------------------------------------------------------------------
# 3-D viewer window
# ---------------------------------------------------------------------------

class Viz3DWindow(QMainWindow):
    """Non-modal window showing the 3-D T-P-x phase diagram.

    Lazy-imports pyvista and pyvistaqt in __init__ so that the rest of the
    application continues to function when those packages are not installed.
    """

    def __init__(self, diagram, colors):
        """
        Parameters
        ----------
        diagram : PhaseDiagram3D
        colors  : dict[phase_name, color_str]  — from MainWindow._colors
        """
        super().__init__()
        self.setWindowTitle('3D T-P-x Phase Diagram')

        import pyvista as pv          # noqa: F401 (imported for side-effects / type check)
        from pyvistaqt import QtInteractor

        system        = diagram.system
        surfaces_data = diagram.to_pyvista_surfaces()
        phase_names   = [p.name for p in system.phases]

        # Build the central widget.
        central = QWidget()
        self.setCentralWidget(central)
        vbox = QVBoxLayout(central)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(4)

        # ---- PyVista interactor ----
        self._plotter = QtInteractor(parent=central)
        self._actors  = {}   # key f"{label}|{side}" → VTK actor

        # Add surfaces. surfaces_data is flat: [left0, right0, left1, right1, …].
        surf_idx = 0
        for surf_dict in diagram.two_phase_surfaces:
            sig   = surf_dict['phases']
            label = surf_dict['label']

            left_name  = phase_names[sig[0]] if sig[0] < len(phase_names) else ''
            right_name = (phase_names[sig[1]]
                          if len(sig) > 1 and sig[1] < len(phase_names)
                          else left_name)

            # Left boundary surface colored by the right-phase color (and vice versa)
            # so each surface appears as the "face" of the adjacent single-phase region.
            for side, phase_name in (('x_left', right_name), ('x_right', left_name)):
                sg    = surfaces_data[surf_idx]
                surf_idx += 1
                color = colors.get(phase_name, '#888888')
                actor = self._plotter.add_mesh(
                    sg,
                    color=color,
                    opacity=0.6,
                    label=f'{label} ({side})',
                    show_scalar_bar=False,
                )
                self._actors[f'{label}|{side}'] = actor

        # Scale all three axes to equal visual length so the view looks cubic
        # rather than a thin sliver (composition 0-1, T 250-500 K, P 0.5-5 atm
        # would otherwise differ by two orders of magnitude).
        T_span = max(diagram.T_arr[-1] - diagram.T_arr[0], 1e-9)
        P_span = max(diagram.P_arr[-1] - diagram.P_arr[0], 1e-9)
        # x (composition) is always [0, 1]; scale the other two axes relative to it.
        self._plotter.set_scale(xscale=T_span, yscale=1.0, zscale=T_span / P_span)

        # Axis labels and bounds.
        self._plotter.add_axes()
        P_unit  = getattr(system, 'P_unit', '')
        P_label = f'P ({P_unit})' if P_unit else 'P'
        self._plotter.show_bounds(
            xtitle='x',
            ytitle='T (K)',
            ztitle=P_label,
            show_xaxis=True,
            show_yaxis=True,
            show_zaxis=True,
        )
        self._plotter.add_legend()

        # ---- top control row ----
        top_row = QHBoxLayout()

        # Visibility checkboxes — one per two-phase region.
        self._vis_checks = {}
        for surf_dict in diagram.two_phase_surfaces:
            lbl = surf_dict['label']
            cb  = QCheckBox(lbl)
            cb.setChecked(True)
            cb.toggled.connect(self._make_vis_toggle(lbl))
            top_row.addWidget(cb)
            self._vis_checks[lbl] = cb

        top_row.addStretch()

        # Opacity slider.
        top_row.addWidget(QLabel('Opacity:'))
        self._opacity_slider = QSlider(Qt.Horizontal)
        self._opacity_slider.setMinimum(0)
        self._opacity_slider.setMaximum(100)
        self._opacity_slider.setValue(60)
        self._opacity_slider.setMaximumWidth(120)
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        top_row.addWidget(self._opacity_slider)
        top_row.addSpacing(16)

        # Export button (stub — not yet implemented).
        export_btn = QPushButton('Export\u2026')
        export_btn.setEnabled(False)
        export_btn.setToolTip('XDMF+HDF5 export not yet implemented')
        top_row.addWidget(export_btn)
        top_row.addSpacing(8)

        close_btn = QPushButton('Close')
        close_btn.clicked.connect(self.close)
        top_row.addWidget(close_btn)

        vbox.addLayout(top_row)
        vbox.addWidget(self._plotter, stretch=1)

        self.resize(900, 700)

    # ------------------------------------------------------------------
    # Slot helpers
    # ------------------------------------------------------------------

    def _make_vis_toggle(self, label):
        """Return a slot that shows/hides both surfaces for a two-phase region."""
        def _slot(visible):
            for side in ('x_left', 'x_right'):
                actor = self._actors.get(f'{label}|{side}')
                if actor is not None:
                    actor.SetVisibility(visible)
            self._plotter.render()
        return _slot

    def _on_opacity_changed(self, value):
        opacity = value / 100.0
        for actor in self._actors.values():
            actor.GetProperty().SetOpacity(opacity)
        self._plotter.render()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self._plotter.close()
        super().closeEvent(event)
