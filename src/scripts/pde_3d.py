#!/usr/bin/env python3
"""
3-D field0-field1-x phase diagram visualization.

PhaseDiagram3D
    Reorganises a FullGridWorker result
    (grid[i0][i1] of EqResult objects) into
    two-phase boundary surfaces for 3-D rendering.

Viz3DWindow
    Non-modal QMainWindow wrapping a PyVista
    QtInteractor.  Displays the boundary surfaces
    as semi-transparent StructuredGrids.

Axis convention (any two-field system)
    x-axis : composition (0 → 1)  — right
    y-axis : field1 (secondary)   — depth
    z-axis : field0 (primary)     — up

Visual scaling
    Field values are kept in physical units so that
    show_bounds() displays correct tick labels.
    set_scale() equalises the visual extent of all
    three axes.

Imports of pyvista and pyvistaqt are deferred to
Viz3DWindow.__init__ so that the rest of the
application works when those packages are absent.
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
    """Pre-processed 3-D phase diagram from a full
    field0 × field1 grid.

    Attributes
    ----------
    field0 : Field — primary axis (z in VTK)
    field1 : Field — secondary axis (y in VTK)
    f0_arr : ndarray (N0,) ascending values
    f1_arr : ndarray (N1,) ascending values
    system : System
    two_phase_surfaces : list[dict]
    """

    field0:             object
    field1:             object
    f0_arr:             np.ndarray
    f1_arr:             np.ndarray
    system:             object
    two_phase_surfaces: list

    @classmethod
    def from_grid(cls, grid, field0, f0_arr,
                  field1, f1_arr, system):
        """Build from a FullGridWorker result.

        Parameters
        ----------
        grid   : list[list[EqResult]]
            grid[i0][i1]; arrays may be descending
            (as produced by linspace(max, min, N)).
        field0 : Field — outer-loop field
        f0_arr : ndarray — values (may be descending)
        field1 : Field — inner-loop field
        f1_arr : ndarray — values (may be descending)
        system : System

        Returns
        -------
        PhaseDiagram3D with arrays sorted ascending.
        Each surface dict: label, phases, x_left,
        x_right (shape N0 × N1, NaN where absent).
        """
        N0 = len(f0_arr)
        N1 = len(f1_arr)

        f0_sorted = np.sort(f0_arr)
        f1_sorted = np.sort(f1_arr)

        # Map descending indices → ascending.
        signatures = {}

        for i0 in range(N0):
            j0 = N0 - 1 - i0
            for i1 in range(N1):
                j1 = N1 - 1 - i1
                result = grid[i0][i1]
                for region in (
                        result.two_phase_regions):
                    sig = tuple(region['phases'])
                    if sig not in signatures:
                        signatures[sig] = {
                            'phases': sig,
                            'x_left': np.full(
                                (N0, N1), np.nan),
                            'x_right': np.full(
                                (N0, N1), np.nan),
                        }
                    signatures[sig][
                        'x_left'][j0, j1] = (
                            region['x0'])
                    signatures[sig][
                        'x_right'][j0, j1] = (
                            region['x1'])

        phases = system.phases
        surfaces = []
        for sig, data in signatures.items():
            if len(sig) >= 2:
                label = (
                    f'{phases[sig[0]].name}'
                    f' + {phases[sig[1]].name}')
            else:
                label = phases[sig[0]].name
            surfaces.append({
                'label': label,
                'phases': sig,
                'x_left': data['x_left'],
                'x_right': data['x_right'],
            })

        return cls(
            field0=field0, field1=field1,
            f0_arr=f0_sorted,
            f1_arr=f1_sorted,
            system=system,
            two_phase_surfaces=surfaces)

    # ------------------------------------------------------------------
    # PyVista export
    # ------------------------------------------------------------------

    def to_pyvista_surfaces(self):
        """Return a list of pyvista.StructuredGrid,
        two per two-phase region.

        Axis convention:
            x = composition [0, 1]  (right)
            y = field1 (physical)   (depth)
            z = field0 (physical)   (up)

        Physical coordinates are used directly so
        show_bounds() displays correct tick labels.
        NaN points → VTK blanking.
        """
        import pyvista as pv

        N0 = len(self.f0_arr)
        N1 = len(self.f1_arr)

        # meshgrid 'ij': shape (N0, N1)
        g0, g1 = np.meshgrid(
            self.f0_arr, self.f1_arr,
            indexing='ij')

        surfaces = []
        for surf_data in self.two_phase_surfaces:
            for side in ('x_left', 'x_right'):
                x_2d = surf_data[side]
                # x=composition, y=field1, z=field0
                pts = np.column_stack([
                    x_2d.ravel(),
                    g1.ravel(),
                    g0.ravel(),
                ]).astype(float)

                sg = pv.StructuredGrid()
                sg.points = pts
                sg.dimensions = [N1, N0, 1]
                sg.field_data['label'] = (
                    np.array([surf_data['label']]))
                sg.field_data['side'] = (
                    np.array([side]))
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
    """Non-modal window showing the 3-D phase diagram
    for any two-field system.

    Lazy-imports pyvista and pyvistaqt in __init__.
    """

    def __init__(self, diagram, colors):
        """
        Parameters
        ----------
        diagram : PhaseDiagram3D
        colors  : dict[phase_name, color_str]
        """
        super().__init__()
        f0 = diagram.field0
        f1 = diagram.field1
        self.setWindowTitle(
            f'3D {f0.symbol}-{f1.symbol}-x'
            f' Phase Diagram')

        import pyvista as pv          # noqa: F401
        from pyvistaqt import QtInteractor

        system = diagram.system
        surfaces_data = (
            diagram.to_pyvista_surfaces())
        phase_names = [
            p.name for p in system.phases]

        central = QWidget()
        self.setCentralWidget(central)
        vbox = QVBoxLayout(central)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(4)

        # ---- PyVista interactor ----
        self._plotter = QtInteractor(
            parent=central)
        self._actors = {}

        surf_idx = 0
        for surf_dict in (
                diagram.two_phase_surfaces):
            sig = surf_dict['phases']
            label = surf_dict['label']

            left_name = (
                phase_names[sig[0]]
                if sig[0] < len(phase_names)
                else '')
            right_name = (
                phase_names[sig[1]]
                if (len(sig) > 1
                    and sig[1] < len(phase_names))
                else left_name)

            for side, pname in (
                    ('x_left', right_name),
                    ('x_right', left_name)):
                sg = surfaces_data[surf_idx]
                surf_idx += 1
                color = colors.get(
                    pname, '#888888')
                actor = self._plotter.add_mesh(
                    sg, color=color,
                    opacity=0.6,
                    label=f'{label} ({side})',
                    show_scalar_bar=False)
                self._actors[
                    f'{label}|{side}'] = actor

        # -- Axis labels from Field objects ----
        # z = field0 (up), y = field1 (depth)
        f0_min = diagram.f0_arr[0]
        f0_max = diagram.f0_arr[-1]
        f1_min = diagram.f1_arr[0]
        f1_max = diagram.f1_arr[-1]
        f0_span = max(f0_max - f0_min, 1e-9)
        f1_span = max(f1_max - f1_min, 1e-9)

        self._plotter.set_scale(
            xscale=1.0,
            yscale=1.0 / f1_span,
            zscale=1.0 / f0_span)

        def _axis_label(field):
            if field.unit:
                return (f'{field.symbol}'
                        f' ({field.unit})')
            return field.symbol

        f0_label = _axis_label(f0)
        f1_label = _axis_label(f1)
        comp_label = 'x'
        if (system.components
                and len(system.components) >= 1):
            comp_label = (
                f'x({system.components[-1]})')

        self._plotter.show_bounds(
            xtitle=comp_label,
            ytitle=f1_label,
            ztitle=f0_label,
            show_xaxis=True,
            show_yaxis=True,
            show_zaxis=True)

        self._plotter.add_axes(
            xlabel=comp_label,
            ylabel=f1_label,
            zlabel=f0_label)
        self._plotter.add_legend()
        self._plotter.reset_camera()

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
