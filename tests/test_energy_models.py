"""C-4: Energy model unit tests.

Verifies known-good G values at specific (x, T, P)
inputs for HSModel, PolyModel, PiecewisePatchModel,
and CALPHADModel.  Also checks VLE tangency
conditions for compute_vle_gas_hs output.
"""

import pathlib

import numpy as np
import pytest

from pde_energy import (
    HSModel,
    PolyModel,
    PiecewisePatchModel,
    CALPHADModel,
    compute_vle_gas_hs,
)

# Path to the demo TDB used by CALPHAD tests.
_DEMO_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / 'jobs' / 'demo')
_AL_MG_TDB = _DEMO_DIR / 'al-mg-demo.tdb'


# -------------------------------------------------------
# HSModel — G = H(x) - T*S(x) [+ P*V(x)]
# -------------------------------------------------------

class TestHSModel:
    """Test HSModel with hand-calculated G values.

    Model: H(x) = 8 - 2x + 2x²
           S(x) = 0.01
           V(x) = 0.02
    """

    def setup_method(self):
        """Build the test model once per test."""
        self.model = HSModel(
            [8.0, -2.0, 2.0], [0.01],
            V_coeffs=[0.02])

    def test_G_at_x0(self):
        """G(0) = 8 - 1000*0.01 + 2*0.02 = -1.96"""
        fv = {'temperature': 1000.0,
              'pressure': 2.0}
        G = self.model.gibbs(
            np.array([0.0]), fv)
        assert G == pytest.approx(-1.96)

    def test_G_at_x05(self):
        """G(0.5) = 7.5 - 10 + 0.04 = -2.46"""
        fv = {'temperature': 1000.0,
              'pressure': 2.0}
        G = self.model.gibbs(
            np.array([0.5]), fv)
        assert G == pytest.approx(-2.46)

    def test_G_at_x1(self):
        """G(1) = (8-2+2) - 10 + 0.04 = -1.96"""
        fv = {'temperature': 1000.0,
              'pressure': 2.0}
        G = self.model.gibbs(
            np.array([1.0]), fv)
        assert G == pytest.approx(-1.96)

    def test_no_pressure_term_when_P_zero(self):
        """With P=0, V term contributes nothing."""
        fv_0 = {'temperature': 1000.0,
                'pressure': 0.0}
        fv_2 = {'temperature': 1000.0,
                'pressure': 2.0}
        G0 = self.model.gibbs(
            np.array([0.5]), fv_0)
        G2 = self.model.gibbs(
            np.array([0.5]), fv_2)
        # Difference should be exactly P * V(0.5)
        assert (G2 - G0) == pytest.approx(
            2.0 * 0.02)

    def test_vectorised_evaluation(self):
        """gibbs() should accept and return arrays."""
        fv = {'temperature': 500.0,
              'pressure': 0.0}
        x = np.linspace(0, 1, 11)
        G = self.model.gibbs(x, fv)
        assert G.shape == x.shape
        # Check endpoint: G(0) = 8 - 500*0.01 = 3.0
        assert G[0] == pytest.approx(3.0)


class TestHSModelIdealGas:
    """HSModel with ideal-gas R*T*ln(P/P0) term."""

    def test_ideal_gas_contribution(self):
        """G += R*T*ln(P/P_ref) at any composition."""
        R_gas = 0.008314
        model = HSModel(
            [8.0], [0.01],
            ideal_gas=True,
            R_gas=R_gas, P_ref=1.0)
        fv = {'temperature': 1000.0,
              'pressure': 2.0}
        G = model.gibbs(np.array([0.5]), fv)
        # G = 8 - 10 + R*T*ln(2)
        expected = (8.0 - 10.0
                    + R_gas * 1000.0
                    * np.log(2.0))
        assert G == pytest.approx(expected)

    def test_no_ideal_gas_at_P_ref(self):
        """ln(P/P_ref) = 0 when P == P_ref."""
        model = HSModel(
            [8.0], [0.01],
            ideal_gas=True,
            R_gas=0.008314, P_ref=1.0)
        fv_1 = {'temperature': 1000.0,
                'pressure': 1.0}
        # Same model without ideal gas flag:
        plain = HSModel([8.0], [0.01])
        G_ig = model.gibbs(
            np.array([0.5]), fv_1)
        G_plain = plain.gibbs(
            np.array([0.5]), fv_1)
        assert G_ig == pytest.approx(G_plain)


# -------------------------------------------------------
# PolyModel — G = sum_i c_i(T) * x^i [+ P*V(x)]
# -------------------------------------------------------

class TestPolyModel:
    """Test PolyModel with hand-calculated G values.

    Model: c0(T) = 5 - 2T
           c1(T) = 0
           c2(T) = -1 + 0.05T
    So G(x, T) = (5-2T) + (-1+0.05T)*x²
    """

    def setup_method(self):
        self.model = PolyModel(
            [[5.0, -2.0], [0.0], [-1.0, 0.05]])

    def test_G_at_x05_T800(self):
        """G(0.5, 800) = (5-1600) + (-1+40)*0.25
        = -1595 + 9.75 = -1585.25
        """
        fv = {'temperature': 800.0,
              'pressure': 0.0}
        G = self.model.gibbs(
            np.array([0.5]), fv)
        assert G == pytest.approx(-1585.25)

    def test_G_at_x0_T800(self):
        """G(0, 800) = 5 - 1600 = -1595"""
        fv = {'temperature': 800.0,
              'pressure': 0.0}
        G = self.model.gibbs(
            np.array([0.0]), fv)
        assert G == pytest.approx(-1595.0)

    def test_vectorised(self):
        """Array input produces array output."""
        fv = {'temperature': 800.0,
              'pressure': 0.0}
        x = np.array([0.0, 0.5, 1.0])
        G = self.model.gibbs(x, fv)
        assert G.shape == (3,)


# -------------------------------------------------------
# PiecewisePatchModel
# -------------------------------------------------------

class TestPiecewisePatchModel:
    """Test that PiecewisePatchModel switches between
    the core H polynomial and properly computed patch
    polynomials at the cut-off compositions.

    Patch H coefficients are computed by
    compute_left/right_patch_H (from pde_energy) to
    guarantee value and slope continuity at the cut.
    """

    def _make_patched_model(self):
        """Build a PiecewisePatchModel with properly
        computed left and right patch polynomials.
        """
        from pde_energy import (
            compute_left_patch_H,
            compute_right_patch_H)

        H_core = [8.0, -2.0, 2.0]
        S_core = [0.01]
        # Target phase for patches: a flat curve.
        H_target = [5.0]
        S_target = [0.005]
        x_cut_left = 0.3
        x_cut_right = 0.7
        T_ref = 1000.0

        H_left = compute_left_patch_H(
            H_core, S_core,
            H_target, S_target,
            0.0, x_cut_left, T_ref)
        H_right = compute_right_patch_H(
            H_core, S_core,
            H_target, S_target,
            1.0, x_cut_right, T_ref)

        model = PiecewisePatchModel(
            H_core, S_core,
            H_left=H_left,
            x_cut_left=x_cut_left,
            H_right=H_right,
            x_cut_right=x_cut_right)
        return model

    def test_continuity_at_left_cut(self):
        """G must be continuous at x_cut_left."""
        model = self._make_patched_model()
        fv = {'temperature': 1000.0,
              'pressure': 0.0}
        eps = 1e-8
        G_below = model.gibbs(
            np.array([0.3 - eps]), fv)
        G_above = model.gibbs(
            np.array([0.3 + eps]), fv)
        assert G_below == pytest.approx(
            G_above, abs=1e-4)

    def test_continuity_at_right_cut(self):
        """G must be continuous at x_cut_right."""
        model = self._make_patched_model()
        fv = {'temperature': 1000.0,
              'pressure': 0.0}
        eps = 1e-8
        G_below = model.gibbs(
            np.array([0.7 - eps]), fv)
        G_above = model.gibbs(
            np.array([0.7 + eps]), fv)
        assert G_below == pytest.approx(
            G_above, abs=1e-4)

    def test_core_region_unchanged(self):
        """Between the two cuts, G matches plain HS."""
        model = self._make_patched_model()
        plain = HSModel(
            [8.0, -2.0, 2.0], [0.01])
        fv = {'temperature': 1000.0,
              'pressure': 0.0}
        x_core = np.array([0.4, 0.5, 0.6])
        G_patched = model.gibbs(x_core, fv)
        G_plain = plain.gibbs(x_core, fv)
        np.testing.assert_allclose(
            G_patched, G_plain)


# -------------------------------------------------------
# VLE tangency conditions
# -------------------------------------------------------

class TestVLETangency:
    """compute_vle_gas_hs must produce H_gas, S_gas
    such that G_gas == G_liq at each pure-component
    boiling point.
    """

    def test_tangency_at_boiling_points(self):
        """G_gas(x, T_bp) == G_liq(x, T_bp) at x=0
        and x=1 for the respective boiling points.
        """
        liq_H = [8.0, -2.0, 2.0]
        liq_S = [0.01, 0.0, 0.0]
        T_bp_A, T_bp_B = 350.0, 400.0
        L_A, L_B = 1.0, 1.0

        H_gas, S_gas = compute_vle_gas_hs(
            liq_H, liq_S,
            T_bp_A, T_bp_B, L_A, L_B)

        gas = HSModel(H_gas, S_gas)
        liq = HSModel(liq_H, liq_S)

        # At x=0, T=T_bp_A: G_gas == G_liq
        fv_A = {'temperature': T_bp_A,
                'pressure': 0.0}
        dG_A = (gas.gibbs(np.array([0.0]), fv_A)
                - liq.gibbs(np.array([0.0]), fv_A))
        assert dG_A == pytest.approx(0.0, abs=1e-10)

        # At x=1, T=T_bp_B: G_gas == G_liq
        fv_B = {'temperature': T_bp_B,
                'pressure': 0.0}
        dG_B = (gas.gibbs(np.array([1.0]), fv_B)
                - liq.gibbs(np.array([1.0]), fv_B))
        assert dG_B == pytest.approx(0.0, abs=1e-10)

    def test_gas_above_liquid_below_Tbp(self):
        """Below T_bp, the gas should be less stable
        (higher G) than the liquid at the pure-
        component endpoint.
        """
        liq_H = [8.0, -2.0, 2.0]
        liq_S = [0.01, 0.0, 0.0]
        H_gas, S_gas = compute_vle_gas_hs(
            liq_H, liq_S,
            350.0, 400.0, 1.0, 1.0)

        gas = HSModel(H_gas, S_gas)
        liq = HSModel(liq_H, liq_S)

        # Below T_bp_A at x=0: G_gas > G_liq
        fv_low = {'temperature': 300.0,
                  'pressure': 0.0}
        G_gas = gas.gibbs(np.array([0.0]), fv_low)
        G_liq = liq.gibbs(np.array([0.0]), fv_low)
        assert G_gas > G_liq


# -------------------------------------------------------
# CALPHADModel — pycalphad TDB wrapper
# -------------------------------------------------------

# All CALPHAD tests require pycalphad; skip the
# entire class gracefully when it is not installed.
pycalphad = pytest.importorskip('pycalphad')


class TestCALPHADModel:
    """Tests for CALPHADModel using the demo
    Al-Mg TDB (al-mg-demo.tdb).

    The TDB defines three phases — LIQUID, FCC_A1,
    HCP_A3 — with simplified parameters that produce
    a qualitatively correct eutectic diagram.  G
    values are in J/mol (SI).
    """

    @pytest.fixture(autouse=True)
    def _load_db(self):
        """Load the TDB database once per test."""
        self.db = pycalphad.Database(
            str(_AL_MG_TDB))

    # -- Sublattice structure -------------------

    def test_liquid_sf_layout(self):
        """LIQUID has one sublattice (AL, MG):
        site-fraction layout should be [['AL','MG']].
        """
        model = CALPHADModel(
            self.db, 'LIQUID', ['AL', 'MG'])
        assert model._sf_layout == [
            ['AL', 'MG']]

    def test_fcc_sf_layout(self):
        """FCC_A1 has two sublattices (AL,MG)(VA):
        layout should be [['AL','MG'], ['VA']].
        """
        model = CALPHADModel(
            self.db, 'FCC_A1', ['AL', 'MG'])
        assert model._sf_layout == [
            ['AL', 'MG'], ['VA']]

    def test_hcp_sf_layout(self):
        """HCP_A3 has two sublattices (AL,MG)(VA):
        layout should be [['AL','MG'], ['VA']].
        """
        model = CALPHADModel(
            self.db, 'HCP_A3', ['AL', 'MG'])
        assert model._sf_layout == [
            ['AL', 'MG'], ['VA']]

    # -- Pure-endpoint G values -----------------

    def test_fcc_pure_al_is_reference(self):
        """G(FCC_A1, pure AL) = 0 at any T (SER
        reference state for aluminium).
        """
        model = CALPHADModel(
            self.db, 'FCC_A1', ['AL', 'MG'])
        fv = {'temperature': 800.0}
        G = model.gibbs(np.array([0.0]), fv)
        assert float(G[0]) == pytest.approx(
            0.0, abs=0.01)

    def test_hcp_pure_mg_is_reference(self):
        """G(HCP_A3, pure MG) = 0 at any T (SER
        reference state for magnesium).
        """
        model = CALPHADModel(
            self.db, 'HCP_A3', ['AL', 'MG'])
        fv = {'temperature': 800.0}
        G = model.gibbs(np.array([1.0]), fv)
        assert float(G[0]) == pytest.approx(
            0.0, abs=0.01)

    def test_liquid_pure_al_melting(self):
        """G(LIQUID, pure AL) should cross zero
        near 933 K (the AL melting point encoded
        in the TDB as 10711/11.473 = 933.4 K).
        """
        model = CALPHADModel(
            self.db, 'LIQUID', ['AL', 'MG'])
        fv_below = {'temperature': 900.0}
        fv_above = {'temperature': 960.0}
        G_below = float(model.gibbs(
            np.array([0.0]), fv_below)[0])
        G_above = float(model.gibbs(
            np.array([0.0]), fv_above)[0])
        # Below melting: liquid less stable (G > 0)
        assert G_below > 0.0
        # Above melting: liquid more stable (G < 0)
        assert G_above < 0.0

    # -- Vectorised evaluation ------------------

    def test_vectorised_output_shape(self):
        """gibbs() returns an array matching the
        input shape.
        """
        model = CALPHADModel(
            self.db, 'LIQUID', ['AL', 'MG'])
        x = np.linspace(0, 1, 21)
        fv = {'temperature': 800.0}
        G = model.gibbs(x, fv)
        assert G.shape == (21,)

    def test_G_finite_across_range(self):
        """G values must be finite (no NaN or inf)
        for all three phases across x in [0, 1].
        """
        x = np.linspace(0, 1, 51)
        fv = {'temperature': 800.0}
        for phase_name in (
                'LIQUID', 'FCC_A1', 'HCP_A3'):
            model = CALPHADModel(
                self.db, phase_name,
                ['AL', 'MG'])
            G = model.gibbs(x, fv)
            assert np.all(np.isfinite(G)), (
                f'{phase_name} has non-finite G')

    # -- Pressure default -----------------------

    def test_default_pressure(self):
        """When no pressure is in field_values,
        P_default (101325 Pa) is used.
        """
        model = CALPHADModel(
            self.db, 'LIQUID', ['AL', 'MG'],
            P_default=101325.0)
        fv_no_P = {'temperature': 800.0}
        fv_with_P = {'temperature': 800.0,
                     'pressure': 101325.0}
        G1 = model.gibbs(
            np.array([0.5]), fv_no_P)
        G2 = model.gibbs(
            np.array([0.5]), fv_with_P)
        assert G1 == pytest.approx(G2)

    # -- Couplings property ---------------------

    def test_couplings_empty(self):
        """CALPHAD couplings returns an empty list
        (opaque model — no CouplingTerm decomposition).
        """
        model = CALPHADModel(
            self.db, 'LIQUID', ['AL', 'MG'])
        assert model.couplings == []
