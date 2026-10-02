"""
Virtual Wind Tunnel
===================
One program, three modules (pick one when you run it):

    1. Standard airfoil comparison   - NACA 2412 / 4412 / 0012 polars + plots
    2. Morphing wing study           - camber / twist / thickness morphing
    3. Interactive tester (GUI)      - live sliders, checkboxes, CSV export

Usage
-----
    python wind_tunnel.py                 # shows a menu
    python wind_tunnel.py --module 3      # jump straight to the GUI
    python wind_tunnel.py -m 2 --morph twist
    python wind_tunnel.py -m 1 --no-show  # save plots only, no windows

All three modules share the same physics core (AirfoilGenerator,
AerodynamicsCalculator, WindTunnelSimulator), so results are identical
no matter which module you use.

Requires: numpy, matplotlib (tkinter only for module 3).
"""

import argparse
import os
import warnings
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

warnings.filterwarnings("ignore")

# ============================================================================
# CONSTANTS & OUTPUT FOLDER
# ============================================================================

# Save outputs next to this script (falls back to current directory in a console)
try:
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(_BASE_DIR, "outputs")

# ISA sea-level speed of sound and the tunnel's speed range
SPEED_OF_SOUND = 340.29                     # m/s
MIN_VELOCITY = 50.0                         # m/s (base speed)
MAX_MACH = 0.9
MAX_VELOCITY = MAX_MACH * SPEED_OF_SOUND    # ~306 m/s

# Default tunnel conditions (sea level, 15 °C)
DEFAULT_VELOCITY = 50.0
DEFAULT_CHORD = 1.0
DEFAULT_SPAN = 8.0
AIR_DENSITY = 1.225                         # kg/m³
DYNAMIC_VISCOSITY = 1.81e-5                 # Pa·s

DEFAULT_ANGLES = np.linspace(-10, 20, 31)   # angle-of-attack sweep (deg)

# GUI camber adjustment range (percent of chord, added to each airfoil's camber;
# negative values give reflexed / reverse camber)
CAMBER_ADJ_MIN = -6.0
CAMBER_ADJ_MAX = 6.0


# ============================================================================
# SHARED CORE  (used by all three modules)
# ============================================================================

@dataclass
class AirfoilConfig:
    """Configuration for airfoil geometry"""
    name: str
    camber: float       # Maximum camber (fraction of chord)
    camber_pos: float   # Position of max camber (fraction of chord)
    thickness: float    # Maximum thickness (fraction of chord)


@dataclass
class WindTunnelConditions:
    """Wind tunnel operating conditions"""
    velocity: float           # m/s
    air_density: float        # kg/m³
    dynamic_viscosity: float  # Pa·s
    chord_length: float       # m
    wingspan: float           # m


def default_conditions(velocity=DEFAULT_VELOCITY, chord=DEFAULT_CHORD,
                       span=DEFAULT_SPAN) -> WindTunnelConditions:
    """Sea-level conditions with the given velocity / geometry."""
    return WindTunnelConditions(
        velocity=velocity,
        air_density=AIR_DENSITY,
        dynamic_viscosity=DYNAMIC_VISCOSITY,
        chord_length=chord,
        wingspan=span,
    )


class AirfoilGenerator:
    """Generate NACA-style airfoil coordinates"""

    @staticmethod
    def naca4(m: float, p: float, t: float, n_points: int = 100) -> np.ndarray:
        """
        Generate NACA 4-digit airfoil
        m: camber (0-9, represents 0.0-0.09)
        p: position of max camber (1-9, represents 0.1-0.9)
        t: thickness (01-40, represents 0.01-0.40)
        """
        m = m / 100
        p = p / 10
        t = t / 100

        x = np.linspace(0, 1, n_points)
        y_t = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                       + 0.2843 * x**3 - 0.1015 * x**4)

        y_c = np.zeros_like(x)
        y_c_prime = np.zeros_like(x)

        for i, xi in enumerate(x):
            if xi < p:
                y_c[i] = (m / p**2) * (2 * p * xi - xi**2)
                y_c_prime[i] = (2 * m / p**2) * (p - xi)
            else:
                y_c[i] = (m / (1 - p)**2) * (1 - 2 * p + 2 * p * xi - xi**2)
                y_c_prime[i] = (2 * m / (1 - p)**2) * (p - xi)

        theta = np.arctan(y_c_prime)

        x_upper = x - y_t * np.sin(theta)
        y_upper = y_c + y_t * np.cos(theta)

        x_lower = x + y_t * np.sin(theta)
        y_lower = y_c - y_t * np.cos(theta)

        return np.vstack([
            np.column_stack([x_upper, y_upper]),
            np.column_stack([x_lower[::-1], y_lower[::-1]]),
        ])

    @staticmethod
    def morphing_wing(base_airfoil: np.ndarray, morph_param: float,
                      morph_type: str = 'camber') -> np.ndarray:
        """
        Create morphing wing by deforming base airfoil
        morph_param: morphing parameter (-1 to 1)
        morph_type: 'camber', 'twist', or 'thickness'
        """
        morph_airfoil = base_airfoil.copy()

        if morph_type == 'camber':
            # Increase/decrease camber
            morph_airfoil[:, 1] += morph_param * 0.1 * np.sin(np.pi * base_airfoil[:, 0])

        elif morph_type == 'twist':
            # Apply twist distribution
            center = np.mean(base_airfoil, axis=0)
            angle = morph_param * np.pi / 6
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            morph_airfoil = (morph_airfoil - center) @ rotation_matrix.T + center

        elif morph_type == 'thickness':
            # Scale thickness
            scale_factor = 1 + morph_param * 0.3
            center_y = (base_airfoil[:, 1].max() + base_airfoil[:, 1].min()) / 2
            morph_airfoil[:, 1] = center_y + (base_airfoil[:, 1] - center_y) * scale_factor

        else:
            raise ValueError(f"Unknown morph_type: {morph_type!r}")

        return morph_airfoil


class AerodynamicsCalculator:
    """Calculate aerodynamic coefficients"""

    @staticmethod
    def reynolds_number(velocity: float, chord: float, mu: float, rho: float) -> float:
        """Calculate Reynolds number"""
        return rho * velocity * chord / mu

    @staticmethod
    def _geometry(airfoil: np.ndarray):
        """Return (x grid, mean camber line, thickness ratio, twist angle [rad])
        in chord-aligned, unit-chord coordinates."""
        n = len(airfoil) // 2
        upper = airfoil[:n]
        lower = airfoil[n:][::-1]              # leading edge -> trailing edge
        le = 0.5 * (upper[0] + lower[0])
        te = 0.5 * (upper[-1] + lower[-1])
        chord_vec = te - le
        chord = np.hypot(*chord_vec)
        twist = np.arctan2(chord_vec[1], chord_vec[0])
        c, s = np.cos(-twist), np.sin(-twist)
        rot = np.array([[c, -s], [s, c]])
        up = (upper - le) @ rot.T / chord
        lo = (lower - le) @ rot.T / chord
        xs = np.linspace(0, 1, 201)
        iu, il = np.argsort(up[:, 0]), np.argsort(lo[:, 0])
        yu = np.interp(xs, up[iu, 0], up[iu, 1])
        yl = np.interp(xs, lo[il, 0], lo[il, 1])
        return xs, 0.5 * (yu + yl), float(np.max(yu - yl)), twist

    @staticmethod
    def lift_with_stall(airfoil: np.ndarray, angle_of_attack: float,
                        reynolds: float = 3e6, mach: float = 0.0):
        """
        Thin airfoil theory, CL = 2*pi*(alpha_eff - alpha_L0), blended into a
        flat-plate post-stall curve (CL = sin 2a) once the geometric angle of
        attack passes the stall angle. Thicker sections stall later.

        Returns (cl, separation, a_eff) where separation is 0 (attached flow)
        to 1 (fully stalled) and a_eff is the angle from zero lift in radians.
        """
        xs, yc, t, twist = AerodynamicsCalculator._geometry(airfoil)
        dyc = np.gradient(yc, xs)
        theta = np.linspace(0, np.pi, 400)
        dyc_t = np.interp((1 - np.cos(theta)) / 2, xs, dyc)
        f = dyc_t * (np.cos(theta) - 1)
        alpha_l0 = -np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(theta)) / np.pi
        a_eff = np.radians(angle_of_attack) - twist - alpha_l0

        # Prandtl-Glauert compressibility correction (capped at M = 0.7; it blows up near M = 1)
        pg = 1 / np.sqrt(1 - min(mach, 0.7) ** 2)
        cl_lin = 2 * np.pi * a_eff * pg
        cl_post = np.sin(2 * a_eff)

        # Stall is triggered by the geometric (twist-corrected) angle of attack,
        # so cambered sections reach a higher CLmax before separating.
        a_geo = np.radians(angle_of_attack) - twist
        # ~14.8 deg for a 12% section at Re = 3e6; higher Re delays stall (higher CLmax)
        re_eff = min(reynolds, 1e7)               # Reynolds benefit saturates
        stall = np.radians(10 + 40 * t + 3.0 * np.log10(re_eff / 3e6))
        # Shock-induced separation: stall comes much earlier as Mach rises past ~0.3
        stall *= 1 - 0.7 * min(max((mach - 0.3) / 0.6, 0.0), 1.0)
        if a_geo < 0:
            stall *= 0.8                          # negative stall comes a bit earlier
        width = np.radians(1.8)
        sep = 1 / (1 + np.exp(-(abs(a_geo) - stall - np.radians(3)) / width))
        cl = (1 - sep) * cl_lin + sep * cl_post
        return cl, sep, a_eff

    @staticmethod
    def lift_coefficient_theoretical(airfoil: np.ndarray, angle_of_attack: float) -> float:
        """Lift coefficient including stall (see lift_with_stall)."""
        return AerodynamicsCalculator.lift_with_stall(airfoil, angle_of_attack)[0]

    @staticmethod
    def drag_coefficient(cl: float, reynolds: float, airfoil: np.ndarray,
                         aspect_ratio: float = 8.0,
                         separation: float = 0.0, a_eff: float = 0.0,
                         mach: float = 0.0):
        """
        Estimate drag coefficient
        cd = cd0 + (cl^2 / (pi * e * ar))
        Returns (cd, cd0, cdi).
        """
        _, _, t, _ = AerodynamicsCalculator._geometry(airfoil)
        # Profile drag: skin friction vs Reynolds number, scaled by a
        # thickness form factor (normalised to 1 for a 12% thick section)
        ff = (1 + 2 * t + 60 * t ** 4) / (1 + 2 * 0.12 + 60 * 0.12 ** 4)
        cd0 = (0.008 + 0.0055 * (reynolds / 1e5) ** (-0.2)) * ff
        # Laminar separation bubbles raise profile drag at low Reynolds number
        cd0 *= 1 + 0.6 * max(0.0, np.log10(1e6 / reynolds))

        # Induced drag (elliptical-ish wing)
        oswald_efficiency = 0.95
        cdi = (cl ** 2) / (np.pi * oswald_efficiency * aspect_ratio)

        # Post-stall separation drag (flat-plate-like, grows with sin^2 of angle)
        cd_sep = separation * 2 * np.sin(a_eff) ** 2
        cd0 = cd0 + cd_sep

        # Wave drag (Lock/Korn): drag divergence above the critical Mach number
        m_dd = 0.87 - t - abs(cl) / 10
        m_crit = m_dd - (0.1 / 80) ** (1 / 3)
        if mach > m_crit:
            cd0 = cd0 + 20 * (mach - m_crit) ** 4

        return cd0 + cdi, cd0, cdi

    @staticmethod
    def calculate_forces(cl: float, cd: float, velocity: float,
                         rho: float, area: float) -> Tuple[float, float]:
        """Calculate lift and drag forces (Newtons)"""
        dynamic_pressure = 0.5 * rho * velocity ** 2
        return cl * dynamic_pressure * area, cd * dynamic_pressure * area


class WindTunnelSimulator:
    """Main wind tunnel simulator"""

    def __init__(self, conditions: WindTunnelConditions):
        self.conditions = conditions
        self.wing_area = conditions.chord_length * conditions.wingspan
        self.aspect_ratio = conditions.wingspan / conditions.chord_length
        self.mach = conditions.velocity / SPEED_OF_SOUND
        self.results = []

    def test_airfoil(self, airfoil: np.ndarray, airfoil_name: str,
                     angles: np.ndarray) -> dict:
        """Test airfoil at multiple angles of attack"""
        reynolds = AerodynamicsCalculator.reynolds_number(
            self.conditions.velocity,
            self.conditions.chord_length,
            self.conditions.dynamic_viscosity,
            self.conditions.air_density,
        )

        results = {
            'name': airfoil_name,
            'airfoil': airfoil,
            'reynolds': reynolds,
            'angles': angles,
            'cl': [], 'cd': [], 'cd0': [], 'cdi': [],
            'lift': [], 'drag': [], 'efficiency': [],
        }

        for alpha in angles:
            cl, sep, a_eff = AerodynamicsCalculator.lift_with_stall(
                airfoil, alpha, reynolds, self.mach)
            cd, cd0, cdi = AerodynamicsCalculator.drag_coefficient(
                cl, reynolds, airfoil, aspect_ratio=self.aspect_ratio,
                separation=sep, a_eff=a_eff, mach=self.mach)

            lift, drag = AerodynamicsCalculator.calculate_forces(
                cl, cd, self.conditions.velocity,
                self.conditions.air_density, self.wing_area)

            results['cl'].append(cl)
            results['cd'].append(cd)
            results['cd0'].append(cd0)
            results['cdi'].append(cdi)
            results['lift'].append(lift)
            results['drag'].append(drag)
            results['efficiency'].append(lift / (drag + 1e-6))   # L/D

        for key in ['cl', 'cd', 'cd0', 'cdi', 'lift', 'drag', 'efficiency']:
            results[key] = np.array(results[key])

        self.results.append(results)
        return results


def export_results_csv(results: List[dict], folder: str) -> List[str]:
    """Write one CSV per result into `folder`. Returns the file paths.
    Shared by the batch modules and the GUI's 'Export Data' button."""
    os.makedirs(folder, exist_ok=True)
    header = "Angle_of_Attack(deg),CL,CD,CD0,CDi,Lift(N),Drag(N),Efficiency(L/D)\n"
    paths = []
    for result in results:
        path = os.path.join(folder, f"{result['name']}_data.csv")
        with open(path, 'w') as f:
            f.write(header)
            for i in range(len(result['angles'])):
                f.write(f"{result['angles'][i]:.2f},"
                        f"{result['cl'][i]:.6f},"
                        f"{result['cd'][i]:.6f},"
                        f"{result['cd0'][i]:.6f},"
                        f"{result['cdi'][i]:.6f},"
                        f"{result['lift'][i]:.4f},"
                        f"{result['drag'][i]:.4f},"
                        f"{result['efficiency'][i]:.4f}\n")
        paths.append(path)
    return paths


class WindTunnelVisualizer:
    """Visualize wind tunnel results (static plots for modules 1 and 2)"""

    @staticmethod
    def plot_airfoil_comparison(results_list: List[dict], figsize=(15, 12)):
        """Plot airfoil shapes and aerodynamic characteristics"""
        fig = plt.figure(figsize=figsize)
        colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))

        # Plot 1: Airfoil shapes
        ax1 = fig.add_subplot(2, 3, 1)
        for result, color in zip(results_list, colors):
            airfoil = result['airfoil']
            ax1.plot(airfoil[:, 0], airfoil[:, 1], '-', linewidth=2,
                     label=result['name'], color=color)
            ax1.fill(airfoil[:, 0], airfoil[:, 1], alpha=0.1, color=color)
        ax1.set_xlabel('Chord Position', fontsize=10)
        ax1.set_ylabel('Thickness', fontsize=10)
        ax1.set_title('Airfoil Geometry Comparison', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')

        # Plot 2: Lift coefficient vs angle
        ax2 = fig.add_subplot(2, 3, 2)
        for result, color in zip(results_list, colors):
            ax2.plot(result['angles'], result['cl'], 'o-', linewidth=2,
                     label=result['name'], color=color)
        ax2.set_xlabel('Angle of Attack (°)', fontsize=10)
        ax2.set_ylabel('Lift Coefficient (CL)', fontsize=10)
        ax2.set_title('Lift Coefficient vs AoA', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Drag coefficient vs angle
        ax3 = fig.add_subplot(2, 3, 3)
        for result, color in zip(results_list, colors):
            ax3.plot(result['angles'], result['cd'], 'o-', linewidth=2,
                     label=result['name'], color=color)
        ax3.set_xlabel('Angle of Attack (°)', fontsize=10)
        ax3.set_ylabel('Drag Coefficient (CD)', fontsize=10)
        ax3.set_title('Drag Coefficient vs AoA', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Drag polar (CL vs CD)
        ax4 = fig.add_subplot(2, 3, 4)
        for result, color in zip(results_list, colors):
            ax4.plot(result['cd'], result['cl'], 'o-', linewidth=2,
                     label=result['name'], color=color)
        ax4.set_xlabel('Drag Coefficient (CD)', fontsize=10)
        ax4.set_ylabel('Lift Coefficient (CL)', fontsize=10)
        ax4.set_title('Drag Polar', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Plot 5: Efficiency (L/D) vs angle
        ax5 = fig.add_subplot(2, 3, 5)
        for result, color in zip(results_list, colors):
            ax5.plot(result['angles'], result['efficiency'], 'o-', linewidth=2,
                     label=result['name'], color=color)
        ax5.set_xlabel('Angle of Attack (°)', fontsize=10)
        ax5.set_ylabel('Efficiency (L/D)', fontsize=10)
        ax5.set_title('Aerodynamic Efficiency vs AoA', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Plot 6: Drag breakdown
        ax6 = fig.add_subplot(2, 3, 6)
        for result, color in zip(results_list, colors):
            ax6.plot(result['angles'], result['cd0'], 'o--', linewidth=1.5, alpha=0.7,
                     label=f"{result['name']} (Profile)", color=color)
            ax6.plot(result['angles'], result['cdi'], 's--', linewidth=1.5, alpha=0.7,
                     label=f"{result['name']} (Induced)", color=color)
        ax6.set_xlabel('Angle of Attack (°)', fontsize=10)
        ax6.set_ylabel('Drag Coefficient', fontsize=10)
        ax6.set_title('Drag Coefficient Breakdown', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    @staticmethod
    def plot_morphing_study(base_results: dict, morph_results_list: List[dict],
                            morph_type: str, figsize=(15, 5)):
        """Plot morphing wing performance study"""
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        all_results = [base_results] + morph_results_list
        morph_params = [0] + [float(r['name'].split('_')[-1]) for r in morph_results_list]
        order = np.argsort(morph_params)          # keep the lines monotonic in x

        # Select an angle for comparison (5 degrees)
        angle_idx = np.argmin(np.abs(base_results['angles'] - 5))
        aoa = base_results["angles"][angle_idx]

        panels = [
            ('cl', 'Lift Coefficient', 'bo-', 'Lift'),
            ('cd', 'Drag Coefficient', 'ro-', 'Drag'),
            ('efficiency', 'Efficiency (L/D)', 'go-', 'Efficiency'),
        ]
        for ax, (key, ylabel, style, short) in zip(axes, panels):
            values = [r[key][angle_idx] for r in all_results]
            ax.plot(np.array(morph_params)[order], np.array(values)[order],
                    style, linewidth=2, markersize=8)
            ax.set_xlabel('Morphing Parameter', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f'{short} vs {morph_type} Morphing\n(AoA = {aoa:.1f}°)',
                         fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig


def _banner(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================================
# MODULE 1 - STANDARD AIRFOIL COMPARISON
# ============================================================================

STANDARD_AIRFOILS = [
    ("NACA 2412", (2, 4, 12)),
    ("NACA 4412", (4, 4, 12)),
    ("NACA 0012", (0, 0, 12)),
]


def run_standard_comparison(show: bool = True):
    """Module 1: test the standard NACA airfoils and plot the comparison."""
    _banner("MODULE 1 - STANDARD AIRFOIL COMPARISON")

    conditions = default_conditions()
    print(f"\nWind Tunnel Conditions:")
    print(f"  Velocity:     {conditions.velocity} m/s")
    print(f"  Air Density:  {conditions.air_density} kg/m³")
    print(f"  Chord Length: {conditions.chord_length} m")
    print(f"  Wingspan:     {conditions.wingspan} m")
    print(f"  Wing Area:    {conditions.chord_length * conditions.wingspan} m²")

    simulator = WindTunnelSimulator(conditions)
    angles = DEFAULT_ANGLES

    results = []
    for name, (m, p, t) in STANDARD_AIRFOILS:
        print(f"\n  Testing {name}...")
        airfoil = AirfoilGenerator.naca4(m, p, t)
        result = simulator.test_airfoil(airfoil, name, angles)
        results.append(result)
        print(f"    Reynolds Number: {result['reynolds']:.2e}")
        print(f"    Max CL: {np.max(result['cl']):.4f} at {angles[np.argmax(result['cl'])]}°")
        print(f"    Max Efficiency: {np.max(result['efficiency']):.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig = WindTunnelVisualizer.plot_airfoil_comparison(results)
    png = os.path.join(OUTPUT_DIR, 'airfoil_comparison.png')
    fig.savefig(png, dpi=300, bbox_inches='tight')
    print(f"\n  ✓ Saved: {png}")
    for path in export_results_csv(results, os.path.join(OUTPUT_DIR, 'csv')):
        print(f"  ✓ Saved: {path}")

    _banner("SUMMARY (at 5° angle of attack)")
    idx = np.argmin(np.abs(angles - 5))
    for r in results:
        print(f"\n  {r['name']}:")
        print(f"    CL:  {r['cl'][idx]:.4f}")
        print(f"    CD:  {r['cd'][idx]:.4f}")
        print(f"    L/D: {r['efficiency'][idx]:.2f}")

    if show:
        plt.show()
    plt.close(fig)
    return results


# ============================================================================
# MODULE 2 - MORPHING WING STUDY
# ============================================================================

MORPH_TYPES = ("camber", "twist", "thickness")
MORPH_PARAMS = [-0.5, -0.25, 0.25, 0.5]


def run_morphing_study(morph_type: str = "camber", show: bool = True):
    """Module 2: morph a NACA 4412 and compare against the base airfoil.
    morph_type may be 'camber', 'twist', 'thickness' or 'all'."""
    if morph_type == "all":
        figs = [run_morphing_study(mt, show=False) for mt in MORPH_TYPES]
        if show:
            plt.show()
        for f in figs:
            plt.close(f)
        return figs

    _banner(f"MODULE 2 - MORPHING WING STUDY ({morph_type.upper()})")

    simulator = WindTunnelSimulator(default_conditions())
    angles = DEFAULT_ANGLES

    base_airfoil = AirfoilGenerator.naca4(4, 4, 12)
    base_result = simulator.test_airfoil(base_airfoil.copy(), "Base_Airfoil_0.0", angles)

    morph_results = []
    for param in MORPH_PARAMS:
        print(f"\n  Testing {morph_type} morphing (param={param})...")
        morphed = AirfoilGenerator.morphing_wing(base_airfoil, param, morph_type)
        result = simulator.test_airfoil(morphed, f"Morphed_{morph_type.capitalize()}_{param}", angles)
        morph_results.append(result)
        print(f"    Max CL: {np.max(result['cl']):.4f}")
        print(f"    Max Efficiency: {np.max(result['efficiency']):.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig = WindTunnelVisualizer.plot_morphing_study(
        base_result, morph_results, morph_type.capitalize())
    png = os.path.join(OUTPUT_DIR, f'morphing_wing_study_{morph_type}.png')
    fig.savefig(png, dpi=300, bbox_inches='tight')
    print(f"\n  ✓ Saved: {png}")
    for path in export_results_csv([base_result] + morph_results,
                                   os.path.join(OUTPUT_DIR, 'csv')):
        print(f"  ✓ Saved: {path}")

    _banner("SUMMARY (at 5° angle of attack)")
    idx = np.argmin(np.abs(angles - 5))
    b = base_result
    print(f"\n  Base Airfoil:  CL {b['cl'][idx]:.4f}   CD {b['cd'][idx]:.4f}   "
          f"L/D {b['efficiency'][idx]:.2f}")
    for r in morph_results:
        param = float(r['name'].split('_')[-1])
        print(f"\n  Morphing Parameter = {param}:")
        print(f"    CL:  {r['cl'][idx]:.4f} ({r['cl'][idx] - b['cl'][idx]:+.4f})")
        print(f"    CD:  {r['cd'][idx]:.4f} ({r['cd'][idx] - b['cd'][idx]:+.4f})")
        print(f"    L/D: {r['efficiency'][idx]:.2f} ({r['efficiency'][idx] - b['efficiency'][idx]:+.2f})")

    if show:
        plt.show()
        plt.close(fig)
    return fig


# ============================================================================
# MODULE 3 - INTERACTIVE GUI
# ============================================================================

class InteractiveWindTunnel:
    """Interactive wind tunnel GUI application (built on the shared core)"""

    def __init__(self, root, tk, ttk, messagebox, filedialog, canvas_cls):
        self.tk, self.ttk = tk, ttk
        self.messagebox, self.filedialog = messagebox, filedialog
        self.root = root
        self.root.title("Virtual Wind Tunnel - Interactive Airfoil Tester")
        self.root.geometry("1600x900")

        self.results = []
        self.reference = None
        self.conditions = None
        self.angles = DEFAULT_ANGLES

        self.create_widgets(canvas_cls)
        self.update_plots()

    def create_widgets(self, canvas_cls):
        tk, ttk = self.tk, self.ttk

        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Controls
        left_panel = ttk.LabelFrame(main_container, text="Wind Tunnel Controls", padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))

        # Wind conditions
        conditions_frame = ttk.LabelFrame(left_panel, text="Wind Conditions", padding=10)
        conditions_frame.pack(fill=tk.X, pady=10)

        ttk.Label(conditions_frame, text="Velocity (m/s):").grid(row=0, column=0, sticky=tk.W)
        self.velocity_var = tk.DoubleVar(value=DEFAULT_VELOCITY)
        ttk.Scale(conditions_frame, from_=MIN_VELOCITY, to=MAX_VELOCITY, orient=tk.HORIZONTAL,
                  variable=self.velocity_var,
                  command=lambda x: self.update_plots()).grid(row=0, column=1, sticky=tk.EW)
        self.velocity_label = ttk.Label(conditions_frame, text="50 m/s (M 0.15)", width=16)
        self.velocity_label.grid(row=0, column=2)
        self.velocity_var.trace_add('write', self.update_velocity_label)

        ttk.Label(conditions_frame, text="Chord Length (m):").grid(row=1, column=0, sticky=tk.W)
        self.chord_var = tk.DoubleVar(value=DEFAULT_CHORD)
        ttk.Scale(conditions_frame, from_=0.5, to=3.0, orient=tk.HORIZONTAL,
                  variable=self.chord_var,
                  command=lambda x: self.update_plots()).grid(row=1, column=1, sticky=tk.EW)
        self.chord_label = ttk.Label(conditions_frame, text="1.0")
        self.chord_label.grid(row=1, column=2)
        self.chord_var.trace_add('write', self.update_chord_label)

        ttk.Label(conditions_frame, text="Wingspan (m):").grid(row=2, column=0, sticky=tk.W)
        self.wingspan_var = tk.DoubleVar(value=DEFAULT_SPAN)
        ttk.Scale(conditions_frame, from_=2, to=15, orient=tk.HORIZONTAL,
                  variable=self.wingspan_var,
                  command=lambda x: self.update_plots()).grid(row=2, column=1, sticky=tk.EW)
        self.wingspan_label = ttk.Label(conditions_frame, text="8.0")
        self.wingspan_label.grid(row=2, column=2)
        self.wingspan_var.trace_add('write', self.update_wingspan_label)

        conditions_frame.columnconfigure(1, weight=1)

        # Airfoil selection
        airfoil_frame = ttk.LabelFrame(left_panel, text="Select Airfoils", padding=10)
        airfoil_frame.pack(fill=tk.X, pady=10)

        airfoils = [
            ("NACA 2412", (2, 4, 12)),
            ("NACA 4412", (4, 4, 12)),
            ("NACA 0012", (0, 0, 12)),
            ("NACA 6412", (6, 4, 12)),
        ]
        self.selected_airfoils = {}
        for name, params in airfoils:
            var = tk.BooleanVar(value=name in ("NACA 2412", "NACA 4412"))
            self.selected_airfoils[name] = (var, params)
            ttk.Checkbutton(airfoil_frame, text=name, variable=var,
                            command=self.update_plots).pack(anchor=tk.W)

        # Camber adjustment: offsets the max camber (% chord) of every selected airfoil
        ttk.Separator(airfoil_frame).pack(fill=tk.X, pady=6)
        ttk.Label(airfoil_frame, text="Camber Adjustment (% chord):").pack(anchor=tk.W)
        self.camber_adj_var = tk.DoubleVar(value=0.0)
        ttk.Scale(airfoil_frame, from_=CAMBER_ADJ_MIN, to=CAMBER_ADJ_MAX, orient=tk.HORIZONTAL,
                  variable=self.camber_adj_var,
                  command=lambda x: self.update_plots()).pack(fill=tk.X)
        self.camber_adj_label = ttk.Label(airfoil_frame, text="+0.0  (added to each airfoil's camber)")
        self.camber_adj_label.pack()
        self.camber_adj_var.trace_add('write', self.update_camber_adj_label)

        # Morphing wing section
        morph_frame = ttk.LabelFrame(left_panel, text="Morphing Wing", padding=10)
        morph_frame.pack(fill=tk.X, pady=10)

        ttk.Label(morph_frame, text="Morphing Type:").pack(anchor=tk.W)
        self.morph_type_var = tk.StringVar(value="camber")
        for morph_type in MORPH_TYPES:
            ttk.Radiobutton(morph_frame, text=morph_type.capitalize(), value=morph_type,
                            variable=self.morph_type_var,
                            command=self.update_plots).pack(anchor=tk.W)

        ttk.Label(morph_frame, text="Morphing Parameter:").pack(anchor=tk.W)
        self.morph_param_var = tk.DoubleVar(value=0.0)
        ttk.Scale(morph_frame, from_=-1.0, to=1.0, orient=tk.HORIZONTAL,
                  variable=self.morph_param_var,
                  command=lambda x: self.update_plots()).pack(fill=tk.X)
        self.morph_label = ttk.Label(morph_frame, text="0.00")
        self.morph_label.pack()
        self.morph_param_var.trace_add('write', self.update_morph_label)

        self.morph_enable_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(morph_frame, text="Enable Morphing", variable=self.morph_enable_var,
                        command=self.update_plots).pack(anchor=tk.W)

        # Buttons
        button_frame = ttk.Frame(left_panel)
        button_frame.pack(fill=tk.X, pady=10)
        ttk.Button(button_frame, text="Refresh Plots", command=self.update_plots).pack(fill=tk.X)
        ttk.Button(button_frame, text="Export Data (CSV)",
                   command=self.export_data).pack(fill=tk.X, pady=(5, 0))
        ttk.Button(button_frame, text="Save Plot (PNG)",
                   command=self.save_plot).pack(fill=tk.X, pady=(5, 0))
        ttk.Button(button_frame, text="Clear All",
                   command=self.clear_results).pack(fill=tk.X, pady=(5, 0))

        # Right panel - Plots
        right_panel = ttk.LabelFrame(main_container, text="Aerodynamic Analysis", padding=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.canvas = canvas_cls(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # -- label callbacks -----------------------------------------------------
    def update_velocity_label(self, *args):
        v = self.velocity_var.get()
        self.velocity_label.config(text=f"{v:.0f} m/s (M {v / SPEED_OF_SOUND:.2f})")

    def update_chord_label(self, *args):
        self.chord_label.config(text=f"{self.chord_var.get():.2f}")

    def update_wingspan_label(self, *args):
        self.wingspan_label.config(text=f"{self.wingspan_var.get():.1f}")

    def update_camber_adj_label(self, *args):
        self.camber_adj_label.config(
            text=f"{self.camber_adj_var.get():+.1f}  (added to each airfoil's camber)")

    def update_morph_label(self, *args):
        self.morph_label.config(text=f"{self.morph_param_var.get():.2f}")

    # -- simulation ------------------------------------------------------------
    def update_plots(self):
        """Re-run the simulation for the current selections and redraw."""
        self.conditions = default_conditions(
            velocity=self.velocity_var.get(),
            chord=self.chord_var.get(),
            span=self.wingspan_var.get(),
        )
        simulator = WindTunnelSimulator(self.conditions)
        self.results = []

        camber_adj = self.camber_adj_var.get()
        self.base_params = None          # unadjusted (m, p, t) of the first selected airfoil

        for name, (var, params) in self.selected_airfoils.items():
            if var.get():
                if self.base_params is None:
                    self.base_params = params
                m, p, t = params
                label = name
                if abs(camber_adj) >= 0.05:
                    m = m + camber_adj
                    if p == 0:           # symmetric section has no camber position: use 40%
                        p = 4
                    label = f"{name} ({camber_adj:+.1f}% camber)"
                airfoil = AirfoilGenerator.naca4(m, p, t)
                self.results.append(simulator.test_airfoil(airfoil, label, self.angles))

        if self.morph_enable_var.get() and self.results:
            base_airfoil = self.results[0]['airfoil'].copy()
            morph_param = self.morph_param_var.get()
            morph_type = self.morph_type_var.get()
            morphed = AirfoilGenerator.morphing_wing(base_airfoil, morph_param, morph_type)
            morph_name = f"Morphed_{morph_type.capitalize()}_{morph_param:.2f}"
            self.results.append(simulator.test_airfoil(morphed, morph_name, self.angles))

        # Reference polar at the default tunnel settings, for comparison
        self.reference = None
        if self.results:
            ref_sim = WindTunnelSimulator(default_conditions())
            self.reference = ref_sim.test_airfoil(
                AirfoilGenerator.naca4(*self.base_params), "Reference", self.angles)

        self.plot_results()

    def plot_results(self):
        self.fig.clear()
        if not self.results:
            self.fig.text(0.5, 0.5, "Select at least one airfoil", ha='center', va='center')
            self.canvas.draw()
            return

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.results)))

        ax1 = self.fig.add_subplot(2, 3, 1)  # Airfoil shapes
        ax2 = self.fig.add_subplot(2, 3, 2)  # CL vs AoA
        ax3 = self.fig.add_subplot(2, 3, 3)  # Lift/drag forces
        ax3b = ax3.twinx()
        ax4 = self.fig.add_subplot(2, 3, 4)  # Drag polar
        ax5 = self.fig.add_subplot(2, 3, 5)  # Efficiency
        ax6 = self.fig.add_subplot(2, 3, 6)  # Drag breakdown

        for result, color in zip(self.results, colors):
            airfoil = result['airfoil']
            ax1.plot(airfoil[:, 0], airfoil[:, 1], '-', linewidth=2,
                     label=result['name'], color=color)
            ax1.fill(airfoil[:, 0], airfoil[:, 1], alpha=0.1, color=color)

            ax2.plot(result['angles'], result['cl'], 'o-', linewidth=2,
                     label=result['name'], color=color, markersize=4)
            i_max = int(np.argmax(result['cl']))       # mark CLmax (stall)
            ax2.plot(result['angles'][i_max], result['cl'][i_max], '*', markersize=12,
                     color=color, markeredgecolor='black')

            ax3.plot(result['angles'], result['lift'], '-', linewidth=2,
                     label=f"{result['name']} lift", color=color)
            ax3b.plot(result['angles'], result['drag'], '--', linewidth=1.5,
                      label=f"{result['name']} drag", color=color)

            ax4.plot(result['cd'], result['cl'], 'o-', linewidth=2,
                     label=result['name'], color=color, markersize=4)

            ax5.plot(result['angles'], result['efficiency'], 'o-', linewidth=2,
                     label=result['name'], color=color, markersize=4)

            ax6.plot(result['angles'], result['cd0'], 'o--', linewidth=1.5, alpha=0.7,
                     label=f"{result['name']} (Profile)", color=color, markersize=3)
            ax6.plot(result['angles'], result['cdi'], 's--', linewidth=1.5, alpha=0.7,
                     label=f"{result['name']} (Induced)", color=color, markersize=3)

        if self.reference is not None:
            ax4.plot(self.reference['cd'], self.reference['cl'], ':', linewidth=1.5,
                     color='gray', label='Reference (first airfoil, no adj., 50 m/s, c=1, b=8)')

        ax1.set_xlabel('Chord Position', fontsize=9)
        ax1.set_ylabel('Thickness', fontsize=9)
        ax1.set_title('Airfoil Geometry', fontsize=10, fontweight='bold')
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')

        ax2.set_xlabel('Angle of Attack (°)', fontsize=9)
        ax2.set_ylabel('Lift Coefficient (CL)', fontsize=9)
        ax2.set_title('Lift Coefficient vs AoA (★ = CLmax)', fontsize=10, fontweight='bold')
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

        ax3.set_xlabel('Angle of Attack (°)', fontsize=9)
        ax3.set_ylabel('Lift (N) — solid', fontsize=9)
        ax3b.set_ylabel('Drag (N) — dashed', fontsize=9)
        ax3.set_title('Lift & Drag Forces', fontsize=10, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        ax4.set_xlabel('Drag Coefficient (CD)', fontsize=9)
        ax4.set_ylabel('Lift Coefficient (CL)', fontsize=9)
        ax4.set_title('Drag Polar', fontsize=10, fontweight='bold')
        ax4.legend(fontsize=7)
        ax4.grid(True, alpha=0.3)

        ax5.set_xlabel('Angle of Attack (°)', fontsize=9)
        ax5.set_ylabel('Efficiency (L/D)', fontsize=9)
        ax5.set_title('Aerodynamic Efficiency vs AoA', fontsize=10, fontweight='bold')
        ax5.legend(fontsize=7)
        ax5.grid(True, alpha=0.3)

        ax6.set_xlabel('Angle of Attack (°)', fontsize=9)
        ax6.set_ylabel('Drag Coefficient', fontsize=9)
        ax6.set_title('Drag Coefficient Breakdown', fontsize=10, fontweight='bold')
        ax6.legend(fontsize=6)
        ax6.grid(True, alpha=0.3)

        c = self.conditions
        re = self.results[0]['reynolds']
        q = 0.5 * c.air_density * c.velocity ** 2
        self.fig.suptitle(
            f"V = {c.velocity:.0f} m/s (M {c.velocity / SPEED_OF_SOUND:.2f})   "
            f"chord = {c.chord_length:.2f} m   span = {c.wingspan:.1f} m   "
            f"AR = {c.wingspan / c.chord_length:.1f}   Re = {re:.2e}   q = {q:.0f} Pa",
            fontsize=10)
        self.fig.tight_layout(rect=(0, 0, 1, 0.95))
        self.canvas.draw()

    # -- actions -----------------------------------------------------------------
    def export_data(self):
        """Export results to CSV files"""
        if not self.results:
            self.messagebox.showwarning("No Data", "No simulation results to export.")
            return
        folder = self.filedialog.askdirectory(title="Choose export folder")
        if not folder:
            return
        export_results_csv(self.results, folder)
        self.messagebox.showinfo(
            "Export Complete", f"Data exported for {len(self.results)} airfoil(s)")

    def save_plot(self):
        """Save the current 6-panel figure as a PNG"""
        path = self.filedialog.asksaveasfilename(
            title="Save plot", defaultextension=".png",
            filetypes=[("PNG image", "*.png")], initialfile="wind_tunnel_plot.png")
        if path:
            self.fig.savefig(path, dpi=200, bbox_inches='tight')
            self.messagebox.showinfo("Saved", f"Plot saved to:\n{path}")

    def clear_results(self):
        """Uncheck everything and reset the plots"""
        self.results = []
        for var, _ in self.selected_airfoils.values():
            var.set(False)
        self.morph_enable_var.set(False)
        self.camber_adj_var.set(0.0)
        self.update_plots()


def run_interactive():
    """Module 3: launch the interactive GUI (tkinter is imported lazily so
    modules 1 and 2 work on machines without it)."""
    _banner("MODULE 3 - INTERACTIVE WIND TUNNEL (GUI)")
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox, filedialog
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    except ImportError as exc:
        print(f"\n  ✗ The GUI needs tkinter, which isn't available here ({exc}).")
        print("    Linux: sudo apt install python3-tk   |   Modules 1 and 2 still work.")
        return
    print("\n  Opening window... close it to return.")
    root = tk.Tk()
    InteractiveWindTunnel(root, tk, ttk, messagebox, filedialog, FigureCanvasTkAgg)
    root.mainloop()


# ============================================================================
# LAUNCHER
# ============================================================================

MODULES = {
    "1": ("Standard airfoil comparison (NACA 2412 / 4412 / 0012)", "standard"),
    "2": ("Morphing wing study (camber / twist / thickness)",       "morph"),
    "3": ("Interactive wind tunnel (GUI with live controls)",       "gui"),
}


def _ask_morph_type() -> str:
    choices = "/".join(MORPH_TYPES) + "/all"
    while True:
        ans = input(f"  Morph type [{choices}] (default camber): ").strip().lower()
        if ans == "":
            return "camber"
        if ans in MORPH_TYPES or ans == "all":
            return ans
        print("  Please pick one of:", choices)


def _dispatch(key: str, morph: str = None, show: bool = True):
    kind = MODULES[key][1]
    if kind == "standard":
        run_standard_comparison(show=show)
    elif kind == "morph":
        run_morphing_study(morph or _ask_morph_type(), show=show)
    else:
        run_interactive()


def main():
    parser = argparse.ArgumentParser(description="Virtual Wind Tunnel")
    parser.add_argument("-m", "--module", choices=list(MODULES),
                        help="1 = standard comparison, 2 = morphing study, 3 = interact2ive GUI")
    parser.add_argument("--morph", choices=list(MORPH_TYPES) + ["all"],
                        help="morph type for module 2 (skips the prompt)")
    parser.add_argument("--no-show", action="store_true",
                        help="modules 1/2: save plots without opening windows")
    args = parser.parse_args()

    # Direct launch: run one module and exit
    if args.module:
        _dispatch(args.module, args.morph, show=not args.no_show)
        return

    # Interactive menu: keeps returning here until you quit
    while True:
        print("\n" + "=" * 70)
        print("VIRTUAL WIND TUNNEL")
        print("=" * 70)
        for key, (label, _) in MODULES.items():
            print(f"  [{key}] {label}")
        print("  [q] Quit")
        choice = input("\nSelect a module: ").strip().lower()
        if choice in ("q", "quit", "exit"):
            print("Goodbye!")
            break
        if choice not in MODULES:
            print("  Please enter 1, 2, 3 or q.")
            continue
        _dispatch(choice)


if __name__ == "__main__":
    main()
