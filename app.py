import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Page configuration
st.set_page_config(
    page_title="Differential Equation Optimizer",
    layout="wide"
)

# --- 1. SYMBOLIC COMPUTATION & CACHING ---
# We use st.cache_resource to solve the ODEs only once, not every time a slider moves.
@st.cache_resource
def get_symbolic_solution():
    t = sp.symbols('t', real=True, positive=True)
    x, u = sp.symbols('x u', real=True)
    xp = sp.Function('x_p')(t)
    a = sp.Function('a')(t)
    
    # Parameters
    a0 = sp.symbols('a_0', real=True)
    x0 = sp.symbols('x_0', real=True)
    xm = sp.symbols('x_m', real=True)
    St = sp.symbols('St', real=True)
    alpha = sp.symbols('alpha', real=True)

    # Force term
    F = 1 + alpha * a
    
    # 1. Dynamic Equation: -da/dt = (F/St)*a
    # Rearranged: da/dt = -(1 + alpha*a)*a / St
    dyn_eq = sp.Eq(-sp.diff(a, t), (F/St)*a)
    sol_dyn = sp.dsolve(dyn_eq, a, ics={a.subs(t, 0): a0})
    a_sol = sp.simplify(sol_dyn.rhs)

    # 2. Kinematic Equation: dxp/dt = u - a
    kin_eq = sp.Eq(sp.diff(xp, t), u - a_sol)
    sol_kin = sp.dsolve(kin_eq, xp, ics={xp.subs(t, 0): x0})
    xp_sol = sp.simplify(sol_kin.rhs)

    # 3. Cost Function and Gradient
    # J = 1/2 * |xp - xm|
    J = (1/2) * sp.sqrt((xp_sol - xm)**2)
    dJdalpha = sp.diff(J, alpha)

    # Create numerical functions (lambdify)
    # Args order: t, St, u, a0, x0, xm, alpha
    args = [t, St, u, a0, x0, xm, alpha]
    f_xp = sp.lambdify(args, xp_sol, modules='numpy')
    f_J = sp.lambdify(args, J, modules='numpy')
    f_dJdalpha = sp.lambdify(args, dJdalpha, modules='numpy')
    
    return {
        "a_latex": sp.latex(a_sol),
        "xp_latex": sp.latex(xp_sol),
        "J_latex": sp.latex(J),
        "dJ_latex": sp.latex(dJdalpha),
        "f_J": f_J,
        "f_dJdalpha": f_dJdalpha,
        "f_xp": f_xp
    }

# Load the cached solutions
sol = get_symbolic_solution()

# --- 2. APP LAYOUT ---

st.title("Linear Forcing Optimization & Differential Equations")

# Section 1: Problem Description
st.header("1. Problem Description")
st.markdown("""
This application solves a system of coupled differential equations governing a particle's motion under a forcing term.
We aim to minimize a cost function $J$ with respect to the parameter $\\alpha$.

**The System Equations:**
""")

cols_desc = st.columns(2)
with cols_desc[0]:
    st.markdown("### Dynamics (a)")
    st.latex(r"-\frac{da(t)}{dt} = \frac{1 + \alpha a(t)}{St} a(t)")
    st.latex(r"a(0) = a_0")

with cols_desc[1]:
    st.markdown("### Kinematics (x)")
    st.latex(r"\frac{dx_p(t)}{dt} = u - a(t)")
    st.latex(r"x_p(0) = x_0")

st.markdown("---")

# Section 2: Analytic Solution
st.header("2. Analytic Solutions & Gradient")
st.markdown("Using symbolic mathematics, we derive the exact solutions for the state variables and the cost gradient.")

st.subheader("State Solutions")
st.latex(rf"a(t) = {sol['a_latex']}")
st.latex(rf"x_p(t) = {sol['xp_latex']}")

st.subheader("Cost Function & Gradient")
st.markdown("We define the cost function $J$ as the distance from a target position $x_m$:")
st.latex(r"J = \frac{1}{2} \sqrt{(x_p(t) - x_m)^2} = \frac{1}{2} |x_p(t) - x_m|")
st.markdown("The gradient with respect to $\\alpha$ is:")
st.latex(rf"\frac{{dJ}}{{d\alpha}} = {sol['dJ_latex']}")

st.markdown("---")

# Section 3: Interactive Analysis
st.header("3. Interactive Analysis")
st.markdown("Adjust the parameters below to visualize how the cost function and its gradient change with respect to $\\alpha$.")

# Sidebar for controls
st.sidebar.header("System Parameters")

# Sliders
num_t = st.sidebar.slider("Time (t)", min_value=1.0, max_value=20.0, value=10.0, step=0.5)
num_St = st.sidebar.slider("Stokes Number (St)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
num_u = st.sidebar.slider("Velocity (u)", min_value=-5.0, max_value=5.0, value=1.0, step=0.1)
num_a0 = st.sidebar.slider("Initial a (a0)", min_value=0.1, max_value=5.0, value=2.0, step=0.1)
num_x0 = st.sidebar.slider("Initial Position (x0)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.header("Optimization Target")
# Instead of setting xm manually (which is hard to guess), we set the 'True Alpha'
# and calculate the resulting xm, similar to the original script logic.
true_alpha_input = st.sidebar.slider("Target Alpha (to generate xm)", 0.0, 5.0, 1.0, 0.1)

# Calculate xm based on the "True Alpha"
num_xm = sol['f_xp'](num_t, num_St, num_u, num_a0, num_x0, true_alpha_input)
st.sidebar.info(f"Target Position ($x_m$): {num_xm:.4f}")

# Plotting
alpha_range = np.linspace(0, 8, 500)

# Calculate values
J_vals = sol['f_J'](num_t, num_St, num_u, num_a0, num_x0, num_xm, alpha_range)
grad_vals = sol['f_dJdalpha'](num_t, num_St, num_u, num_a0, num_x0, num_xm, alpha_range)

# Create Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Cost Function
ax1.plot(alpha_range, J_vals, color='blue', linewidth=2, label=r'$J(\alpha)$')
ax1.axvline(true_alpha_input, color='green', linestyle='--', alpha=0.5, label='Target Alpha')
ax1.set_title(r"Cost Function $J$")
ax1.set_xlabel(r"$\alpha$")
ax1.set_ylabel(r"$J$")
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Gradient
ax2.plot(alpha_range, grad_vals, color='red', linewidth=2, label=r'$\frac{dJ}{d\alpha}$')
ax2.axhline(0, color='black', linewidth=1, linestyle='-')
ax2.axvline(true_alpha_input, color='green', linestyle='--', alpha=0.5, label='Target Alpha')
ax2.set_title(r"Gradient $\frac{dJ}{d\alpha}$")
ax2.set_xlabel(r"$\alpha$")
ax2.set_ylabel(r"Gradient Value")
ax2.grid(True, alpha=0.3)
ax2.legend()

st.pyplot(fig)
