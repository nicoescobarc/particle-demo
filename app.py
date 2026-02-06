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
    f_a = sp.lambdify(args, a_sol, modules='numpy')
    f_J = sp.lambdify(args, J, modules='numpy')
    f_dJdalpha = sp.lambdify(args, dJdalpha, modules='numpy')
    
    return {
        "a_latex": sp.latex(a_sol),
        "xp_latex": sp.latex(xp_sol),
        "J_latex": sp.latex(J),
        "dJ_latex": sp.latex(dJdalpha),
        "f_J": f_J,
        "f_dJdalpha": f_dJdalpha,
        "f_xp": f_xp,
        "f_a": f_a
    }

# Load the cached solutions
sol = get_symbolic_solution()

# --- 2. APP LAYOUT ---

st.title("Uniform flow, linear Forcing particle case")

# Section 1: Problem Description
st.header("1. Problem Description")
st.markdown("""
This application solves a system of coupled differential equations governing a particle's motion under a forcing term.
We aim to minimize a cost function $J$ with respect to the parameter $\\alpha$.

**The System Equations:**
""")

cols_desc = st.columns(2)
with cols_desc[0]:
    st.markdown("### Dynamics (to get $a$)")
    st.latex(r"-\frac{\mathrm{d}\,a(t)}{\mathrm{d}t} = \frac{\mathscr{F}}{\mathrm{St}} a(t) = \frac{1 + \alpha \, a(t)}{\mathrm{St}} a(t), \quad a>0.")
    st.latex(r"a(0) = a_0")

with cols_desc[1]:
    st.markdown("### Kinematics (to get $x_p$)")
    st.latex(r"\frac{\mathrm{d}x_p(t)}{\mathrm{d}t} = u - a(t).")
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
st.latex(rf"J = {sol['J_latex']}")
st.markdown("The gradient with respect to $\\alpha$ is:")
st.latex(rf"\frac{{dJ}}{{d\alpha}} = {sol['dJ_latex']}")

st.markdown("---")

# Section 3: Interactive Analysis
st.header("3. Interactive Analysis")
st.markdown("Adjust the parameters below to visualize how the cost function and its gradient change with respect to $\\alpha$.")

# Sidebar for controls
st.sidebar.header("System Parameters")

# Sliders
num_t = st.sidebar.slider("Final time ($t_m$)", min_value=0.1, max_value=20.0, value=5.0, step=0.1)
num_St = st.sidebar.slider("Stokes Number ($\mathrm{St}$)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
num_u = st.sidebar.slider("Velocity ($u$)", min_value=-5.0, max_value=5.0, value=1.0, step=0.1)
num_a0 = st.sidebar.slider("Initial relative velocity ($a_0$)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
num_x0 = st.sidebar.slider("Initial Position ($x_0$)", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.header("Optimization Target")
# Instead of setting xm manually (which is hard to guess), we set the 'True Alpha'
# and calculate the resulting xm, similar to the original script logic.
true_alpha_input = st.sidebar.slider(r"True $\alpha$ ($\alpha_T$ to generate $x_m$)", min_value=0.0, max_value=50.0, value=5.0, step=0.1)

# Calculate xm based on the "True Alpha"
# [t, St, u, a0, x0, xm, alpha]
num_xm = sol['f_xp'](num_t, num_St, num_u, num_a0, num_x0, num_x0, true_alpha_input)
st.sidebar.info(f"Target Position ($x_m$): {num_xm:.4f}")

# --- NEW PLOTS: Time Evolution ---
st.subheader("System Evolution over Time (at Target Alpha)")
# Use the same 't' slider for the final time of the plot
t_array = np.linspace(0, num_t, 200)

# Evaluate state variables over time using the "Target Alpha"
# args: t, St, u, a0, x0, xm, alpha
xp_time_vals = sol['f_xp'](t_array, num_St, num_u, num_a0, num_x0, num_xm, true_alpha_input)
a_time_vals = sol['f_a'](t_array, num_St, num_u, num_a0, num_x0, num_xm, true_alpha_input)

fig_time, (ax_t1, ax_t2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot xp(t)
ax_t1.plot(t_array, xp_time_vals, label=r'$x_p(t)$', color='blue')
ax_t1.set_xlabel('Time (t)')
ax_t1.set_ylabel(r'$x_p$')
ax_t1.set_title(r'Position $x_p$ vs Time')
ax_t1.grid(True, alpha=0.3)
ax_t1.legend()

# Plot a(t)
ax_t2.plot(t_array, a_time_vals, label=r'$a(t)$', color='orange')
ax_t2.set_xlabel('Time (t)')
ax_t2.set_ylabel(r'$a$')
ax_t2.set_title(r'Variable $a$ vs Time')
ax_t2.set_ylim(bottom=0)
ax_t2.grid(True, alpha=0.3)
ax_t2.legend()

st.pyplot(fig_time)
st.markdown("---")

# Plot Configuration
st.subheader("4. Cost & Gradient Plot Settings")
col_plt_1, col_plt_2, col_plt_3 = st.columns(3)

# Plotting


with col_plt_1:
    st.markdown("**Alpha Range (X-axis)**")
    x_min_input = st.number_input("Min Alpha", value=-10.0, step=1.0)
    x_max_input = st.number_input("Max Alpha", value=20.0, step=1.0)

# Calculate values
alpha_range = np.linspace(x_min_input, x_max_input, 1000)
J_vals = sol['f_J'](num_t, num_St, num_u, num_a0, num_x0, num_xm, alpha_range)
grad_vals = sol['f_dJdalpha'](num_t, num_St, num_u, num_a0, num_x0, num_xm, alpha_range)

with col_plt_2:
    st.markdown("**Cost Function (Y-axis)**")
    y_j_min = st.text_input("Min J", value="", placeholder="Auto")
    y_j_max = st.text_input("Max J", value="", placeholder="Auto")

with col_plt_3:
    st.markdown("**Gradient (Y-axis)**")
    y_g_min = st.text_input("Min Grad", value=-0.2)
    y_g_max = st.text_input("Max Grad", value=0.1)


# Create Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

def apply_limits(ax, y_min, y_max):
    try:
        if y_min.strip(): ax.set_ylim(bottom=float(y_min))
        if y_max.strip(): ax.set_ylim(top=float(y_max))
    except (ValueError, AttributeError):
        pass

# Plot 1: Cost Function
ax1.plot(alpha_range, J_vals, color='blue', linewidth=2, label=r'$J(\alpha)$')
ax1.axvline(true_alpha_input, color='green', linestyle='--', alpha=0.5, label=r'True $\alpha$')
ax1.set_title(r"Cost Function $J$")
ax1.set_xlabel(r"$\alpha$")
ax1.set_ylabel(r"$J$")
ax1.set_xlim(x_min_input, x_max_input)
apply_limits(ax1, y_j_min, y_j_max)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Gradient
ax2.plot(alpha_range, grad_vals, color='red', linewidth=2, label=r'$\frac{dJ}{d\alpha}$')
ax2.axhline(0, color='black', linewidth=1, linestyle='-')
ax2.axvline(true_alpha_input, color='green', linestyle='--', alpha=0.5, label=r'True $\alpha$')
ax2.set_title(r"Gradient $\frac{dJ}{d\alpha}$")
ax2.set_xlabel(r"$\alpha$")
ax2.set_ylabel(r"Gradient Value")
ax2.set_xlim(x_min_input, x_max_input)
apply_limits(ax2, y_g_min, y_g_max)
ax2.grid(True, alpha=0.3)
ax2.legend()

st.pyplot(fig)