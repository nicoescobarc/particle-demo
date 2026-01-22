import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Helper Functions based on Analytical Solutions ---

def get_constants(u_fluid, u_p0):
    """Calculates initial relative velocity magnitude (a0) and direction (n_hat)."""
    rel_vec = u_fluid - u_p0
    a0 = np.linalg.norm(rel_vec)
    
    # Handle case where particle moves exactly with fluid
    if a0 < 1e-9:
        n_hat = np.zeros_like(rel_vec)
    else:
        n_hat = rel_vec / a0
    return a0, n_hat

def compute_a_t(t, k, a0, St=1.0):
    """Computes relative velocity magnitude a(t)."""
    if a0 < 1e-9:
        return 0.0
    
    C1 = a0 / (1 + k * a0)
    exp_term = np.exp(-t / St)
    denom = 1 - k * C1 * exp_term
    
    # Safety check for singularity
    if abs(denom) < 1e-9:
        return 0.0
        
    return (C1 * exp_term) / denom

def compute_position(t, k, u_fluid, x0, a0, n_hat, St=1.0):
    """
    Computes position x_p(t) using the analytical derivation.
    """
    if a0 < 1e-9:
        return x0 + u_fluid * t

    # Avoid division by zero if k is extremely small
    if abs(k) < 1e-9: 
        k = 1e-9

    C1 = a0 / (1 + k * a0)
    
    # Calculate the log term argument
    numerator = 1 - k * C1 * np.exp(-t / St)
    denominator = 1 - k * C1
    
    # Safety check for log domain
    if numerator <= 0 or denominator <= 0:
        return x0 + u_fluid * t
        
    log_arg = numerator / denominator
    log_term = np.log(log_arg)
    
    displacement = u_fluid * t - (St / k) * log_term * n_hat
    return x0 + displacement

def compute_adjoint_integral(t, k, a0, c1, n_hat, St=1.0):
    """
    Computes the adjoint sensitivity integral I(t).
    """
    if a0 < 1e-9 or abs(k) < 1e-9:
        return 0.0

    # 1. Project c1 onto n_hat (dot product)
    c1_dot_n = np.dot(c1, n_hat)
    
    # 2. Get a(t)
    a_t = compute_a_t(t, k, a0, St)
    
    # 3. Compute terms inside the bracket
    val_a = (1 + k * a_t) / (1 + k * a0)
    if val_a <= 0: return 0.0
    term_log = np.log(val_a)
    
    val_b = (1 + k * a0) / (1 + k * a_t)
    term_frac = val_b
    
    # Combine
    prefactor = (St * c1_dot_n) / (k**2)
    result = prefactor * (term_log + term_frac - 1)
    
    return result

# --- Streamlit App UI ---

st.set_page_config(page_title="Adjoint Sensitivity Demo", layout="wide")

st.title("Adjoint Sensitivity Demo")
st.markdown("""
This app demonstrates the **analytical adjoint sensitivity** for a particle in a fluid flow.
It compares a "True" system ($k=4$) against a "Model" system (user guess) and computes the sensitivity integral.
""")

# Constants
St = 1.0
u_fluid = np.array([0.5, 0.5, 0.5])
k_true = 4.0
x0 = np.array([0.0, 0.0, 0.0]) # Assume starting at origin

# Sidebar Inputs
st.sidebar.header("Simulation Parameters")

st.sidebar.subheader("1. Initial Particle Velocity")
col1, col2, col3 = st.sidebar.columns(3)
u0_x = col1.number_input("x", value=0.0)
u0_y = col2.number_input("y", value=0.0)
u0_z = col3.number_input("z", value=0.0)
u_p0 = np.array([u0_x, u0_y, u0_z])

st.sidebar.subheader("2. Settings")
tm = st.sidebar.number_input("Final Time (tm)", value=2.0, min_value=0.1, step=0.1)
k_guess = st.sidebar.number_input("Guess for k", value=1.0, min_value=0.1, step=0.1)

# Main Computation
if st.button("Run Simulation", type="primary"):
    
    # 1. Pre-computation
    a0, n_hat = get_constants(u_fluid, u_p0)
    
    # 2. Forward Simulations
    pos_true = compute_position(tm, k_true, u_fluid, x0, a0, n_hat, St)
    pos_guess = compute_position(tm, k_guess, u_fluid, x0, a0, n_hat, St)
    
    # 3. Compute c1
    c1 = pos_true - pos_guess
    
    # 4. Compute Integral
    integral_value = compute_adjoint_integral(tm, k_guess, a0, c1, n_hat, St)
    
    # --- Results Display ---
    
    st.divider()
    
    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("True Position ($k=4$)")
        st.info(f"{pos_true}")
    with c_right:
        st.subheader(f"Model Position ($k={k_guess}$)")
        st.warning(f"{pos_guess}")
        
    st.divider()
    
    st.subheader("The Forcing Vector ($\mathbf{c}_1$)")
    st.code(f"xdiff = {c1}")
    
    st.subheader("∫ Integral Result")
    st.metric(label="Adjoint Sensitivity Integral", value=f"{integral_value:.6f}")
    
    st.divider()
    
    st.header("🔍 Deep Dive Analysis")
    st.markdown("Explore the physics and sensitivity landscapes through these interactive plots.")
    
    # TABS for the 3 New Plots
    tab1, tab2, tab3 = st.tabs(["Trajectory Divergence", "Error Evolution", "Sensitivity Landscape"])
    
    # --- PLOT 3: DIVERGENCE OF TRAJECTORIES (FAN PLOT) ---
    with tab1:
        st.subheader("Trajectory Divergence (X-Y Projection)")
        st.write("Visualizing how different parameter values ($k$) cause trajectories to fan out over time.")
        
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        
        # Time vector for trajectories
        t_traj = np.linspace(0, tm, 100)
        
        # 1. Plot True Trajectory
        traj_true = np.array([compute_position(t, k_true, u_fluid, x0, a0, n_hat, St) for t in t_traj])
        ax3.plot(traj_true[:,0], traj_true[:,1], 'k-', linewidth=3, label=f'True (k={k_true})')
        
        # 2. Plot User Guess
        traj_guess = np.array([compute_position(t, k_guess, u_fluid, x0, a0, n_hat, St) for t in t_traj])
        ax3.plot(traj_guess[:,0], traj_guess[:,1], 'r--', linewidth=2.5, label=f'Your Guess (k={k_guess})')
        
        # 3. Plot "Ghost" Trajectories (The Fan)
        ghost_ks = [1, 2, 3, 5, 6, 7]
        for k_g in ghost_ks:
            if k_g == k_guess or k_g == k_true: continue # Skip duplicates
            traj_ghost = np.array([compute_position(t, k_g, u_fluid, x0, a0, n_hat, St) for t in t_traj])
            ax3.plot(traj_ghost[:,0], traj_ghost[:,1], 'b-', alpha=0.15) # Faint blue lines
            
        ax3.set_xlabel("X Position")
        ax3.set_ylabel("Y Position")
        ax3.set_title("Trajectory Fan: Effect of Parameter k")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)

    # --- PLOT 4: ERROR MAGNITUDE VS TIME ---
    with tab2:
        st.subheader("Error Magnitude vs. Time")
        st.write("How the distance ($||\mathbf{c}_1||$) between the True particle and your Model particle grows over time.")
        
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        
        # Compute error for a slightly longer range to show drift behavior
        t_err = np.linspace(0, max(tm, 5.0), 100) 
        errors = []
        
        for t_val in t_err:
            p_true = compute_position(t_val, k_true, u_fluid, x0, a0, n_hat, St)
            p_model = compute_position(t_val, k_guess, u_fluid, x0, a0, n_hat, St)
            dist = np.linalg.norm(p_true - p_model)
            errors.append(dist)
            
        ax4.plot(t_err, errors, color='purple', linewidth=2)
        
        # Mark the measurement time tm
        current_error = np.linalg.norm(pos_true - pos_guess)
        ax4.scatter([tm], [current_error], color='red', s=100, zorder=5, label=f'Measurement Time $t_m$')
        ax4.axvline(x=tm, color='gray', linestyle='--', alpha=0.5)
        
        ax4.set_xlabel("Time (t)")
        ax4.set_ylabel("Error Magnitude $||\mathbf{c}_1(t)||$")
        ax4.set_title("Evolution of Prediction Error")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        st.pyplot(fig4)

    # --- PLOT 5: SENSITIVITY LANDSCAPE + GRADIENT ---
    with tab3:
        st.subheader("Sensitivity Landscape & Gradient")
        st.write("The Integral $I(k)$ (Blue) and its Gradient $dI/dk$ (Orange). The gradient represents the 'steepness' of the cost.")
        
        k_range = np.linspace(0.1, 5.0, 200)
        I_vals = []
        
        # Compute I(k) for the whole range
        for k_val in k_range:
            p_k = compute_position(tm, k_val, u_fluid, x0, a0, n_hat, St)
            c1_k = pos_true - p_k # c1 relative to TRUE position
            val = compute_adjoint_integral(tm, k_val, a0, c1_k, n_hat, St)
            I_vals.append(val)
            
        # Compute Gradient numerically
        grad_vals = np.gradient(I_vals, k_range)
        
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        
        # Plot Integral (Left Axis)
        ln1 = ax5.plot(k_range, I_vals, color='tab:blue', linewidth=2, label='Integral I(k)')
        ax5.set_xlabel("Parameter Guess k")
        ax5.set_ylabel("Integral Value", color='tab:blue')
        ax5.tick_params(axis='y', labelcolor='tab:blue')
        
        # Plot Gradient (Right Axis)
        ax5_twin = ax5.twinx()
        ln2 = ax5_twin.plot(k_range, grad_vals, color='tab:orange', linestyle='--', linewidth=2, label='Gradient dI/dk')
        ax5_twin.set_ylabel("Gradient (Sensitivity)", color='tab:orange')
        ax5_twin.tick_params(axis='y', labelcolor='tab:orange')
        
        # User Marker
        ax5.scatter([k_guess], [integral_value], color='red', s=100, zorder=10, label='Your Guess')
        
        # True k line
        ax5.axvline(x=k_true, color='green', linestyle=':', alpha=0.5, label=f'True k={k_true}')
        
        # Combined Legend
        lines = ln1 + ln2
        labels = [l.get_label() for l in lines]
        ax5.legend(lines, labels, loc='upper center')
        
        ax5.set_title("Sensitivity Landscape")
        ax5.grid(True, alpha=0.3)
        
        st.pyplot(fig5)