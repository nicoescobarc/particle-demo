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
    Computes position x_p(t) using the analytical derivation:
    x_p(t) = x0 + u*t - (St/k) * ln(...) * n_hat
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
    Formula: I(t) = (St * (c1 . n) / k^2) * [ ln(...) + ... - 1 ]
    """
    if a0 < 1e-9 or abs(k) < 1e-9:
        return 0.0

    # 1. Project c1 onto n_hat (dot product)
    c1_dot_n = np.dot(c1, n_hat)
    
    # 2. Get a(t)
    a_t = compute_a_t(t, k, a0, St)
    
    # 3. Compute terms inside the bracket
    # Term A: ln( (1 + k*a(t)) / (1 + k*a0) )
    val_a = (1 + k * a_t) / (1 + k * a0)
    if val_a <= 0: return 0.0
    term_log = np.log(val_a)
    
    # Term B: (1 + k*a0) / (1 + k*a(t))
    val_b = (1 + k * a0) / (1 + k * a_t)
    term_frac = val_b
    
    # Combine
    prefactor = (St * c1_dot_n) / (k**2)
    result = prefactor * (term_log + term_frac - 1)
    
    return result

# --- Streamlit App UI ---

st.set_page_config(page_title="Adjoint Sensitivity Demo", layout="wide")

st.title("🌊 Adjoint Sensitivity Demo")
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
    # True System (k=4) - This is the reference "measurement"
    pos_true = compute_position(tm, k_true, u_fluid, x0, a0, n_hat, St)
    
    # Guess System (k=guess)
    pos_guess = compute_position(tm, k_guess, u_fluid, x0, a0, n_hat, St)
    
    # 3. Compute c1 (The Forcing Vector)
    c1 = pos_true - pos_guess
    
    # 4. Compute Integral for the specific user guess
    integral_value = compute_adjoint_integral(tm, k_guess, a0, c1, n_hat, St)
    
    # --- Results Display ---
    
    st.divider()
    
    # Columns for positions
    c_left, c_right = st.columns(2)
    
    with c_left:
        st.subheader("📍 True Position ($k=4$)")
        st.info(f"{pos_true}")
        
    with c_right:
        st.subheader(f"📍 Model Position ($k={k_guess}$)")
        st.warning(f"{pos_guess}")
        
    st.divider()
    
    st.subheader("📉 The Forcing Vector ($\mathbf{c}_1$)")
    st.write("Difference between True and Model positions:")
    st.code(f"c1 = {c1}")
    
    st.divider()
    
    col_metric, col_plot = st.columns([1, 2])
    
    with col_metric:
        st.header("∫ Integral Result")
        st.markdown(r"Computed using the derived formula: $I(t_m)$")
        st.metric(label="Adjoint Sensitivity Integral", value=f"{integral_value:.6f}")
        
    with col_plot:
        st.header("📊 Sensitivity Landscape")
        st.write("Integral value vs. Initial Guess $k$ (Range 0-5)")
        
        # --- PLOTTING LOGIC ---
        
        # Create a range of k values from 0.1 to 5 (avoid 0 singularity)
        k_range = np.linspace(0.1, 5.0, 100)
        integral_results = []
        
        for k_val in k_range:
            # For each k in the plot, we must:
            # 1. Compute where the particle lands with *that* k
            pos_k = compute_position(tm, k_val, u_fluid, x0, a0, n_hat, St)
            
            # 2. Compute the discrepancy vector c1 based on the TRUE position (fixed at k=4)
            c1_k = pos_true - pos_k
            
            # 3. Compute the integral for this k
            val = compute_adjoint_integral(tm, k_val, a0, c1_k, n_hat, St)
            integral_results.append(val)
            
        # Create the Plot
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Plot the curve
        ax.plot(k_range, integral_results, label='Integral I(k)', color='blue', linewidth=2)
        
        # Add the marker for the user's current guess
        ax.scatter([k_guess], [integral_value], color='red', s=100, zorder=5, label=f'Your Guess (k={k_guess})')
        
        # Add a reference line for the "True" k (where error should be 0)
        ax.axvline(x=k_true, color='green', linestyle='--', alpha=0.5, label=f'True k={k_true}')
        
        ax.set_xlabel("Parameter Guess ($k$)")
        ax.set_ylabel("Integral Value")
        ax.set_title("Sensitivity vs Parameter Guess")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        st.pyplot(fig)
    
    # Visualization check
    if a0 < 1e-9:
        st.error("⚠️ Initial relative velocity is zero. No drag forces are active.")