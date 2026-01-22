import streamlit as st
import numpy as np

def get_relative_velocity_params(u_fluid, u_p0, k, St=1.0):
    """
    Computes initial relative parameters:
    a0 (magnitude), n_hat (direction), and C1 (integration constant).
    """
    # Relative velocity vector a_vec = u - u_p
    a_vec_0 = u_fluid - u_p0
    a0 = np.linalg.norm(a_vec_0)
    
    # Handle case where relative velocity is zero
    if a0 < 1e-9:
        return 0.0, np.zeros_like(u_fluid), 0.0
        
    n_hat = a_vec_0 / a0
    C1 = a0 / (1 + k * a0)
    return a0, n_hat, C1

def compute_final_position(tm, k, u_fluid, u_p0, St=1.0):
    """
    Computes the final position x_p(tm) using the analytical solution.
    Assuming x_p(0) = [0, 0, 0].
    """
    a0, n_hat, C1 = get_relative_velocity_params(u_fluid, u_p0, k, St)
    
    if a0 < 1e-9:
        # If relative velocity is zero, particle moves with fluid
        return u_fluid * tm
    
    # Analytical term for the integral of a(t)
    # Integral = (St/k) * ln( (1 - k*C1*exp(-t/St)) / (1 - k*C1) )
    exp_term = np.exp(-tm / St)
    numerator = 1 - k * C1 * exp_term
    denominator = 1 - k * C1
    
    # Avoid log of negative/zero (though physics should prevent this)
    if numerator <= 0 or denominator == 0:
        val_integral = 0
    else:
        val_integral = (St / k) * np.log(numerator / denominator)
    
    # x_p(t) = x_p0 + u*t - n_hat * integral(a)
    # Assuming x_p0 = 0
    x_final = u_fluid * tm - n_hat * val_integral
    return x_final

def compute_integral_term(tm, k, u_fluid, u_p0, c1, St=1.0):
    """
    Computes the integral I(tm) derived in the previous steps.
    """
    a0, n_hat, C1 = get_relative_velocity_params(u_fluid, u_p0, k, St)
    
    if a0 < 1e-9:
        return 0.0

    # Current relative velocity magnitude a(tm)
    exp_term = np.exp(-tm / St)
    a_tm = (C1 * exp_term) / (1 - k * C1 * exp_term)
    
    # Projection of c1 onto flow direction n_hat
    c1_parallel = np.dot(c1, n_hat)
    
    # The Integral Formula
    # I = (St * c1_parallel / k^2) * [ ln((1+k*a(t))/(1+k*a0)) + (1+k*a0)/(1+k*a(t)) - 1 ]
    
    term1 = np.log((1 + k * a_tm) / (1 + k * a0))
    term2 = (1 + k * a0) / (1 + k * a_tm)
    
    I_val = (St * c1_parallel / (k**2)) * (term1 + term2 - 1)
    
    return I_val

# --- STREAMLIT APP LAYOUT ---

st.set_page_config(page_title="Adjoint Sensitivity Demo", layout="wide")

st.title("🌊 Adjoint Sensitivity Demo: Particle in Fluid Flow")
st.markdown("""
This app demonstrates the calculation of position and adjoint sensitivity for a particle moving in a constant fluid flow.
**Fixed Parameters:** Fluid Velocity $\mathbf{u} = (0.5, 0.5, 0.5)$, Stokes Number $St=1$.
""")

# Sidebar Inputs
st.sidebar.header("Simulation Parameters")

st.sidebar.subheader("Initial Particle Velocity u_p(0)")
up0_x = st.sidebar.number_input("x-component", value=0.0)
up0_y = st.sidebar.number_input("y-component", value=0.0)
up0_z = st.sidebar.number_input("z-component", value=0.0)
u_p0 = np.array([up0_x, up0_y, up0_z])

tm = st.sidebar.number_input("Final Time (tm)", value=5.0, min_value=0.1)
k_guess = st.sidebar.number_input("Initial Guess for k", value=1.0, min_value=0.1)

# Fixed Constants
k_true = 4.0
u_fluid = np.array([0.5, 0.5, 0.5])
St = 1.0

# --- COMPUTATIONS ---

if st.button("Run Simulation"):
    # 1. Compute Forward Position with True k=4
    x_true = compute_final_position(tm, k_true, u_fluid, u_p0, St)
    
    # 2. Compute Forward Position with Guess k
    x_guess = compute_final_position(tm, k_guess, u_fluid, u_p0, St)
    
    # Compute c1 (The difference / Error)
    c1 = x_true - x_guess
    
    # 3. Compute the Integral I(tm) using the Guess parameters and c1
    integral_result = compute_integral_term(tm, k_guess, u_fluid, u_p0, c1, St)
    
    # --- DISPLAY RESULTS ---
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Positions")
        st.info(f"**True Final Position (k={k_true}):**\n\n {np.round(x_true, 4)}")
        st.warning(f"**Guessed Final Position (k={k_guess}):**\n\n {np.round(x_guess, 4)}")
        
    with col2:
        st.subheader("📉 Differences & Sensitivity")
        st.error(f"**Error Vector c₁ (True - Guess):**\n\n {np.round(c1, 4)}")
        st.success(f"**Computed Integral Result:**\n\n `{integral_result:.6f}`")
        
    st.markdown("---")
    st.markdown("### Visualization of Inputs")
    st.write(f"Fluid Velocity: `{u_fluid}`")
    st.write(f"Initial Relative Velocity $a_0$: `{np.linalg.norm(u_fluid - u_p0):.4f}`")