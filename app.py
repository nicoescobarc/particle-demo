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
st.sidebar