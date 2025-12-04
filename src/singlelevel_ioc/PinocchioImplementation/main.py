#!/usr/bin/env python3
import numpy as np
import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin
from pinocchio.visualize import GepettoVisualizer
import matplotlib.pyplot as plt
import casadi_fun as csf
from pinocchio.visualize import GepettoVisualizer
from gepetto.corbaserver import Client
import ploting_fun as pf
import numpy as _np

# Ensure shapes are (nq, T) / (nv, T)
def ensure_shape(arr, n_dof):
    arr = np.atleast_1d(arr)
    if arr.ndim == 1:
        if arr.size % n_dof == 0:
            arr = arr.reshape(n_dof, -1)
        else:
            arr = arr.reshape(n_dof, -1, order='F')
    elif arr.ndim == 2:
        if arr.shape[0] != n_dof and arr.shape[1] == n_dof:
            arr = arr.T
    else:
        arr = arr.reshape(n_dof, -1)
    return arr



# ---------------------------------------------------------------------------
# 1) Load URDF and build Pinocchio + CasADi dynamics
# ---------------------------------------------------------------------------
root = "world"

urdf_path = "double_pendulum_simple.urdf"
# 1) Pinocchio model
model = pin.buildModelFromUrdf(urdf_path)

# 2) Geometry models
collision_model = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.COLLISION)
visual_model    = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.VISUAL)

data = model.createData()
# We assume fixed-base model here so nq = nv = number of joints
print("nq, nv =", model.nq, model.nv)
assert model.nq == model.nv, "This example assumes a fixed-base model (nq == nv)."
nq = model.nq
nv = model.nv

# CasADi version of the model
cmodel = cpin.Model(model)
cdata = cpin.Data(cmodel)

# CasADi symbols for state variables
q  = ca.SX.sym("q", nq)
dq = ca.SX.sym("dq", nv)
ddq = ca.SX.sym("ddq", nv)

# Torque computed via RNEA
tau = cpin.rnea(cmodel, cdata, q, dq, ddq)

# COM (whole robot)
com = cpin.centerOfMass(cmodel, cdata, q)  # shape (3,)

# Wrap as CasADi Functions
rnea_fun = ca.Function("rnea", [q, dq, ddq], [tau])
com_fun  = ca.Function("com",  [q], [com])

N = 120
T = 1
dt = T / N


q0_meas = np.array([np.pi/2, 0.0])  # initial joint angles
dq0_meas = np.zeros(nv)

q_goal = np.array([0.0, 1.57])

# Set the model and data to the current configuration and velocity
pin.forwardKinematics(model, data, q_goal)
pin.updateFramePlacements(model, data)

# Compute the center of mass for q_goal
com_goal = pin.centerOfMass(model, data, q_goal, dq0_meas)

q_init = np.array((np.linspace(1.5708, 2.0399, N), np.linspace(0, -2.2524, N)))  # initial guess
dq_init = np.diff(q_init, axis=1) / dt
ddq_init = np.diff(dq_init, axis=1) / dt 


opti, var = csf.make_pinocchio_model(cmodel, rnea_fun, com_fun, N, model, data)

csf.instantiate_pinocchio_model(
    var=var,
    opti=opti,
    dt=dt,
    q0=q0_meas,
    dq0=dq0_meas,
    goal_COM=com_goal,
    q_guess=q_init,
    dq_guess=dq_init,
    ddq_guess=ddq_init
)

opti.solver('ipopt')
sol = opti.solve()

q_sol   = sol.value(var['variables']['q'])
dq_sol  = sol.value(var['variables']['dq'])
ddq_sol = sol.value(var['variables']['ddq'])
tau_sol = sol.value(var['functions']['model_tau'])

# Prepare arrays
q_arr = np.array(q_sol)
dq_arr = np.array(dq_sol)
ddq_arr = np.array(ddq_sol)
tau_arr = np.array(tau_sol)

q_arr   = ensure_shape(q_arr, nq)
dq_arr  = ensure_shape(dq_arr, nq)
ddq_arr = ensure_shape(ddq_arr, nq)
tau_arr = ensure_shape(tau_arr, nv)

# Set the model and data to the current configuration and velocity
com_arr = np.zeros((3, q_arr.shape[1]))
for i in range(0, q_arr.shape[1]-1):
    q_i = q_arr[:, i]
    dq_i = dq_arr[:, i]
    pin.forwardKinematics(model, data, q_i, dq_i)
    pin.updateFramePlacements(model, data)
    com_i = pin.centerOfMass(model, data, q_i, dq_i)
    com_arr[:, i] = com_i

# Time vector (use solver timestep if available)
T_len = q_arr.shape[1]
t = np.arange(T_len) * dt

# Plot results
pf.plot_results(t, q_arr, dq_arr, ddq_arr, tau_arr, com_arr, com_goal)

# Visualization Pinocchio + Gepetto Viewer
# Make sure q has shape (nq, T)
q_sol = np.array(q_sol)
lam_g = np.array(sol.value(opti.lam_g)).flatten()

# Optional: set playback dt (this is NOT the simulation dt)
viewer_dt = 0.03
pf.play_in_gepetto(
    model,
    collision_model,
    visual_model,
    q_sol,
    com_goal,
    com_arr,
    root,
    dt=viewer_dt
)

#### IOC Computation, ######

[opti_ioc, vars_ioc] = csf.make_pinocchio_model(cmodel, rnea_fun, com_fun, N, model, data)

nparam = 2 

# Initialize lambda_ioc_init based on lam_g from previous solve
# We need to extract the relevant dual variables corresponding to the constraints
# in the IOC formulation.
n_init_pos = nq           # 2
n_init_vel = nv           # 2
n_dyn_pos  = nq * (N-1)   # 2*(N-1)
n_dyn_vel  = nv * (N-2)   # 2*(N-2)
n_com_final= 3

n_eq = n_init_pos + n_init_vel + n_dyn_pos + n_dyn_vel + n_com_final
lam_eq = lam_g[:n_eq]

lam_init_pos   = lam_eq[0:n_init_pos]
lam_init_vel   = lam_eq[n_init_pos:n_init_pos+n_init_vel]
lam_dyn_pos    = lam_eq[n_init_pos+n_init_vel : n_init_pos+n_init_vel+n_dyn_pos]
lam_dyn_vel    = lam_eq[n_init_pos+n_init_vel+n_dyn_pos :
                        n_init_pos+n_init_vel+n_dyn_pos+n_dyn_vel]
lam_com_final  = lam_eq[-3:]

lambda_ioc_init = np.concatenate([
    lam_init_pos,
    lam_init_vel,
    lam_dyn_pos,
    lam_dyn_vel,
    lam_com_final,  
])

# Create model parameter
vars_ioc["variables"]["theta"] = opti_ioc.variable(nparam)

# Prepare stationarity constraint
vars_ioc["costs"]["compound_cost"] = vars_ioc["variables"]["theta"][0] * vars_ioc["costs"]["safety_cost"] + vars_ioc["variables"]["theta"][1] * vars_ioc["costs"]["energy_cost"] 
q_vec = ca.vec(vars_ioc["variables"]["q"])
dq_vec = ca.vec(vars_ioc["variables"]["dq"])
ddq_vec = ca.vec(vars_ioc["variables"]["ddq"])

all_vars = ca.vertcat(q_vec, dq_vec, ddq_vec)

# -----------------------------
# Compute gradient of compound cost w.r.t all variables
# -----------------------------
grad_compound_cost = ca.jacobian(vars_ioc["costs"]["compound_cost"], all_vars).T  
vars_ioc["costs"]["grad_compound_cost"] = grad_compound_cost


init_pos = ca.vec(vars_ioc["constraints"]["initial_pos"]) 
init_vel = ca.vec(vars_ioc["constraints"]["initial_vel"])
dynamics_pos = ca.vec(vars_ioc["constraints"]["dynamics_pos"])
dynamics_vel = ca.vec(vars_ioc["constraints"]["dynamics_vel"])
goal_com = ca.vec(vars_ioc["constraints"]["com_final"])

# Concatenate all constraints
compound_constraints = ca.vertcat(init_pos, init_vel, dynamics_pos, dynamics_vel, goal_com)
n_constraints = compound_constraints.size1()   # should be 486

# Create dual variable parameter
vars_ioc["variables"]["lambda"] = opti_ioc.variable(n_constraints)

vars_ioc["constraints"]["compound_constraints"] = compound_constraints

# Compute gradient of compound constraints w.r.t all variables (use the original all_vars)
vars_ioc["constraints"]["grad_compound_constraints"] = ca.jacobian(compound_constraints, all_vars)

vars_ioc["constraints"]["stationarity"] = vars_ioc["costs"]["grad_compound_cost"] + vars_ioc["constraints"]["grad_compound_constraints"].T @ vars_ioc["variables"]["lambda"]



# 1. Stationarity constraint
opti_ioc.subject_to(vars_ioc["constraints"]["stationarity"] == 0)

# 2. Theta sum equals 1
opti_ioc.subject_to(ca.sum1(vars_ioc["variables"]["theta"]) == 1)

# 3. Theta non-negativity
opti_ioc.subject_to(vars_ioc["variables"]["theta"] >= 0)

# 4. Joint limits around q_1
q = vars_ioc["variables"]["q"]

# Flatten q - q_1
q_diff = ca.vec(q - q_arr)
opti_ioc.subject_to(q_diff <= np.pi / 4)
opti_ioc.subject_to(-q_diff <= np.pi / 4)

# -----------------------------
# Create L2 loss
# -----------------------------
q_diff = vars_ioc["variables"]["q"] - q_arr
vars_ioc["costs"]["L2_loss"] = ca.sumsqr(q_diff)  # sumsqr does exactly sum(sum(...^2))

# -----------------------------
# Minimize
# -----------------------------
opti_ioc.minimize(vars_ioc["costs"]["L2_loss"])

# -----------------------------
# Instantiate model (set parameter values and initial guesses)
# -----------------------------

# Compute velocities/accelerations from q_arr (ndarray)
dq_guess = np.diff(q_arr, axis=1) / dt
ddq_guess = np.diff(dq_guess, axis=1) / dt

csf.instantiate_pinocchio_model(
    var=vars_ioc,
    opti=opti_ioc,
    dt=dt,
    q0=q0_meas,
    dq0=dq0_meas,
    goal_COM=com_goal,
    q_guess=q_arr,
    dq_guess=dq_guess,
    ddq_guess=ddq_guess
)

# -----------------------------
# Set initial guesses for duals / theta
# -----------------------------
opti_ioc.set_initial(vars_ioc["variables"]["lambda"], lambda_ioc_init)
opti_ioc.set_initial(vars_ioc["variables"]["theta"], np.array([0.2, 0.8]))

# -----------------------------
# Solver options and solve
# -----------------------------
opti_ioc.solver("ipopt",
    {
        "ipopt.print_level": 5,
        "ipopt.max_iter": 5000,
        "ipopt.tol": 1e-4,
        "ipopt.acceptable_tol": 5e-3,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.hessian_approximation": "limited-memory",
        "ipopt.limited_memory_max_history": 20,
    }
)
sol_ioc = opti_ioc.solve()

q_sol   = sol_ioc.value(vars_ioc['variables']['q'])
theta_id = np.array(sol_ioc.value(vars_ioc["variables"]["theta"])).flatten()

print("Identified theta:", theta_id)
# Save identified trajectory and parameters to CSV files

pf.play_in_gepetto(
    model,
    collision_model,
    visual_model,
    q_sol,
    com_goal,
    com_arr,
    root,
    dt=viewer_dt
)

