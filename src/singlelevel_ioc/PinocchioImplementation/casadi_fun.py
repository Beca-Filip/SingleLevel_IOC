import casadi as cs
from pinocchio import casadi as cpin
from sympy import var
import pinocchio as pin
import numpy as np


def make_pinocchio_model(cmodel, rnea_fun, com_fun, N, model, data):
    opti = cs.Opti()
    var = {}

    params = {}
    params['dt']   = opti.parameter(1)
    params['q0']   = opti.parameter(cmodel.nq)
    params['dq0']  = opti.parameter(cmodel.nv)
    params['goal_COM'] = opti.parameter(3)   # or 2 if planar COM
    params['COM_init'] = opti.parameter(3)

    # (Optional) other parameters like safety obstacles etc.
    var['parameters'] = params

    n = cmodel.nv  # assume nq == nv

    # ---------- Decision variables ----------
    variables = {}
    variables['q']   = opti.variable(n, N)      # q_0 ... q_{N-1}
    variables['dq']  = opti.variable(n, N-1)    # dq_0 ... dq_{N-2}
    variables['ddq'] = opti.variable(n, N-2)    # ddq_0 ... ddq_{N-3}
    var['variables'] = variables

    # ---------- Functions over trajectory ----------
    functions = {}

    q_full   = variables['q']
    dq_full  = cs.horzcat(variables['dq'],  cs.DM.zeros(n,1))  # pad with zero at final step
    ddq_full = cs.horzcat(variables['ddq'], cs.DM.zeros(n,2))  # pad with zeros at last 2 steps

    # Compute tau for each k
    tau_list   = []
    com_list   = []

    for k in range(N):
        qk   = q_full[:, k]
        dqk  = dq_full[:, k]
        ddqk = ddq_full[:, k]

        tau_k  = rnea_fun(qk, dqk, ddqk)   # (n,)
        com_k  = com_fun(qk)    # (3,)

        tau_list.append(tau_k)
        com_list.append(com_k)

    functions['model_tau'] = cs.horzcat(*tau_list)      # (n, N)
    functions['COM']       = cs.horzcat(*com_list)      # (3, N)

    var['functions'] = functions


    # ---------- Constraints ----------
    constraints = {}

    # initial conditions
    constraints['initial_pos'] = variables['q'][:, 0]   - params['q0']
    constraints['initial_vel'] = variables['dq'][:, 0]  - params['dq0']

    # discrete dynamics (Euler)
    constraints['dynamics_pos'] = (variables['q'][:,1:] - variables['q'][:,:-1]
                                   - variables['dq'] * params['dt'])
    constraints['dynamics_vel'] = (variables['dq'][:,1:] - variables['dq'][:,:-1]
                                   - variables['ddq'] * params['dt'])

    # COM constraints
    com = functions['COM']  # (3,N)
    constraints['com_final']= com[:, -1]  - params['goal_COM']  # or COM_goal param


    # Add to Opti
    opti.subject_to(constraints['initial_pos'] == 0)
    opti.subject_to(constraints['initial_vel'] == 0)
    opti.subject_to(constraints['dynamics_pos'] == 0)
    opti.subject_to(constraints['dynamics_vel'] == 0)
    opti.subject_to(constraints['com_final']  == 0)
    
    opti.subject_to(opti.bounded(-5, variables['dq'], 5))
    opti.subject_to(opti.bounded(-5, functions['model_tau'], 5))

    var['constraints'] = constraints

    costs = {}
    tau = functions['model_tau']

    # Energy cost
    costs['energy_cost'] = cs.sumsqr(tau) * (params['dt'] / ( (N-1) ))  # scale as you like

    # Safety cost: sum over time of (1/d)^2
    safety_terms = []
    for k in range(N-1):
        com_k = functions['COM'][:, k]
        d_k = safety_distance(com_k, model, data)
        safety_terms.append( 1.0 / (d_k**2 + 1e-6) )  # small epsilon

    safety_terms = cs.vertcat(*safety_terms)
    costs['safety_cost'] = cs.sumsqr(safety_terms) * (params['dt'] / ( (N-1) ))

    var['costs'] = costs

    # Total cost (weights w_safety, w_energy as parameters or constants)
    w_safety = 0.1
    w_energy = 0.9
    w_total = w_safety + w_energy
    w_safety = w_safety / w_total
    w_energy = w_energy / w_total
    J = w_safety * costs['safety_cost'] + w_energy * costs['energy_cost']
    opti.minimize(J)

    return opti, var


def safety_distance(com_fun, model, data):
    """
    User-defined function that returns a positive scalar distance 
    to boundaries given joint state (qk, dqk).
    For safety, COM should stay within some limits (e.g., x in [-X_boundary, X_boundary])
    """
    # for double pendulum, only considere y direction 
    q_max_com = np.array([np.pi/2, 0.0])  # initial joint angles
    pin.forwardKinematics(model, data, q_max_com)
    pin.updateFramePlacements(model, data)
    com_max_pos = pin.centerOfMass(model, data, q_max_com, np.zeros(model.nv))
    print(com_fun)
    Y_boundary = 0.9*com_max_pos[1]  # get bos limit
    print("Y_boundary:", Y_boundary)
    return abs(Y_boundary) - com_fun[1] 

def instantiate_pinocchio_model(var, opti, dt, q0, dq0, goal_COM, q_guess, dq_guess, ddq_guess):
    p = var['parameters']
    opti.set_value(p['dt'], dt)
    opti.set_value(p['q0'], q0)
    opti.set_value(p['dq0'], dq0)
    opti.set_value(p['goal_COM'], goal_COM)

    v = var['variables']
    opti.set_initial(v['q'],   q_guess)
    opti.set_initial(v['dq'],  dq_guess)
    opti.set_initial(v['ddq'], ddq_guess)
