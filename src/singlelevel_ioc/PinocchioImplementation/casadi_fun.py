import casadi as cs
import pandas as pd
import numpy as np
from pinocchio import casadi as cpin

def make_tau_fun(cmodel):
    q  = cs.SX.sym("q",  cmodel.nq)
    dq = cs.SX.sym("dq", cmodel.nv)
    ddq = cs.SX.sym("ddq", cmodel.nv)

    cdata = cmodel.createData()
    # forward kinematics & dynamics
    cpin.forwardKinematics(cmodel, cdata, q, dq, ddq)
    cpin.crba(cmodel, cdata, q)          # mass matrix M
    cpin.nonLinearEffects(cmodel, cdata, q, dq)  # b(q,dq) = C*qdot + g

    # rnea gives tau directly
    tau = cpin.rnea(cmodel, cdata, q, dq, ddq)

    return cs.Function("tau_fun", [q, dq, ddq], [tau])

def make_com_fun(cmodel):
    q  = cs.SX.sym("q",  cmodel.nq)
    dq = cs.SX.sym("dq", cmodel.nv)
    ddq = cs.SX.sym("ddq", cmodel.nv)

    cdata = cmodel.createData()
    com = cpin.centerOfMass(cmodel, cdata, q)  # 3×1 SX
    vcom = cdata.vcom[0]  # 3×1 SX
    acom = cdata.acom[0]  # 3×1 SX
    
    return cs.Function("com_fun", [q, dq, ddq], [com, vcom, acom])

def make_pinocchio_model(cmodel, tau_fun, com_fun, safety_fun, N, w ):
    """
    N: number of time steps
    model: Pinocchio model (with free-flyer)
    tau_fun, com_fun: CasADi Functions built above
    """
    opti = cs.Opti()
    var = {}

    params = {}
    params['dt']   = opti.parameter(1)
    params['q0']   = opti.parameter(cmodel.nq)
    params['dq0']  = opti.parameter(cmodel.nv)
    params['goal_COM'] = opti.parameter(3)   # or 2 if planar COM
    params['COM_init'] = opti.parameter(3)
    params['T'] = opti.parameter(1)
    dt = params['T'] / (N-1)

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
    safety_list  = []

    for k in range(N):
        qk   = q_full[:, k]
        dqk  = dq_full[:, k]
        ddqk = ddq_full[:, k]

        tau_k  = tau_fun(qk, dqk, ddqk)   # (n,)
        com_k  = com_fun(qk)    # (3,)
        d_k   = safety_fun(qk, dqk, ddqk)

        tau_list.append(tau_k)
        com_list.append(com_k)
        safety_list.append(d_k)

    functions['model_tau'] = cs.horzcat(*tau_list)      # (n, N)
    functions['COM']       = cs.horzcat(*com_list)      # (3, N)
    functions['safety']   = cs.horzcat(*safety_list)

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
    
    # ---- Joint q3 (right hip flexion) bounds ----
    # q3_min = -0.8   # rad (example: extension)
    # q3_max =  0.8   # rad (example: flexion)

    # q3 = variables['q'][2, :]   # q3 over time

    # opti.subject_to(opti.bounded(q3_min, q3, q3_max))
   
    # Add to Opti
    opti.subject_to(constraints['initial_pos'] == 0)
    opti.subject_to(constraints['initial_vel'] == 0)
    opti.subject_to(constraints['dynamics_pos'] == 0)
    opti.subject_to(constraints['dynamics_vel'] == 0)
    opti.subject_to(constraints['com_final']  == 0)

    # opti.subject_to(opti.bounded(-100, variables['dq'], 100))
    # opti.subject_to(opti.bounded(-100, functions['model_tau'], 100))
    opti.subject_to(opti.bounded(-2, variables['q'], 2))

    var['constraints'] = constraints

    costs = {}
    tau = functions['model_tau']

    # Energy cost
    tau_max =  10 # extracted from data with rnea
    scaling_factor_tau = tau_max**2
    costs['energy_cost'] = cs.sumsqr(tau) / N / scaling_factor_tau # scale as you like

    safety_terms = 1/functions['safety']
    costs['safety_cost'] = cs.sumsqr(safety_terms) * (params['dt'] / ( (N-1) ))

    var['costs'] = costs

    # Total cost (weights w_safety, w_energy as parameters or constants)
    w_safety = w[0]
    w_energy = w[1]
    w_total = w_safety + w_energy
    w_safety = w_safety / w_total
    w_energy = w_energy / w_total
    J = w_safety * costs['safety_cost'] + w_energy * costs['energy_cost']
    opti.minimize(J)

    return opti, var


def safety_distance(df, k,
                    left_toe_cols=("Left_foot3.0", "Left_foot3.1", "Left_foot3.2"),
                    right_toe_cols=("Right_foot3.0", "Right_foot3.1", "Right_foot3.2"),
                    com_cols=("com.0", "com.1", "com.2"),
                    forward_axis=1):
    """
    Compute the forward safety distance for a single dataframe row (index k).

    safety(k) = forward_BoS_boundary(k) - COM_forward(k)

    forward_BoS_boundary(k) = max( left_toe_forward(k), right_toe_forward(k) )
    """

    # Extract forward coordinate for left and right toes
    left_fwd = df.loc[k, left_toe_cols[forward_axis]]
    right_fwd = df.loc[k, right_toe_cols[forward_axis]]
    
    # Forward boundary of BoS
    bos_forward = max(left_fwd, right_fwd)

    # COM forward coordinate
    com_fwd = df.loc[k, com_cols[forward_axis]]

    # Safety = boundary - COM
    return bos_forward - com_fwd

def make_safety_fun(cmodel, toe_id):
    print(toe_id)
    q  = cs.SX.sym("q",  cmodel.nq)
    dq = cs.SX.sym("dq", cmodel.nv)
    ddq = cs.SX.sym("ddq", cmodel.nv)

    cdata = cmodel.createData()

    # Forward kinematics
    cpin.forwardKinematics(cmodel, cdata, q, dq, ddq)
    cpin.updateFramePlacements(cmodel, cdata)

    # COM
    com = cpin.centerOfMass(cmodel, cdata, q)

    # Toe frames
    toe = cdata.oMf[toe_id].translation

    # Forward axis = Y
    toe_forward  = toe[1]
    safety_dist  = toe_forward - com[1]     # positive = stable

    return cs.Function("safety_fun", [q, dq, ddq], [safety_dist])



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

