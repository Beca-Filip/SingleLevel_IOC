import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from roboticstoolbox import DHRobot, RevoluteMDH, PrismaticDH

class Arm4dof:
    def __init__(self, L, M, I, COM, alpha, offset, n):
        self.n = n  # Number of joints
        self.q = np.zeros(self.n)  # Joint angles in radians
        self.qd = np.zeros(self.n) # Joint velocities in rad/s
        self.q_null = np.zeros(self.n) # Desired Nullspace position
        self.W_null = np.eye(self.n) # Weight matrix for nullspace
        self.L = L
        self.M = M
        self.I = I
        self.COM = COM
        self.alpha = alpha
        self.offset = offset    
    
        

    def create_DH_model(self):
        
        list_links = []
        
        print(self.n)
        for i in np.arange(0,self.n):
            L = RevoluteMDH(a=self.L[i], m=self.M[i], alpha=self.alpha[i], r=self.COM[:,i], inertia=self.I[i], modified=True, offset=self.offset[i])
            list_links.append(L)
        #To show last segment
        L_last = RevoluteMDH(a=0, m=0, alpha=0, r=[0, 0, 0], modified=True, offset=0)
        list_links.append(L_last)  
        return DHRobot(list_links, name="robot_arm")
       
    def get_dh_params(self):
        n = self.n
        dh = np.zeros((n, 3))
        L_arr = np.asarray(self.L)
        alpha_arr = np.asarray(self.alpha)

        if L_arr.size == n and alpha_arr.size == n:
            dh[:, 0] = L_arr
            dh[:, 1] = alpha_arr
        elif L_arr.size == n - 1 and alpha_arr.size == n - 1:
            # first row stays zeros (base link), fill remaining rows
            dh[1:, 0] = L_arr
            dh[1:, 1] = alpha_arr
        else:
            raise ValueError("Lengths of self.L and self.alpha must be either n or n-1")

        # last column remains zeros
        return dh
    
    
    
    def compute_mass_matrix(self, model, q, eps=1e-8):
        """Try model.inertia(q) first; if not available, compute M by RNE trick."""
        n = len(q)
        try:
            M = model.inertia(q)   # robotics-toolbox style inertia matrix
            return np.asarray(M)
        except Exception:
            M = np.zeros((n,n))
            zero_qd = np.zeros(n)
            zero_grav = [0,0,0]
            for i in range(n):
                qdd = np.zeros(n); qdd[i] = 1.0
                tau = np.asarray(model.rne(q, zero_qd, qdd, gravity=zero_grav))
                M[:, i] = tau
            return M

    def f_state(self,x, u, model, gravity):
        """Compute f(x,u) = [ qdot; M^{-1}(u - h(q,qdot)) ]"""
        n = len(x)//2
        q = x[:n]
        qd = x[n:]
        # h = C(q,qd)*qd + G(q) -> RNE with zero qdd
        h = np.asarray(model.rne(q, qd, np.zeros(n), gravity=gravity))
        M = self.compute_mass_matrix(model, q)
        # ensure invertible
        Minv = np.linalg.inv(M)
        qdd = Minv.dot(u - h)
        return np.concatenate([qd, qdd])

    def linearize_finite_diff(self, model, q0, qd0, u0, gravity=[0,0,9.81], eps=1e-6):
        n = len(q0)
        x0 = np.hstack([q0, qd0])
        m = 2*n
        # compute nominal f
        f0 = self.f_state(x0, u0, model, gravity)
        # allocate
        A = np.zeros((m, m))
        B = np.zeros((m, n))
        # central differences for A (w.r.t. each state element)
        for i in range(m):
            dx = np.zeros(m)
            dx[i] = eps
            f_plus = self.f_state(x0 + dx, u0, model, gravity)
            f_minus = self.f_state(x0 - dx, u0, model, gravity)
            A[:, i] = (f_plus - f_minus) / (2*eps)
        # central differences for B (w.r.t. each input torque)
        for j in range(n):
            du = np.zeros(n); du[j] = eps
            f_plus = self.f_state(x0, u0 + du, model, gravity)
            f_minus = self.f_state(x0, u0 - du, model, gravity)
            B[:, j] = (f_plus - f_minus) / (2*eps)
        return A, B