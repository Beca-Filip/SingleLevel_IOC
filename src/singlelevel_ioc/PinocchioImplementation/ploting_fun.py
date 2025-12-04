import matplotlib.pyplot as plt
import numpy as np
import time
import casadi as cs
from pinocchio.visualize import GepettoVisualizer
from gepetto.corbaserver import Client


def plot_results(t, q_arr, dq_arr, ddq_arr, tau_arr, com_arr, com_goal):
    """
    Plot the results of the simulation or optimization.
    """
    # Plotting
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(10, 9))

    # q
    for i in range(q_arr.shape[0]):
        axes[0].plot(t, q_arr[i, :], label=f"q[{i}]")
    axes[0].set_ylabel("q (rad)")
    axes[0].grid(True)
    axes[0].legend(loc='best', fontsize='small')

    # dq
    for i in range(dq_arr.shape[0]):
        axes[1].plot(t[1:], dq_arr[i, :], label=f"dq[{i}]")
    axes[1].set_ylabel("dq (rad/s)")
    axes[1].grid(True)
    axes[1].legend(loc='best', fontsize='small')

    # ddq
    for i in range(ddq_arr.shape[0]):
        axes[2].plot(t[2:], ddq_arr[i, :], label=f"ddq[{i}]")
    axes[2].set_ylabel("ddq (rad/s²)")
    axes[2].grid(True)
    axes[2].legend(loc='best', fontsize='small')

    # tau
    for i in range(tau_arr.shape[0]):
        axes[3].plot(t, tau_arr[i, :], label=f"tau[{i}]")
    axes[3].set_ylabel("tau (N·m)")
    axes[3].set_xlabel("time (s)")
    axes[3].grid(True)
    axes[3].legend(loc='best', fontsize='small')

    axes[4].plot(t, com_arr[0, :], label="COM x", color='r')
    axes[4].plot(t, com_arr[1, :], label="COM y", color='g')
    axes[4].plot(t, com_arr[2, :], label="COM z", color='b')
    # add horizontal dotted lines for goal COM components (x,y,z)
    axes[4].axhline(y=float(com_goal[0]), color='r', linestyle=':', linewidth=1.5, label='goal COM x')
    axes[4].axhline(y=float(com_goal[1]), color='g', linestyle=':', linewidth=1.5, label='goal COM y')
    axes[4].axhline(y=float(com_goal[2]), color='b', linestyle=':', linewidth=1.5, label='goal COM z')
    axes[4].set_ylabel("COM (m)")
    axes[4].set_xlabel("time (s)")
    axes[4].legend(loc='best', fontsize='small')
    plt.tight_layout()
    plt.show(block=True)


def play_in_gepetto(model, collision_model, visual_model, q_trajectory, goal_com,  com_traj, root,  dt=0.02, window_name="OCP"):
    """
    Play a joint configuration trajectory in Gepetto Viewer.
    Inputs:
        model           - Pinocchio model (pin.Model)
        collision_model - Pinocchio collision model (pin.GeometryModel)
        visual_model    - Pinocchio visual model (pin.GeometryModel)
        q_trajectory    - array of shape (nq, T)
        dt              - playback timestep (viewer update speed)
    """

    viz = GepettoVisualizer(model, collision_model, visual_model)
    try:
        viz.initViewer(loadModel=True)
    except Exception as e:
        print("Could not initialize Gepetto Viewer. Is gepetto-gui running?")
        raise e

    viz.display(q_trajectory[:,0])
    gui = Client().gui
    gv = viz.viewer.gui

    if not gui.nodeExists(f"{root}/floor"):
        gui.addFloor(f"{root}/floor")
        gui.addToGroup(f"{root}/floor", root)

    gui.setVisibility(f"{root}/floor", "ON")
    gui.setColor(f"{root}/floor", [0.3, 0.3, 0.3, 1.0])
    gui.refresh()

    com_name = add_com_sphere(viz)
    
    # Create goal COM sphere (green & static)
    goal_sphere_name = add_static_sphere(
        viz,
        name="world/goalCOM",
        position=goal_com,
        radius=0.06,
        color=(0,1,0,1)     # green
    )

    print("\n▶ Playing optimized motion in Gepetto Viewer...\n")

    # Smooth playback
    for k in range(q_trajectory.shape[1]):
        qk = q_trajectory[:, k]
        com_k = com_traj[:,k]   # 3D position

        viz.display(qk)

         # update COM sphere pose (x,y,z, quaternion)
        gv.applyConfiguration(
            com_name,
            [float(com_k[0]), float(com_k[1]), float(com_k[2]), 0,0,0,1]
        )

        time.sleep(dt)

    print("\n✓ Playback finished.\n")


def add_com_sphere(viz, com_radius=0.005, color=(1,1,0,1)):
    """
    Add a COM sphere to Gepetto Viewer.
    """
    com_name = "world/COM_sphere"

    # Mass display: create sphere node
    gv = viz.viewer
    gv.gui.addSphere(com_name, com_radius, color)
    gv.gui.setVisibility(com_name, "ON")

    # Add it to the world (not attached to robot kinematics)
    viz.viewer.gui.applyConfiguration(com_name, [0,0,0,0,0,0,1])

    return com_name

def add_static_sphere(viz, name, position, radius=0.005, color=(0,1,0,1)):
    """
    Add a static sphere to Gepetto Viewer.

    Args:
      viz: GepettoVisualizer instance
      name: string name for the sphere (e.g. "world/goalCOM")
      position: 3D position [x, y, z]
      radius: sphere radius
      color: (r,g,b,a)
    """
    gv = viz.viewer.gui

    # Create and color the sphere
    gv.addSphere(name, radius, color)
    gv.setVisibility(name, "ON")

    # place at given world position (identity quaternion)
    gv.applyConfiguration(
        name,
        [float(position[0]), float(position[1]), float(position[2]),
         0, 0, 0, 1]
    )

    return name
    