\documentclass[11pt]{article}

\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{bm}
\usepackage{geometry}
\geometry{margin=1in}

\title{Replacing the FAST-LIO2 Kinematic Model with a Fossen-Style Dynamic Model}
\author{}
\date{}

\begin{document}
\maketitle
% -----------------------------------------------------------
% Underwater Vehicle State, Dynamics and Measurement Models
% -----------------------------------------------------------

\subsection{Underwater Vehicle Dynamics in Fossen Form}

We consider an underwater vehicle with
\begin{itemize}
  \item Earth-fixed (inertial) frame $\{G\}$,
  \item Body-fixed frame attached to the IMU/vehicle $\{B\}$,
  \item LiDAR frame $\{L\}$.
\end{itemize}

The pose of the vehicle is
\begin{equation}
  \boldsymbol{\eta}
  =
  \begin{bmatrix}
    \mathbf{p}^G_B \\
    \boldsymbol{\Theta}
  \end{bmatrix}
  \in \mathbb{R}^6,
  \qquad
  \mathbf{p}^G_B = [x, y, z]^\top,
  \quad
  \boldsymbol{\Theta} = [\phi,\theta,\psi]^\top,
\end{equation}
and the generalized velocities in the body frame are
\begin{equation}
  \boldsymbol{\nu}
  =
  \begin{bmatrix}
    \mathbf{v}^B \\
    \boldsymbol{\omega}^B
  \end{bmatrix}
  \in \mathbb{R}^6,
  \qquad
  \mathbf{v}^B = [u,v,w]^\top,
  \quad
  \boldsymbol{\omega}^B = [p,q,r]^\top .
\end{equation}

\subsubsection{Kinematics}

The kinematic equations are
\begin{equation}
  \dot{\boldsymbol{\eta}} = \mathbf{J}(\boldsymbol{\eta}) \, \boldsymbol{\nu},
  \label{eq:fossen_kinematics_eta}
\end{equation}
with
\begin{equation}
  \mathbf{J}(\boldsymbol{\eta})
  =
  \begin{bmatrix}
    \mathbf{R}^G_B(\boldsymbol{\Theta}) & \mathbf{0}_{3\times 3} \\
    \mathbf{0}_{3\times 3} & \mathbf{T}(\boldsymbol{\Theta})
  \end{bmatrix},
\end{equation}
where $\mathbf{R}^G_B(\boldsymbol{\Theta}) \in SO(3)$ is the rotation matrix from body to global frame and
$\mathbf{T}(\boldsymbol{\Theta})$ is the standard mapping from body angular rates to Euler angle rates.

In the FAST-LIO/IKFOM formulation, we represent attitude directly on $SO(3)$ via
$\mathbf{R}^G_B \in SO(3)$ and write the kinematics as
\begin{align}
  \dot{\mathbf{R}}^G_B &= \mathbf{R}^G_B [\boldsymbol{\omega}^B]_\times,
  \label{eq:kin_R}\\
  \dot{\mathbf{p}}^G_B &= \mathbf{R}^G_B \, \mathbf{v}^B,
  \label{eq:kin_p}
\end{align}
where $[\cdot]_\times$ denotes the skew-symmetric matrix.

\subsubsection{Fossen Dynamics}

The 6-DOF underwater vehicle dynamics in Fossen form are
\begin{equation}
  \mathbf{M}(\boldsymbol{\nu})\,\dot{\boldsymbol{\nu}}
  +
  \mathbf{C}(\boldsymbol{\nu})\,\boldsymbol{\nu}
  +
  \mathbf{D}(\boldsymbol{\nu})\,\boldsymbol{\nu}
  +
  \mathbf{g}(\boldsymbol{\eta})
  =
  \boldsymbol{\tau}
  +
  \mathbf{w}_\tau,
  \label{eq:fossen_dyn}
\end{equation}
where
\begin{itemize}
  \item $\mathbf{M}(\boldsymbol{\nu}) \in \mathbb{R}^{6\times 6}$ is the inertia + added mass matrix,
  \item $\mathbf{C}(\boldsymbol{\nu})$ is the Coriolis and centripetal matrix,
  \item $\mathbf{D}(\boldsymbol{\nu})$ is the hydrodynamic damping matrix,
  \item $\mathbf{g}(\boldsymbol{\eta})$ is the restoring (gravity + buoyancy) vector,
  \item $\boldsymbol{\tau} \in \mathbb{R}^6$ is the generalized force/torque input from thrusters etc.,
  \item $\mathbf{w}_\tau$ is process noise representing unmodelled effects.
\end{itemize}

Solving for $\dot{\boldsymbol{\nu}}$ gives
\begin{equation}
  \dot{\boldsymbol{\nu}}
  =
  \mathbf{M}^{-1}(\boldsymbol{\nu})
  \big(
    \boldsymbol{\tau}
    -
    \mathbf{C}(\boldsymbol{\nu})\boldsymbol{\nu}
    -
    \mathbf{D}(\boldsymbol{\nu})\boldsymbol{\nu}
    -
    \mathbf{g}(\boldsymbol{\eta})
    +
    \mathbf{w}_\tau
  \big).
  \label{eq:nu_dot}
\end{equation}

Writing $\boldsymbol{\nu} = [\mathbf{v}^B; \boldsymbol{\omega}^B]$, we obtain
\begin{equation}
  \begin{bmatrix}
    \dot{\mathbf{v}}^B \\
    \dot{\boldsymbol{\omega}}^B
  \end{bmatrix}
  =
  \mathbf{M}^{-1}(\boldsymbol{\nu})
  \big(
    \boldsymbol{\tau}
    -
    \mathbf{C}(\boldsymbol{\nu})\boldsymbol{\nu}
    -
    \mathbf{D}(\boldsymbol{\nu})\boldsymbol{\nu}
    -
    \mathbf{g}(\boldsymbol{\eta})
    +
    \mathbf{w}_\tau
  \big).
  \label{eq:vb_omega_dot}
\end{equation}

\subsection{State Vector and Process Model for FAST-LIO with Fossen Dynamics}

We define the state on the manifold
\begin{equation}
  \mathcal{M}_d
  =
  SO(3)
  \times \mathbb{R}^3
  \times \mathbb{R}^3
  \times \mathbb{R}^3
  \times \mathbb{R}^3
  \times \mathbb{R}^3
  \times \mathbb{R}^3
  \times SO(3)
  \times \mathbb{R}^3,
\end{equation}
as
\begin{equation}
  \mathbf{x}
  =
  \big(
    \mathbf{R}^G_B,\;
    \mathbf{p}^G_B,\;
    \mathbf{v}^B,\;
    \boldsymbol{\omega}^B,\;
    \mathbf{b}_\omega,\;
    \mathbf{b}_a,\;
    \mathbf{g}^G,\;
    \mathbf{R}^B_L,\;
    \mathbf{p}^B_L
  \big)^\top
  \in \mathcal{M}_d,
  \label{eq:state_def}
\end{equation}
where
\begin{itemize}
  \item $\mathbf{R}^G_B \in SO(3)$ is the rotation from body to global frame,
  \item $\mathbf{p}^G_B \in \mathbb{R}^3$ is the body position,
  \item $\mathbf{v}^B \in \mathbb{R}^3$ and $\boldsymbol{\omega}^B \in \mathbb{R}^3$ are body-frame linear and angular velocities,
  \item $\mathbf{b}_\omega, \mathbf{b}_a \in \mathbb{R}^3$ are gyroscope and accelerometer biases,
  \item $\mathbf{g}^G \in \mathbb{R}^3$ is the gravity vector in the global frame,
  \item $\mathbf{R}^B_L \in SO(3)$, $\mathbf{p}^B_L \in \mathbb{R}^3$ are LiDAR extrinsics.
\end{itemize}

Let the input be the generalized forces
\begin{equation}
  \mathbf{u} = \boldsymbol{\tau},
\end{equation}
and the process noise be
\begin{equation}
  \mathbf{w}
  =
  \begin{bmatrix}
    \mathbf{w}_\nu^\top &
    \mathbf{n}_{b\omega}^\top &
    \mathbf{n}_{ba}^\top
  \end{bmatrix}^\top,
\end{equation}
with $\mathbf{w}_\nu$ corresponding to $\mathbf{w}_\tau$ and $\mathbf{n}_{b\omega}, \mathbf{n}_{ba}$ being random-walk noise on the biases.

The continuous-time process model is
\begin{equation}
  \dot{\mathbf{x}} = \mathbf{f}_d(\mathbf{x}, \mathbf{u}, \mathbf{w}),
  \label{eq:process_model_cont}
\end{equation}
with components
\begin{align}
  \dot{\mathbf{R}}^G_B &= \mathbf{R}^G_B [\boldsymbol{\omega}^B]_\times,
  \label{eq:f_d_R}\\
  \dot{\mathbf{p}}^G_B &= \mathbf{R}^G_B \, \mathbf{v}^B,
  \label{eq:f_d_p}\\
  \begin{bmatrix}
    \dot{\mathbf{v}}^B \\
    \dot{\boldsymbol{\omega}}^B
  \end{bmatrix}
  &= \mathbf{M}^{-1}(\boldsymbol{\nu})
  \big(
    \boldsymbol{\tau}
    -
    \mathbf{C}(\boldsymbol{\nu})\boldsymbol{\nu}
    -
    \mathbf{D}(\boldsymbol{\nu})\boldsymbol{\nu}
    -
    \mathbf{g}(\boldsymbol{\eta})
  \big)
  + \mathbf{w}_\nu,
  \label{eq:f_d_nu}\\
  \dot{\mathbf{b}}_\omega &= \mathbf{n}_{b\omega},
  \label{eq:f_d_bw}\\
  \dot{\mathbf{b}}_a &= \mathbf{n}_{ba},
  \label{eq:f_d_ba}\\
  \dot{\mathbf{g}}^G &= \mathbf{0}_{3\times 1},
  \label{eq:f_d_g}\\
  \dot{\mathbf{R}}^B_L &= \mathbf{0}, \qquad
  \dot{\mathbf{p}}^B_L = \mathbf{0}.
  \label{eq:f_d_extrinsics}
\end{align}

The discrete-time propagation (e.g. explicit Euler) over a time step $\Delta t$ can be written as
\begin{equation}
  \mathbf{x}_{k+1}
  =
  \mathbf{x}_k \boxplus
  \big(
    \Delta t\, \mathbf{f}_d(\mathbf{x}_k, \mathbf{u}_k, \mathbf{0})
  \big),
  \label{eq:process_model_disc}
\end{equation}
where $\boxplus$ denotes the manifold-plus operation (with exponential map on $SO(3)$).

For implementation, one may write explicitly
\begin{align}
  \mathbf{R}^G_{B,k+1} &= \mathbf{R}^G_{B,k}
  \exp\!\big([\boldsymbol{\omega}^B_k]_\times \Delta t\big),
  \label{eq:disc_R}\\
  \mathbf{p}^G_{B,k+1} &= \mathbf{p}^G_{B,k}
  + \mathbf{R}^G_{B,k} \mathbf{v}^B_k \Delta t,
  \label{eq:disc_p}\\
  \boldsymbol{\nu}_{k+1} &= \boldsymbol{\nu}_k + \dot{\boldsymbol{\nu}}_k \Delta t,
  \label{eq:disc_nu}
\end{align}
with $\dot{\boldsymbol{\nu}}_k$ given by \eqref{eq:nu_dot}, and the remaining components propagated according to
\eqref{eq:f_d_bw}--\eqref{eq:f_d_extrinsics}.

\subsection{IMU Measurement Model}

The gyroscope and accelerometer measurements in the body frame are modelled as
\begin{align}
  \boldsymbol{\omega}_m
  &= \boldsymbol{\omega}^B + \mathbf{b}_\omega + \mathbf{n}_\omega,
  \label{eq:imu_gyro_meas}\\
  \mathbf{a}_m
  &= \mathbf{f}^B + \mathbf{b}_a + \mathbf{n}_a,
  \label{eq:imu_acc_meas}
\end{align}
where $\mathbf{n}_\omega, \mathbf{n}_a$ are measurement noise and $\mathbf{f}^B$ is the specific force.

From the kinematics,
\begin{equation}
  \dot{\mathbf{p}}^G_B = \mathbf{R}^G_B \mathbf{v}^B
  \quad\Rightarrow\quad
  \ddot{\mathbf{p}}^G_B
  = \mathbf{R}^G_B
  \big(\dot{\mathbf{v}}^B + \boldsymbol{\omega}^B \times \mathbf{v}^B\big).
\end{equation}
Specific force in the body frame is
\begin{equation}
  \mathbf{f}^B
  = \mathbf{R}^B_G(\ddot{\mathbf{p}}^G_B - \mathbf{g}^G)
  = \dot{\mathbf{v}}^B + \boldsymbol{\omega}^B \times \mathbf{v}^B - \mathbf{R}^B_G \mathbf{g}^G.
\end{equation}
Therefore,
\begin{equation}
  \mathbf{a}_m
  =
  \dot{\mathbf{v}}^B
  + \boldsymbol{\omega}^B \times \mathbf{v}^B
  - \mathbf{R}^B_G \mathbf{g}^G
  + \mathbf{b}_a
  + \mathbf{n}_a.
  \label{eq:imu_acc_model_final}
\end{equation}

Collecting gyroscope and accelerometer into a single IMU measurement,
\begin{equation}
  \mathbf{z}_{\text{IMU}}
  =
  \begin{bmatrix}
    \boldsymbol{\omega}_m \\
    \mathbf{a}_m
  \end{bmatrix}
  =
  \mathbf{h}_{\text{IMU}}(\mathbf{x}, \mathbf{u})
  + \mathbf{n}_{\text{IMU}},
\end{equation}
with
\begin{equation}
  \mathbf{h}_{\text{IMU}}(\mathbf{x}, \mathbf{u})
  =
  \begin{bmatrix}
    \boldsymbol{\omega}^B + \mathbf{b}_\omega \\
    \dot{\mathbf{v}}^B(\mathbf{x}, \mathbf{u}) + \boldsymbol{\omega}^B \times \mathbf{v}^B
    - \mathbf{R}^B_G \mathbf{g}^G + \mathbf{b}_a
  \end{bmatrix}.
\end{equation}

\subsection{LiDAR Measurement Model}

Let $\mathbf{T}^G_B = (\mathbf{R}^G_B, \mathbf{p}^G_B)$ and
$\mathbf{T}^B_L = (\mathbf{R}^B_L, \mathbf{p}^B_L)$
denote the transformations from body to global and from LiDAR to body, respectively.

A LiDAR point $\mathbf{p}^L_j$ expressed in the LiDAR frame is transformed to the global frame as
\begin{equation}
  \mathbf{p}^G_j
  =
  \mathbf{T}^G_B \mathbf{T}^B_L \mathbf{p}^L_j
  =
  \mathbf{R}^G_B
  \big(
    \mathbf{R}^B_L \mathbf{p}^L_j + \mathbf{p}^B_L
  \big)
  + \mathbf{p}^G_B.
\end{equation}

Given a local plane with normal $\mathbf{u}_j$ and anchor point $\mathbf{q}_j$ in the global frame,
the point-to-plane residual for point $j$ is
\begin{equation}
  z_j
  =
  \mathbf{u}_j^\top
  \big(
    \mathbf{p}^G_j - \mathbf{q}_j
  \big),
\end{equation}
which defines the LiDAR measurement model used in the EKF update step.

\subsection{Underwater Dynamics Library, Simulation Code, and Thruster Inputs}

The Fossen-type dynamics in \eqref{eq:fossen_dyn} are implemented in an existing
C++ library, provided by the class \texttt{mvm::UnderwaterVehicleModel} defined in
\texttt{underwater\_vehicle\_model.hpp}. This class encapsulates the 6-DOF
underwater vehicle model by constructing the rigid-body and added-mass inertia
matrix, Coriolis/centripetal matrix, hydrodynamic damping, and restoring forces
from a configuration file, and by mapping individual thruster forces into the
generalized force vector.

Given body-frame velocity
\[
\boldsymbol{\nu}
=
\begin{bmatrix}
\mathbf{v}^B \\[2pt]
\boldsymbol{\omega}^B
\end{bmatrix}
=
[u, v, w, p, q, r]^\top
\]
and pose
\[
\boldsymbol{\eta}
=
[x, y, z, \phi, \theta, \psi]^\top,
\]
the method
\texttt{UnderwaterVehicleModel::UpdateModel(velocity, pose)} updates
\(\mathbf{M}(\boldsymbol{\nu})\), \(\mathbf{C}(\boldsymbol{\nu})\),
\(\mathbf{D}(\boldsymbol{\nu})\), and \(\mathbf{g}(\boldsymbol{\eta})\) internally.
The generalized thruster forces \(\boldsymbol{\tau}\) are obtained from a vector
of individual thruster inputs \(\boldsymbol{f}_{\text{thr}}\) via a precomputed
thruster-wrench matrix:
\begin{equation}
  \boldsymbol{\tau}
  =
  \mathbf{T}_{\mathrm{thr}} \, \boldsymbol{f}_{\text{thr}},
\end{equation}
where \(\mathbf{T}_{\mathrm{thr}} \in \mathbb{R}^{6 \times n_{\mathrm{thr}}}\)
is built from the thruster positions and orientations.

The corresponding body-frame acceleration is computed by
\texttt{UnderwaterVehicleModel::ComputeAcceleration(forces)}, which numerically
implements
\begin{equation}
  \dot{\boldsymbol{\nu}}
  =
  \mathbf{M}^{-1}(\boldsymbol{\nu})
  \big(
    \boldsymbol{\tau}
    -
    \mathbf{C}(\boldsymbol{\nu})\boldsymbol{\nu}
    -
    \mathbf{D}(\boldsymbol{\nu})\boldsymbol{\nu}
    -
    \mathbf{g}(\boldsymbol{\eta})
  \big),
\end{equation}
fully consistent with the Fossen dynamics in \eqref{eq:fossen_dyn}.

This library is already used inside the ROS~2 simulation code located at
\begin{center}
  \texttt{/home/attia/ros2\_ws/src/auv\_core\_ros2/sim/src/sim/simulator.cpp},
\end{center}
in the \texttt{Simulator} node. In particular,
\begin{itemize}
  \item the constructor loads the dynamic model configuration from
        \texttt{auv\_core\_helper/param/dynamic\_model/\textit{config\_name}.conf}
        and instantiates
        \texttt{dynamicsModel\_ = std::make\_unique<mvm::UnderwaterVehicleModel>(config, configNameParam);};
  \item at each simulation step \texttt{Simulator::Simulate()}, the code
        constructs the body-frame relative velocity \(\boldsymbol{\nu}\) and the
        world-frame pose \(\boldsymbol{\eta}\), then calls
        \begin{align}
          &\texttt{dynamicsModel\_->UpdateModel(velocityActualRel\_, poseActual\_);}\\
          &\texttt{accelerationActual\_ = dynamicsModel\_->ComputeAcceleration(forcesDesired\_);},
        \end{align}
        to obtain the body-frame acceleration \(\dot{\boldsymbol{\nu}}\);
  \item the resulting acceleration is then integrated to update the body-frame
        velocity, and the pose is integrated in the world frame using the
        current rotation matrix and Euler angle rates.
\end{itemize}

For the thruster inputs, including their timestamps, we can make use of the ROS~2
topic
\begin{center}
  \texttt{/auv/forces\_desired\_stamped},
\end{center}
whose type is
\begin{center}
  \texttt{auv\_core\_helper/msg/ThrusterForces}.
\end{center}
By subscribing to this topic, we obtain time-stamped thruster commands
\(\boldsymbol{f}_{\text{thr}}(t)\), which can be fed directly into the
\texttt{UnderwaterVehicleModel} via
\(\boldsymbol{\tau}(t) = \mathbf{T}_{\mathrm{thr}} \boldsymbol{f}_{\text{thr}}(t)\).
This is precisely the mechanism used in the existing simulation node and can be
reused in the proposed estimator: the same thruster command stream that drives
the simulator can be used as the input \(\mathbf{u}(t)\) of the Fossen-based
process model \(\mathbf{f}_d(\mathbf{x},\mathbf{u})\) within the FAST-LIO
framework.


\end{document}
