\documentclass[11pt,a4paper]{article}

\usepackage[margin=2.5cm]{geometry}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{bm}
\usepackage{hyperref}

\title{Dynamic vs.\ Kinematic Model in FAST-LIO:\\
Why Dynamics Performed Worse and How to Fix It}
\author{}
\date{}

\begin{document}
\maketitle

This document explains why, in the current implementation, the \emph{kinematic}
FAST-LIO variant can outperform the \emph{dynamic (Fossen-based)} variant, even
though the simulator and the estimator nominally use the same underwater vehicle
model. It also describes the required mathematical and implementation fixes.

Key points:
\begin{enumerate}
  \item Water current is set to zero in the simulation, so relative vs.\ absolute
        velocity is \emph{not} the culprit in the current experiments.
  \item The gravity state is mathematically inconsistent (unit vector vs.\ full vector).
  \item The \emph{dynamics trust scalar} must be applied via covariances, not
        by blending dynamics prediction and raw IMU measurements.
\end{enumerate}

\section{Gravity State Consistency}

\subsection{Current Situation}

In the current formulation, the state manifold includes a gravity term defined
on the unit sphere,
\begin{equation}
  \bm{g}^G \in S^2,
\end{equation}
i.e., gravity is represented as a pure \emph{direction} of unit length.

However, in the IMU accelerometer model, it is used as if it were the full
gravity vector in $\mathbb{R}^3$, with physical units:
\begin{equation}
  \bm{a}_m
  =
  \dot{\bm{v}}^B
  +
  \bm{\omega}^B \times \bm{v}^B
  -
  \bm{R}^B_G \bm{g}^G
  +
  \bm{b}_a
  +
  \bm{n}_a.
  \label{eq:acc_model_current}
\end{equation}
If $\bm{g}^G \in S^2$, it is dimensionless and of unit norm, while in
\eqref{eq:acc_model_current} it is implicitly treated as having units
$\mathrm{m/s^2}$. This is mathematically inconsistent and can introduce scale
or bias issues in the dynamic variant, especially if gravity is being estimated
or refined.

\subsection{Recommended Fix (Simple and Consistent)}

A straightforward fix is to make gravity a full vector again, as in the original
FAST-LIO2 formulation. That is,
\begin{equation}
  \bm{g}^G \in \mathbb{R}^3,
\end{equation}
and the IMU accelerometer model is
\begin{equation}
  \bm{a}_m
  =
  \dot{\bm{v}}^B
  +
  \bm{\omega}^B \times \bm{v}^B
  -
  \bm{R}^B_G \bm{g}^G
  +
  \bm{b}_a
  +
  \bm{n}_a.
  \label{eq:acc_model_vecg}
\end{equation}
This keeps the math internally consistent and matches the original inertial
formulation used by FAST-LIO.

\paragraph{Alternative (\texorpdfstring{$S^2$}{S2} + scalar $g$).}
If one insists on storing only the gravity direction, one could keep
$\hat{\bm{g}}^G \in S^2$ and introduce a scalar $g>0$ (e.g.\ $9.81\ \mathrm{m/s^2}$).
Then the full gravity vector is $g\,\hat{\bm{g}}^G$ and the IMU model becomes
\begin{equation}
  \bm{a}_m
  =
  \dot{\bm{v}}^B
  +
  \bm{\omega}^B \times \bm{v}^B
  -
  \bm{R}^B_G \big( g\,\hat{\bm{g}}^G \big)
  +
  \bm{b}_a
  +
  \bm{n}_a.
\end{equation}
However, the simplest and most compatible option is to take
$\bm{g}^G \in \mathbb{R}^3$ directly as in \eqref{eq:acc_model_vecg}.

\section{IMU Measurement Model and the Dynamics Trust Scalar}

\subsection{Correct Accelerometer Model with Fossen Dynamics}

From the pose kinematics,
\begin{equation}
  \dot{\bm{p}}^G_B = \bm{R}^G_B \bm{v}^B,
\end{equation}
we obtain
\begin{equation}
  \ddot{\bm{p}}^G_B
  =
  \bm{R}^G_B
  \big(
    \dot{\bm{v}}^B + \bm{\omega}^B \times \bm{v}^B
  \big),
\end{equation}
where
\begin{itemize}
  \item $\bm{p}^G_B$ is the position of the body in the global frame,
  \item $\bm{R}^G_B \in SO(3)$ is the rotation from body frame to global frame,
  \item $\bm{v}^B$ and $\bm{\omega}^B$ are linear and angular velocities in the
        body frame.
\end{itemize}

The specific force in the body frame (what the accelerometer measures) is
\begin{equation}
  \bm{f}^B
  =
  \bm{R}^B_G(\ddot{\bm{p}}^G_B - \bm{g}^G)
  =
  \dot{\bm{v}}^B
  +
  \bm{\omega}^B \times \bm{v}^B
  -
  \bm{R}^B_G \bm{g}^G.
\end{equation}
Therefore, the accelerometer measurement is
\begin{equation}
  \bm{a}_m
  =
  \bm{f}^B + \bm{b}_a + \bm{n}_a
  =
  \dot{\bm{v}}^B
  +
  \bm{\omega}^B \times \bm{v}^B
  -
  \bm{R}^B_G \bm{g}^G
  +
  \bm{b}_a
  +
  \bm{n}_a.
  \label{eq:acc_model_clean}
\end{equation}

On the dynamics side, with water current set to zero in these experiments, the
Fossen model reads
\begin{equation}
  \bm{M}(\bm{\nu})\,\dot{\bm{\nu}}
  +
  \bm{C}(\bm{\nu})\,\bm{\nu}
  +
  \bm{D}(\bm{\nu})\,\bm{\nu}
  +
  \bm{g}(\bm{\eta})
  =
  \bm{\tau},
\end{equation}
where
\begin{align}
  \bm{\nu} &= \begin{bmatrix} \bm{v}^B \\ \bm{\omega}^B \end{bmatrix}, \\
  \bm{\eta} &= [x,y,z,\phi,\theta,\psi]^\top,
\end{align}
and $\bm{\tau}$ denotes the generalized thruster wrench.

Solving for accelerations, we obtain
\begin{equation}
  \begin{bmatrix}
    \dot{\bm{v}}^B \\
    \dot{\bm{\omega}}^B
  \end{bmatrix}
  =
  \bm{M}^{-1}(\bm{\nu})
  \big(
    \bm{\tau}
    -
    \bm{C}(\bm{\nu})\,\bm{\nu}
    -
    \bm{D}(\bm{\nu})\,\bm{\nu}
    -
    \bm{g}(\bm{\eta})
  \big).
\end{equation}
Thus, given the state $\bm{x}$ and input $\bm{u}$ (containing, e.g., thruster
forces), the \emph{predicted accelerometer measurement} is
\begin{equation}
  \bm{h}_a(\bm{x},\bm{u})
  =
  \dot{\bm{v}}^B(\bm{x},\bm{u})
  +
  \bm{\omega}^B \times \bm{v}^B
  -
  \bm{R}^B_G \bm{g}^G
  +
  \bm{b}_a.
\end{equation}
The residual is then
\begin{equation}
  \bm{r}_a
  =
  \bm{a}_m - \bm{h}_a(\bm{x},\bm{u}).
\end{equation}
Critically, $\bm{h}_a$ depends only on the \emph{state} and \emph{inputs}. It
must not depend on $\bm{a}_m$ itself.

For the gyroscope, we have
\begin{equation}
  \bm{\omega}_m
  =
  \bm{\omega}^B + \bm{b}_\omega + \bm{n}_\omega,
\end{equation}
so the predicted gyro measurement and residual are
\begin{equation}
  \bm{h}_\omega(\bm{x})
  =
  \bm{\omega}^B + \bm{b}_\omega,
  \qquad
  \bm{r}_\omega
  =
  \bm{\omega}_m - \bm{h}_\omega(\bm{x}).
\end{equation}

Combining both, the IMU measurement model is
\begin{equation}
  \bm{z}_{\mathrm{IMU}}
  =
  \begin{bmatrix}
    \bm{\omega}_m \\
    \bm{a}_m
  \end{bmatrix}
  =
  \bm{h}_{\mathrm{IMU}}(\bm{x},\bm{u})
  +
  \bm{n}_{\mathrm{IMU}},
\end{equation}
with
\begin{equation}
  \bm{h}_{\mathrm{IMU}}(\bm{x},\bm{u})
  =
  \begin{bmatrix}
    \bm{\omega}^B + \bm{b}_\omega \\
    \dot{\bm{v}}^B(\bm{x},\bm{u})
    +
    \bm{\omega}^B \times \bm{v}^B
    -
    \bm{R}^B_G \bm{g}^G
    +
    \bm{b}_a
  \end{bmatrix}.
\end{equation}

\subsection{What the Dynamics Trust Scalar Should \texorpdfstring{Not}{Not} Do}

In the existing description, it is stated that the accelerometer residual
``uses the dynamics-predicted specific force and blends it with the raw IMU
measurement via a tunable trust scalar.'' Any blending of the form
\begin{equation}
  \tilde{\bm{h}}_a
  =
  \alpha\,\bm{h}_a(\bm{x},\bm{u})
  +
  (1-\alpha)\,\bm{a}_m
\end{equation}
for some $\alpha$ (``trust parameter''), and then defining the residual
\begin{equation}
  \bm{r}_a
  =
  \bm{a}_m - \tilde{\bm{h}}_a,
\end{equation}
is mathematically incorrect in the standard EKF framework, because the
\emph{predicted measurement} $\tilde{\bm{h}}_a$ now depends on the measurement
$\bm{a}_m$ itself. This:
\begin{itemize}
  \item hides part of the residual,
  \item leads to non-standard and hard-to-interpret filter behavior,
  \item tends to over-trust the dynamics model even when it is slightly mismatched.
\end{itemize}

\subsection{Where the Dynamics Trust Scalar Should Live}

The relative trust between dynamics, IMU, and LiDAR should be expressed through
the \emph{covariance matrices}, not by modifying the measurement function.

Two standard, clean approaches are:

\paragraph{(A) Scale the Process Noise on Velocities.}
Let $Q_\nu$ be the process noise covariance for $(\bm{v}^B,\bm{\omega}^B)$.
Introduce a scalar $\lambda_{\mathrm{dyn}} > 0$ and define
\begin{equation}
  Q_\nu(\lambda_{\mathrm{dyn}})
  =
  \lambda_{\mathrm{dyn}}\,Q_{\nu,0},
\end{equation}
where $Q_{\nu,0}$ is a nominal covariance. Then:
\begin{itemize}
  \item small $\lambda_{\mathrm{dyn}}$ $\Rightarrow$ strong trust in the Fossen
        dynamics (less process noise);
  \item large $\lambda_{\mathrm{dyn}}$ $\Rightarrow$ weaker trust in dynamics
        (more process noise), allowing LiDAR and IMU to correct more strongly.
\end{itemize}

\paragraph{(B) Scale the Accelerometer Measurement Covariance.}
Let $R_a$ be the accelerometer measurement covariance. Introduce
\begin{equation}
  R_a(\lambda_{\mathrm{dyn}})
  =
  \lambda_{\mathrm{dyn}}\,R_{a,0},
\end{equation}
with $R_{a,0}$ a baseline IMU noise covariance. Then:
\begin{itemize}
  \item large $\lambda_{\mathrm{dyn}}$ $\Rightarrow$ downweight accelerometer
        residuals (trust dynamics more);
  \item small $\lambda_{\mathrm{dyn}}$ $\Rightarrow$ upweight accelerometer
        residuals (trust IMU more).
\end{itemize}

In both cases, the measurement function remains
\begin{equation}
  \bm{z}_{\mathrm{IMU}}
  =
  \bm{h}_{\mathrm{IMU}}(\bm{x},\bm{u})
  +
  \bm{n}_{\mathrm{IMU}},
\end{equation}
with $\bm{h}_{\mathrm{IMU}}(\bm{x},\bm{u})$ defined as above. There is no
blending of $\bm{a}_m$ into $\bm{h}_{\mathrm{IMU}}$.

\subsection{Suggested Text Change in the Report}

Any phrasing such as
\begin{quote}
  ``The accelerometer residual uses the dynamics-predicted specific force and
   blends it with the raw IMU measurement via a tunable trust scalar.''
\end{quote}
should be replaced by something along the lines of
\begin{quote}
  ``The accelerometer residual compares the raw IMU measurement directly to
   the dynamics-predicted specific force:
   \[
     \bm{r}_a
     =
     \bm{a}_m
     -
     \big(
       \dot{\bm{v}}^B(\bm{x},\bm{u})
       +
       \bm{\omega}^B \times \bm{v}^B
       -
       \bm{R}^B_G \bm{g}^G
       +
       \bm{b}_a
     \big).
   \]
   A dynamics trust scalar $\lambda_{\mathrm{dyn}}$ is used only to scale the
   covariance of this residual (or the process noise on $\bm{v}^B$), thereby
   tuning how strongly the filter trusts the Fossen-based prediction relative
   to LiDAR and IMU data, without feeding the measurement back into the
   prediction.''
\end{quote}

\section{Water Current Note (for Completeness)}

The Fossen equations with current typically distinguish between the body
velocity $\bm{\nu}$ and the water-relative velocity
$\bm{\nu}_r = \bm{\nu} - \bm{\nu}_c$, where $\bm{\nu}_c$ is the current in the
body frame.

In the present experiments, however, the simulation parameters are set such that
the current is zero, e.g.,
\begin{equation}
  \mathrm{current\_y\_velocity} = 0.0,
  \qquad
  \mathrm{current\_z\_velocity} = 0.0,
\end{equation}
so $\bm{\nu}_c = \bm{0}$ and therefore $\bm{\nu}_r = \bm{\nu}$. This implies:
\begin{itemize}
  \item there is no relative-vs-absolute velocity mismatch between simulator and
        estimator in these runs,
  \item the observed performance gap between dynamic and kinematic models must
        be due to other modeling or tuning issues (e.g.\ gravity representation,
        IMU/dynamics fusion, parameter mismatch, etc.).
\end{itemize}

It can be useful to state this explicitly in the report, e.g.:
\begin{quote}
  ``In all experiments reported here, the water current is set to zero in the
  simulator (i.e.\ $\bm{\nu}_c = \bm{0}$). Therefore, the hydrodynamic model
  effectively uses $\bm{\nu}_r = \bm{\nu}$, and dynamic-vs-kinematic performance
  differences cannot be attributed to a relative-vs-absolute velocity mismatch.''
\end{quote}

\section{Practical Tuning Recipe After Fixes}

Once the gravity state is consistent and the IMU model is clean (no blending,
trust scalar only via covariances), the following tuning strategy is
recommended:

\begin{enumerate}
  \item \textbf{Start conservative (do not over-trust dynamics).}
        Use relatively large process noise on $(\bm{v}^B,\bm{\omega}^B)$ (or a
        large $\lambda_{\mathrm{dyn}}$ if it scales $Q_\nu$). Use IMU
        measurement covariances similar to the original kinematic FAST-LIO2
        configuration.
  \item \textbf{Compare dynamic vs.\ kinematic odometry.}
        Run both variants against the simulator ground truth. If the dynamic
        variant is similar or slightly worse, gradually reduce $Q_\nu$ (or
        reduce $\lambda_{\mathrm{dyn}}$ if it multiplies $Q_\nu$) to trust the
        dynamics more.
  \item \textbf{Monitor behavior.}
        If trusting dynamics more improves performance, continue tuning. If it
        makes performance worse (more drift or bias), this indicates remaining
        mismatches in the dynamic model or inputs (thruster timing, drag
        parameters, etc.), not in the EKF math. In that case, keep process
        noise relatively high so LiDAR can correct those mismatches.
\end{enumerate}

\section{Summary Checklist}

To make the dynamic model mathematically sound and give it a fair comparison
against the kinematic FAST-LIO, ensure the following:

\begin{itemize}
  \item[\(\square\)] \textbf{Gravity state:} use $\bm{g}^G \in \mathbb{R}^3$
                     (or explicitly use $g\,\hat{\bm{g}}^G$ if staying on $S^2$).
  \item[\(\square\)] \textbf{IMU measurement function:} ensure
                     $\bm{h}_{\mathrm{IMU}}(\bm{x},\bm{u})$ depends only on state
                     and inputs, never on the IMU measurements themselves.
  \item[\(\square\)] \textbf{Dynamics trust scalar:} use it only to scale
                     process noise $Q$ and/or measurement noise $R$, not to
                     blend model prediction and raw IMU inside
                     $\bm{h}_{\mathrm{IMU}}$.
  \item[\(\square\)] \textbf{Water current:} with current set to zero, verify
                     that the dynamic equations and parameters in the estimator
                     match those in the simulator.
  \item[\(\square\)] \textbf{Retune covariances:} tune $Q_\nu$, $R_a$, and the
                     trust scalar $\lambda_{\mathrm{dyn}}$ so as not to
                     over-trust an imperfect dynamic model.
\end{itemize}

With these changes, the Fossen-based dynamic model and the associated IMU fusion
become mathematically consistent, and it becomes plausible for the dynamic
variant to match or outperform the kinematic FAST-LIO in underwater SLAM
experiments.

\end{document}
