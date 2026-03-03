
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

## SIMULATION SET UO
dt = 0.01 ## time step = 100 Hz
T = 60 ## tiem for sim
N = int(T / dt) ## number of steps

t = np.linspace(0, T, N) ## generate da time vector

## sim tracks a circular trajectory with some vertical oscillation
R = 5 ## radius of the circle
omega = 0.2 ## angular velocity (rad/s)

# here is the grounf truth
## The drone flies in a circle of radius R, with a vertical oscillation of amplitude 0.5m and frequency 0.3 Hz
x_true = R * np.cos(omega * t)
y_true = R * np.sin(omega * t)
z_true = 1 + 0.5 * np.sin(0.3 * t)

## The true velocities and accelerations re derived
vx_true = -R * omega * np.sin(omega * t)
vy_true = R * omega * np.cos(omega * t)
vz_true = 0.5 * 0.3 * np.cos(0.3 * t)

ax_true = -R * omega**2 * np.cos(omega * t)
ay_true = -R * omega**2 * np.sin(omega * t)
az_true = -0.5 * (0.3**2) * np.sin(0.3 * t)

## set yaw and yaw rate to some simple functions of time
yaw_true = 0.2 * t
yaw_rate_true = 0.2 * np.ones(N)

# imulate noisy nmeasurments for sensors (IMU, barometer, optical flow, position sensor)
## IMU
ax_meas = ax_true + np.random.normal(0, 0.05, N)
ay_meas = ay_true + np.random.normal(0, 0.05, N)
az_meas = az_true + np.random.normal(0, 0.05, N)
yaw_rate_meas = yaw_rate_true + np.random.normal(0, 0.01, N)

## Barometer
z_meas = z_true + np.random.normal(0, 0.1, N)

## Optical flow (velocity)
vx_meas = vx_true + np.random.normal(0, 0.05, N)
vy_meas = vy_true + np.random.normal(0, 0.05, N)

## Position sensor (stereo/UWB)
x_meas = x_true + np.random.normal(0, 0.2, N)
y_meas = y_true + np.random.normal(0, 0.2, N)

# EKF IMPLEMENTATION
x_hat = np.zeros((7, 1)) ## state vector: [x, y, z, vx, vy, vz, yaw]
P = np.eye(7) * 0.5 ## covariance matrix: what is the uncertainty in ur state estimate?
Q = np.eye(7) * 0.01 ## noise covariance for the process model (how much we trust our motion model)
estimates = np.zeros((7, N)) ## estimates at each time step

# EKF prediction and update loop
for k in range(N):

    ## get imu measurements at time step k
    u = np.array([
        ax_meas[k],
        ay_meas[k],
        az_meas[k],
        yaw_rate_meas[k]
    ])

    ## prediction step
    x, y, z, vx, vy, vz, yaw = x_hat.flatten()
    ax, ay, az, omega_z = u

    ### apply kinematic equations to predict the next state
    x_pred = np.array([
        x + vx*dt + 0.5*ax*dt**2,
        y + vy*dt + 0.5*ay*dt**2,
        z + vz*dt + 0.5*az*dt**2,
        vx + ax*dt,
        vy + ay*dt,
        vz + az*dt,
        yaw + omega_z*dt
    ]).reshape(-1,1)

    ### jacobian F: linearizes motion model for covariance prediction
    F = np.eye(7)
    F[0,3] = dt
    F[1,4] = dt
    F[2,5] = dt

    ### Covariance prediction: ppropogate uncertainty and add noise Q
    P = F @ P @ F.T + Q

    x_hat = x_pred ### update state estimate with prediction

    ## position update
    z_pos = np.array([[x_meas[k]], [y_meas[k]]]) ### measurement vector
    
    ### map state to measurement (x and y only)
    H_pos = np.zeros((2,7))
    H_pos[0,0] = 1
    H_pos[1,1] = 1

    R_pos = np.eye(2) * 0.2**2 ### measurement noise covariance for position sensor

    y_res = z_pos - H_pos @ x_hat ### measurement - prediction residual = 'how off are we?'
    S = H_pos @ P @ H_pos.T + R_pos ### residual covariance = how much uncertainty do we have in this measurement?
    K = P @ H_pos.T @ np.linalg.inv(S) ### Kalman gain = how much should we trust this measurement vs our prediction?

    x_hat = x_hat + K @ y_res ### correct estimate to account for measurement
    P = (np.eye(7) - K @ H_pos) @ P ### update covariance with increased confisence

    ## barometer update (for altitude)
    z_bar = np.array([[z_meas[k]]])

    H_bar = np.zeros((1,7))
    H_bar[0,2] = 1

    R_bar = np.array([[0.1**2]])

    y_res = z_bar - H_bar @ x_hat
    S = H_bar @ P @ H_bar.T + R_bar
    K = P @ H_bar.T @ np.linalg.inv(S)

    x_hat = x_hat + K @ y_res
    P = (np.eye(7) - K @ H_bar) @ P

    # Opitacl flow update (for velocity)
    z_vel = np.array([[vx_meas[k]], [vy_meas[k]]])

    H_vel = np.zeros((2,7))
    H_vel[0,3] = 1
    H_vel[1,4] = 1

    R_vel = np.eye(2) * 0.05**2

    y_res = z_vel - H_vel @ x_hat
    S = H_vel @ P @ H_vel.T + R_vel
    K = P @ H_vel.T @ np.linalg.inv(S)

    x_hat = x_hat + K @ y_res
    P = (np.eye(7) - K @ H_vel) @ P

    estimates[:,k] = x_hat.flatten()

# PLOTTING RESULTS
## 2d position vs estimates
plt.figure()
plt.plot(x_true, y_true, label="True")
plt.plot(estimates[0,:], estimates[1,:], label="EKF")
plt.scatter(x_meas, y_meas, s=1, alpha=0.3, label="Noisy Position")
plt.legend()
plt.title("2D Position")
plt.show()

## altitude vs estimates
plt.figure()
plt.plot(t, z_true, label="True Z")
plt.plot(t, estimates[2,:], label="EKF Z")
plt.plot(t, z_meas, alpha=0.3, label="Barometer")
plt.legend()
plt.title("Altitude")
plt.show()

## 2D velocity vs estimates
plt.figure()
plt.plot(t, vx_true, label="True Vx")
plt.plot(t, estimates[3,:], label="EKF Vx")
plt.plot(t, vx_meas, alpha=0.3, label="Optical Flow")
plt.legend()
plt.title("2D Velocity")
plt.show()

## verical velocity vs estimates
plt.figure()
plt.plot(t, vz_true, label="True Vz")
plt.plot(t, estimates[5,:], label="EKF Vz")
plt.plot(t, az_meas, alpha=0.3, label="IMU Acceleration")
plt.legend()
plt.title("Vertical Velocity")
plt.show()

## yaw vs estimates
plt.figure()
plt.plot(t, yaw_true, label="True Yaw")
plt.plot(t, estimates[6,:], label="EKF Yaw")
plt.plot(t, yaw_rate_meas, alpha=0.3, label="IMU Yaw Rate")
plt.legend()
plt.title("Yaw")
plt.show()