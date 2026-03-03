# EKF State Estimation Proof of Concept for a Circling Blimp
## Overview:
This project aims to simulate a multi-sensor Extended Kalman Filter (EKF) for 3D state estimation of a circling blimp. The goal is to validate the proposed sensor stack prior to hardware integration by:
1. simulating noisy sensor measurements
2. performing sensor fusion
3. demonstrating and visualizing improvments in estimation

## Extended Kalman Filters
### Bayesian Estimation
Given noisy sensor estimates, how can we ascertain the current state of the blimp? 
$p(current state at time |sensor measurements up to time k)$
We use Bayes to obtain the probability distribution of states at time k
$$p(state|sensors) = \frac{p(sensors|state) \cdot p(state)}{[p(sensors)}$$
Essentially, we use the **likelihood** (probability of obtaining sensor measuremnets given current state), the **prior** (initial beliefs from the motion model) to get the **posterior** (updated beliefs). EKF's abstracts this by assuming all distributions are gaussian, allowing us to only track the state estimate (mean) and the uncertainty (covariance)

### Covariance
The covariance matrix encodes the variance of each state and the correlation between states. It represents the uncertainty in the estimated state vector. Diagonal elements contain the variance of individual state variables, while off-diagonal elements represent the covariance between state variables (with a value of non-zero indicating that one variable changes with another). In an EKF implementation, the covatriance is updated at each time step to reflect new information from measurements and motion. The more the robot moves = the more information we have = the less uncertainty there is about our estimates.

### Prediction and Update
We use the motion model to predict the mean and covariance. Then, we use the measurement model to compute the **innovation**, or the difference between the actual and predicted measurement. We quantify the uncertainty in the innovation using the innovation covariance matrix, then use this to obtain the **Kalman Gain**. This is the weighting factor that determines how much the filter can trust the new measurement versus the prior state estimate.  Finally, we update the mean and covariance.

## State definition
The filter estimates the following state vector
$$
state = [p_x, p_y, p_z, v_x, v_y, v_z, yaw] 
$$

## Models 
The motion model uses IMU acceleration and yaw rate with standard kinematics. Process noise is modelled using Gaussian distribution

## Simulated Sensors
The following sensors are simulates with Gaussian noise:
* IMU (ax, ay, az, yaw rate)
* Barometer (z)
* Optical flow (vx, vy)
* Stereo/UWB position sensor (x, y)
Each sensor update is incorporated sequentially in the EKF.

## EKF pipeline
At each time step, we:
1. prediction step using IMU
2. position updtate
3. barometer update, optical flow update
4. propagate and update covariance

## Results
We see that incorporation of the EKF successfully reduces sensor noise and produces smoother estimates, demonstrating improved state estimation through fusion

## How to run
install dependencies:
'''pip install numpy matplotlib'''
run:
'''python ekf_sim.py'''

## What's Next?!
* imu biases
* asyncrounous multi-rate updates
* quaternion orientation representation
* ROS2 node implementation -> robot_localization
* hardware integration



