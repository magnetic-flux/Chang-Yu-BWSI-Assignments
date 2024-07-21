import matplotlib.pyplot as plt

# --------------------------------------- Adjust these ---------------------------------------
# Physical constants
mass = 1 # kg
g = 9.81 # m/s²
max_speed = 5 # m/s, maximum vertical speed the drone can attain
acceleration = 2 # m/s, maximum acceleration magnitude the drone can attain

# Flight parameters
initial_altitude = 0 # m
initial_velocity = 0 # m/s, must be less than max_speed
target_altitude = 30 # m
target_velocity = 0 # m/s, must be less than max_speed
# --------------------------------------------------------------------------------------------

# The simulation assumes the drone accelerates and decelerates at its maximum acceleration, up to its maximum speed if necessary, to arrive at the target altitude with the target velocity

seconds_per_frame = 0.01

altitude, velocity, thrust, altitude_list, velocity_list, thrust_list, timestamps = initial_altitude, initial_velocity, 0, [initial_altitude], [initial_velocity], [0], [0]

# Advance one frame
def log_telemetry():
    global altitude, initial_altitude, target_altitude, velocity, initial_velocity, target_velocity, thrust, altitude_list, velocity_list, thrust_list, timestamps
    acceleration = (thrust / mass) - g
    velocity = velocity + (acceleration * seconds_per_frame)
    altitude = altitude + (velocity * seconds_per_frame)
    altitude_list.append(altitude)
    velocity_list.append(velocity)
    thrust_list.append(thrust)
    timestamps.append(timestamps[-1] + seconds_per_frame)

# Sets required thrust to accelerate for one frame in the specified direction (1 = up, -1 = down)
def accelerate(direction):
    global altitude, initial_altitude, target_altitude, velocity, initial_velocity, target_velocity, thrust, altitude_list, velocity_list, thrust_list, timestamps
    thrust = mass * (g + (direction * acceleration))

if target_altitude > altitude:
    while max_speed > velocity and altitude < target_altitude - ((velocity**2 - target_velocity**2) / (2 * acceleration)): # Initial acceleration up to the maximum velocity or the height at which it must start slowing down, whichever comes first
        accelerate(1)
        log_telemetry()
    while altitude < target_altitude - ((velocity**2 - target_velocity**2) / (2 * acceleration)): # Coasting at maximum velocity until the drone reaches the height at which it must slow down to arrive at the target altitude with the target velocity (skipped if the maximum velocity is not attained in the first place before needing to slow down)
        thrust = mass * g
        log_telemetry()
    while target_velocity < velocity: # Deceleration to arrive at the target altitude with the target velocity
        accelerate(-1)
        log_telemetry()
else: # Same thing but with signs adjusted for descent
    while max_speed < velocity and altitude > target_altitude + ((velocity**2 - target_velocity**2) / (2 * acceleration)):
        accelerate(-1)
        log_telemetry()
    while altitude > target_altitude - ((velocity**2 - target_velocity**2) / (2 * acceleration)):
        thrust = mass * g
        log_telemetry()
    while target_velocity > velocity:
        accelerate(1)
        log_telemetry()

fig, axs = plt.subplots(3)
fig.suptitle("Telemetry for drone with mass " + str(mass) + " kg, maximum speed " + str(max_speed) + " m/s, and acceleration " + str(acceleration) + " m/s² flying to height " + str(target_altitude) + " m from height " + str(initial_altitude) + " m with initial velocity " + str(initial_velocity) + " m/s, final velocity " + str(target_velocity) + " m/s, and g = " + str(g) + " m/s²:")
axs[0].plot(timestamps, altitude_list)
axs[1].plot(timestamps, velocity_list)
axs[2].plot(timestamps, thrust_list)
axs[0].set(xlabel='Time (s)', ylabel='Altitude (m)')
axs[1].set(xlabel='Time (s)', ylabel='Velocity (m/s)')
axs[2].set(xlabel='Time (s)', ylabel='Thrust (N)')
plt.show()