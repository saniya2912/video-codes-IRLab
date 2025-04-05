import numpy as np

from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu
import time
#######################################################
"""This can control and query the LEAP Hand

I recommend you only query when necessary and below 90 samples a second.  Each of position, velociy and current costs one sample, so you can sample all three at 30 hz or one at 90hz.

#Allegro hand conventions:
#0.0 is the all the way out beginning pose, and it goes positive as the fingers close more and more
#http://wiki.wonikrobotics.com/AllegroHandWiki/index.php/Joint_Zeros_and_Directions_Setup_Guide I belive the black and white figure (not blue motors) is the zero position, and the + is the correct way around.  LEAP Hand in my videos start at zero position and that looks like that figure.

#LEAP hand conventions:
#180 is flat out for the index, middle, ring, fingers, and positive is closing more and more.

"""
########################################################
class LeapNode_Poscontrol:
    def __init__(self):
        ####Some parameters
        # self.ema_amount = float(rospy.get_param('/leaphand_node/ema', '1.0')) #take only current
        self.kP = 250
        self.kI = 0
        self.kD = 25
        self.kP_slow = 300
        self.kI = 0
        self.kD = 200
        self.curr_lim = 350
        self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(16))
           
        #You can put the correct port here or have the node auto-search for a hand at the first 3 ports.
        self.motors = motors = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        try:
            self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB0', 4000000)
            self.dxl_client.connect()
        except Exception as e:
            print("[DEBUG]", e)
            try:
                self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB1', 4000000)
                self.dxl_client.connect()
            except Exception:
                self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB2', 4000000)
                self.dxl_client.connect()

        self.dxl_client.set_torque_enabled(self.motors, False)
        ADDR_SET_MODE = 11
        LEN_SET_MODE = 1
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * 3, ADDR_SET_MODE, LEN_SET_MODE)
        self.dxl_client.set_torque_enabled(self.motors, True)

        #Enables position-current control mode and the default parameters, it commands a position and then caps the current so the motors don't overload
        self.dxl_client.sync_write(motors, np.ones(len(motors))*5, 11, 1)
        self.dxl_client.set_torque_enabled(motors, True)
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kP, 84, 2) # Pgain stiffness     
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kP * 0.75), 84, 2) # Pgain stiffness for side to side should be a bit less
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kI, 82, 2) # Igain
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kD, 80, 2) # Dgain damping
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kD * 0.75), 80, 2) # Dgain damping for side to side should be a bit less
        #Max at current (in unit 1ma) so don't overheat and grip too hard #500 normal or #350 for lite
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.curr_lim, 102, 2)
        #self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    #Receive LEAP pose and directly control the robot
    def set_leap(self, pose):
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #allegro compatibility
    def set_allegro(self, pose):
        pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #Sim compatibility, first read the sim value in range [-1,1] and then convert to leap
    def set_ones(self, pose):
        pose = lhu.sim_ones_to_LEAPhand(np.array(pose))
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #read position
    def read_pos(self):
        return self.dxl_client.read_pos()
    
    def read_pos_leap(self):
        pos=self.dxl_client.read_pos()-(np.ones(16)*3.14)
        return pos
    #read velocity
    def read_vel(self):
        return self.dxl_client.read_vel()
    #read current
    def read_cur(self):
        return self.dxl_client.read_cur()
    
    def cubic_trajectory(self,q0, v0, q1, v1, t0, t1, current_time):
        # Ensure the current time is within the valid time range
        current_time = np.clip(current_time, t0, t1)

        # Define the matrix M for scalar time t0 and t1 (applies to all elements)
        M = np.array([
            [1, t0, t0**2, t0**3],
            [0, 1, 2*t0, 3*t0**2],
            [1, t1, t1**2, t1**3],
            [0, 1, 2*t1, 3*t1**2]
        ])

        # Stack the q0, v0, q1, v1 values into a matrix (each as a 16-element array)
        b = np.vstack([q0, v0, q1, v1])

        # Solve for the coefficients a for each set of q0, v0, q1, v1
        a = np.linalg.inv(M).dot(b)

        # Compute position (qd), velocity (vd), and acceleration (ad) for each element
        qd = a[0] + a[1]*current_time + a[2]*current_time**2 + a[3]*current_time**3
        vd = a[1] + 2*a[2]*current_time + 3*a[3]*current_time**2
        ad = 2*a[2] + 6*a[3]*current_time

        return qd
    # def cubic_trajectory(self, q0, v0, q1, v1, t0, t1, current_time):
    # # Ensure the current time is within [t0, t1]
    #     current_time = np.clip(current_time, t0, t1)

    #     # Precompute time powers
    #     t = current_time
    #     dt = t - t0

    #     # Coefficient matrix M (same for all joints)
    #     M = np.array([
    #         [1, t0, t0**2, t0**3],
    #         [0, 1, 2*t0, 3*t0**2],
    #         [1, t1, t1**2, t1**3],
    #         [0, 1, 2*t1, 3*t1**2]
    #     ])

    #     # Initialize output
    #     qd = np.zeros_like(q0)

    #     # Solve per joint
    #     for i in range(len(q0)):
    #         b = np.array([q0[i], v0[i], q1[i], v1[i]])  # 4x1
    #         a = np.linalg.solve(M, b)  # 4 coefficients for joint i
    #         qd[i] = a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3  # position at current time

    #     return qd

    

class LeapNode_Taucontrol():
    def __init__(self):
    # List of motor IDs
        self.motors = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        
        try:
            # Try connecting to /dev/ttyUSB0
            self.dxl_client = DynamixelClient(self.motors, '/dev/ttyUSB0', 4000000)
            self.dxl_client.connect()
        except Exception as e:
            print("[DEBUG]", e)
            # Try connecting to /dev/ttyUSB1 if /dev/ttyUSB0 fails
            try:
                self.dxl_client = DynamixelClient(self.motors, '/dev/ttyUSB1', 4000000)
                self.dxl_client.connect()
            except Exception:
                # Try connecting to /dev/ttyUSB2 if /dev/ttyUSB1 fails
                self.dxl_client = DynamixelClient(self.motors, '/dev/ttyUSB2', 4000000)
                self.dxl_client.connect()

        self.dxl_client.set_torque_enabled(self.motors, False)
        # Set the control mode to Torque Control Mode
        # Address 11 typically corresponds to setting the operating mode, and 0 is Torque Control Mode
        ADDR_SET_MODE = 11
        LEN_SET_MODE = 1
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors))* 0, ADDR_SET_MODE, LEN_SET_MODE)
        self.dxl_client.set_torque_enabled(self.motors, True)

        # Set the current limit for Torque Control (Goal Current)
        # Address 102 might correspond to Goal Current, and 2 bytes is the length
        ADDR_GOAL_CURRENT = 102
        LEN_GOAL_CURRENT = 2
        self.curr_lim = 350  # Adjust the current limit as needed (this is just an example)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.curr_lim, ADDR_GOAL_CURRENT, LEN_GOAL_CURRENT)

    def set_desired_torque(self, desired_torque):
    # Convert desired torque to the corresponding current (depends on the motor's torque constant)
    # For example, assume a torque constant where 1 unit of torque corresponds to 1 unit of current
        

        # Ensure all elements in the desired_torque are scalars
        # For instance, you can flatten the list if necessary
        desired_torque_flat = [float(torque) for torque in desired_torque]  # Convert all to floats

        # Convert to NumPy array and calculate the current
        desired_current = np.array([torque / 0.51 for torque in desired_torque_flat])
        # Adjust this based on your motor's torque constant

        # Address for the Goal Current (or Torque) register
        ADDR_GOAL_CURRENT = 102
        LEN_GOAL_CURRENT = 2  # Length is usually 2 bytes

        # Write the desired current (which corresponds to the desired torque) to all motors
        self.dxl_client.sync_write(self.motors, desired_current*1000, ADDR_GOAL_CURRENT, LEN_GOAL_CURRENT)

    def read_pos(self):
        return self.dxl_client.read_pos()
    
    def read_pos_leap(self):
        pos=self.dxl_client.read_pos()-(np.ones(16)*3.14)
        return pos
    #read velocity
    def read_vel(self):
        return self.dxl_client.read_vel()
    #read current
    def read_cur(self):
        return self.dxl_client.read_cur()

        

