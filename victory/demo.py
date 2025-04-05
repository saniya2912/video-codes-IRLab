import numpy as np
import time
from main_6sept import LeapNode_Poscontrol
import numpy as np
from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu
import mujoco
# Assuming LeapNode_Poscontrol is defined as above

# Create an instance of the LeapNode_Poscontrol class
# leap_hand = LeapNode_Poscontrol()
# pos=np.array([-0.03982486,  1.00021397,  0.36514603,  0.28691302,  0.02306829,  1.47881638,
# -0.46320356,  0.43264137, -0.07050456,  1.49722372, -0.28679575,  0.04607807,
# 1.39751516, 0.02767025,  0.6243888,  -0.69023265])

# motors = motors = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# try:
#     dxl_client = DynamixelClient(motors, '/dev/ttyUSB0', 4000000)
#     dxl_client.connect()
# except Exception as e:
#     print("[DEBUG]", e)
#     try:
#         dxl_client = DynamixelClient(motors, '/dev/ttyUSB1', 4000000)
#         dxl_client.connect()
#     except Exception:
#         dxl_client = DynamixelClient(motors, '/dev/ttyUSB2', 4000000)
#         dxl_client.connect()

# try:
#     while True:
#         # Read the current motor positions
#         current_positions = dxl_client.read_pos()-(np.ones(16)*3.14)

#         # Print the motor positions
#         print("Current Motor Positions:", current_positions)

#         # Add a delay between readings (e.g., 100ms)
#         time.sleep(0.1)

# except KeyboardInterrupt:
#     print("Position reading stopped.")
import numpy as np
from main_6sept import LeapNode_Poscontrol, LeapNode_Taucontrol
import time
import mujoco
from leap_hand_utils.dynamixel_client import *
import numpy as np
import pandas as pd

# Initialize dictionaries to store data
df = pd.DataFrame(columns=['Time','start_time', 'Torque', 'Position','Velocity','Control_mode','subtask'])


#single object
leap_hand = LeapNode_Poscontrol()
pos=leap_hand.read_pos_leap()
print(pos)
# pos=np.array([-0.03982486,  1.00021397,  0.36514603,  0.28691302,  0,0,0,0,0,0,0,0,
# 1.39751516, 0.02767025,  0.6243888,  -0.69023265])

#object in palm
leap_hand = LeapNode_Poscontrol()
pos2=leap_hand.read_pos_leap()
print(pos2)


pos2=np.array([-0.02295102,  1.50796162,  0.33753453, -0.24384417,  0.00926267,  1.49262177,
  0.78085481 , 0.6351267,  -0.09504808,  1.5616511,   0.50013648,  0.79312669,
  1.42972885, -0.09504808,  0.05988394, -0.00914515])


#both object
leap_hand = LeapNode_Poscontrol()
pos3=leap_hand.read_pos_leap()
print(pos3)

pos3=np.array([-0.05823268,  1.49568974,  0.24856363, -0.06436862,  0.02613626,  1.33001982,
  1.45273863,  0.24242769, -0.00761117,  1.24565111,  1.46501051,  0.37895189,
  1.42052494,  0.07522379,  0.11510716, -0.15947522])

def J(model,data,site_name):
        # model=mujoco.MjModel.from_xml_path(xml_path)
        # data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        jacp = np.zeros((3, model.nv))  # translation jacobian
        jacr = np.zeros((3, model.nv)) 

        site_id=model.site(site_name).id
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

        return np.vstack((jacp, jacr))
pos_index= pos[0:4].reshape(4)
pos_index_mujoco=[pos_index[1], pos_index[0], pos_index[2], pos_index[3]]

pos_thumb=pos_thumb_mujoco= pos[12:16].reshape(4)


index_model_path = '/home/saniya/LEAP/leap_hand_mujoco/model/leap hand/index_finger.xml'

index_m = mujoco.MjModel.from_xml_path(index_model_path)
index_d = mujoco.MjData(index_m)


index_d.qpos= pos_index_mujoco
mujoco.mj_forward(index_m, index_d)
index_J=J(index_m,index_d,'contact_index')


thumb_model_path = '/home/saniya/LEAP/leap_hand_mujoco/model/leap hand/thumb.xml'

thumb_m = mujoco.MjModel.from_xml_path(thumb_model_path)
thumb_d = mujoco.MjData(thumb_m)

thumb_d.qpos=pos_thumb_mujoco
mujoco.mj_forward(thumb_m, thumb_d)
thumb_J=J(thumb_m,thumb_d,'contact_thumb')



pos_middle= pos2[4:8].reshape(4)
pos_middle_mujoco=[pos_middle[1], pos_middle[0], pos_middle[2], pos_middle[3]]

middle_model_path = '/home/saniya/LEAP/leap_hand_mujoco/model/leap hand/middle.xml'

middle_m = mujoco.MjModel.from_xml_path(middle_model_path)
middle_d = mujoco.MjData(middle_m)


middle_d.qpos= pos_middle_mujoco
mujoco.mj_forward(middle_m, middle_d)
middle_J=J(middle_m,middle_d,'contact_middle')



pos_tertiary= pos[8:12].reshape(4)
pos_tertiary_mujoco=[pos_tertiary[1], pos_tertiary[0], pos_tertiary[2], pos_tertiary[3]]

tertiary_model_path = '/home/saniya/LEAP/leap_hand_mujoco/model/leap hand/tertiary.xml'

tertiary_m = mujoco.MjModel.from_xml_path(tertiary_model_path)
tertiary_d = mujoco.MjData(tertiary_m)


tertiary_d.qpos= pos_tertiary_mujoco
mujoco.mj_forward(tertiary_m, tertiary_d)
tertiary_J=J(tertiary_m,tertiary_d,'contact_tertiary')


pos_index2= pos3[0:4].reshape(4)
pos_index2_mujoco=[pos_index2[1], pos_index2[0], pos_index2[2], pos_index2[3]]

pos_thumb2=pos_thumb2_mujoco= pos3[-4:].reshape(4)


index_model_path = '/home/saniya/LEAP/leap_hand_mujoco/model/leap hand/index_finger.xml'

index_m = mujoco.MjModel.from_xml_path(index_model_path)
index_d = mujoco.MjData(index_m)


index_d.qpos= pos_index2_mujoco
mujoco.mj_forward(index_m, index_d)
index_J2=J(index_m,index_d,'contact_index')


thumb_model_path = '/home/saniya/LEAP/leap_hand_mujoco/model/leap hand/thumb.xml'

thumb_m = mujoco.MjModel.from_xml_path(thumb_model_path)
thumb_d = mujoco.MjData(thumb_m)

thumb_d.qpos=pos_thumb2_mujoco
mujoco.mj_forward(thumb_m, thumb_d)
thumb_J2=J(thumb_m,thumb_d,'contact_thumb')

# print(index_J2)
# print(thumb_J2)
F_index1 = np.reshape([-0.20, 0, 0, 0, 0, 0], [6, 1])
F_thumb1=np.reshape([0.20, 0, 0, 0, 0, 0], [6, 1])

# Compute torque values
Tau_index1 = index_J.T @ F_index1
Tau_index1[[0, 1]] = Tau_index1[[1, 0]]

Tau_thumb1 = thumb_J.T @ F_thumb1

# Convert torque values to float
Tau_index1 = [float(torque[0]) for torque in Tau_index1]
Tau_thumb1 = [float(torque[0]) for torque in Tau_thumb1]


F_index2 = np.reshape([-0.15, 0, 0, 0, 0, 0], [6, 1])
F_thumb2=np.reshape([0.15, 0, 0, 0, 0, 0], [6, 1])

# Compute torque values
Tau_index2 = index_J.T @ F_index2
Tau_index2[[0, 1]] = Tau_index2[[1, 0]]

Tau_thumb2 = thumb_J.T @ F_thumb2

# Convert torque values to float
Tau_index2 = [float(torque[0]) for torque in Tau_index2]
Tau_thumb2 = [float(torque[0]) for torque in Tau_thumb2]


leap_pos = LeapNode_Poscontrol()

q0 = np.zeros(16) # 16-element array
v0 = np.zeros(16)  # initial velocity (array of zeros)
q1 = pos.reshape(16)  # final positions
v1 = np.zeros(16)  # final velocity (array of zeros)
t0 = 3 # initial time
t1 = 7

# q2=np.array([a1,b1,c1,d1, 0,0,0,0,0,0,0,0,e1,f1,g1,h1])
# v2=np.zeros(16)

start_time = time.time()

while time.time() - start_time < t0:
    leap_pos.set_allegro(np.zeros(16))
    # Create a new row to append
    new_row = pd.DataFrame({
        'Time': [time.time()],
        'start_time': [start_time],
        'Torque': [leap_pos.read_cur()],
        'Position': [leap_pos.read_pos()],
        'Velocity': [leap_pos.read_vel()],
        'Control_mode': ['pos'],
        'Subtask': ['start']
    })

    # Append the new row using concat
    if not new_row.empty:
        df = pd.concat([df, new_row], ignore_index=True)
    time.sleep(0.03)

while time.time() - start_time < t1 and time.time() - start_time>t0:
    current_time=time.time()
    # Ensure the current time falls within [t0, t1]
    
    qd = leap_pos.cubic_trajectory(q0, v0, q1, v1, t0, t1, current_time)
    leap_pos.set_allegro(qd)
    # Create a new row to append
    new_row = pd.DataFrame({
        'Time': [time.time()],
        'start_time': [start_time],
        'Torque': [leap_pos.read_cur()],
        'Position': [leap_pos.read_pos()],
        'Velocity': [leap_pos.read_vel()],
        'Control_mode': ['pos'],
        'Subtask': ['touch_object1']
    })

    # Append the new row using concat
    if not new_row.empty:
        df = pd.concat([df, new_row], ignore_index=True)
    # time.sleep(0.03)

leap_torque=LeapNode_Taucontrol()


# Apply torque
while True:
    curr_pos = leap_torque.read_pos_leap()
    
    # After updating Tau_thumb, store current position for the next time step
    Tau_thumb1[0] = 0.42*((pos[12] - curr_pos[12]))
    leap_torque.set_desired_torque(Tau_index1+ [0,-0.1,0,0,0,-0.1,0,0]+Tau_thumb1)
    # Create a new row to append
    new_row = pd.DataFrame({
        'Time': [time.time()],
        'start_time': [start_time],
        'Torque': [leap_torque.read_cur()],
        'Position': [leap_torque.read_pos()],
        'Velocity': [leap_torque.read_vel()],
        'Control_mode': ['Tau'],
        'Subtask': ['grasp_object1']
    })

    # Append the new row using concat
    if not new_row.empty:
        df = pd.concat([df, new_row], ignore_index=True)






# def J(model,data,site_name):
#         # model=mujoco.MjModel.from_xml_path(xml_path)
#         # data = mujoco.MjData(model)
#         mujoco.mj_forward(model, data)
#         jacp = np.zeros((3, model.nv))  # translation jacobian
#         jacr = np.zeros((3, model.nv)) 

#         site_id=model.site(site_name).id
#         mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

#         return np.vstack((jacp, jacr))

# tertiary_model_path = '/home/saniya/LEAP/leap_hand_mujoco/model/leap hand/tertiary.xml'

# tertiary_m = mujoco.MjModel.from_xml_path(tertiary_model_path)
# tertiary_d = mujoco.MjData(tertiary_m)

# tertiary_d.qpos=[1.4558066, 0.01386438,  0.7240976,   0.64433061]
# mujoco.mj_forward(tertiary_m, tertiary_d)
# tertiary_J=J(tertiary_m,tertiary_d,'contact_tertiary')
# print(tertiary_J)
