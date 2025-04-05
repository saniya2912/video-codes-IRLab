import numpy as np
import time
from main_victory import *  # Assuming main.py contains the LeapNode class

leap=LeapNode_Poscontrol()
q0=np.array([0.12891303,  2.01724325,  1.13673864,  0.59677731, -0.13032974,  2.17217527,
  0.30071889,  1.53864132, -0.14106764,  2.12462221,  0.34367047,  1.35302959,
 -0.34355296, -0.03215493,  0.78699075,  1.2364472 ])

victory=np.array([-0.36502875, -0.31133951, -0.01374711 , 0.00312673 , 0.1902722 , -0.11959185,
 -0.16407718,  0.0061947,   0.62745677,  1.48035036,  1.53250538 , 0.90203939,
  1.07384525,  1.23798119,  1.06157337,  1.18582569])

# leap.set_allegro(q0)

# # q0=leap.read_pos_leap()
v0= np.zeros(16)
v1=np.zeros(16)
t0=0
t1=5
start_time=time.time()

while time.time() - start_time < 5:
    current_time=time.time() - start_time 
    qd = leap.cubic_trajectory(q0, v0, victory, v1, t0, t1, current_time)
    leap.set_allegro(qd)

# print(leap.read_pos_leap())
# while True:
#     leap.set_allegro(np.zeros(16))