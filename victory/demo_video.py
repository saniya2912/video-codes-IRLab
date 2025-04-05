# import numpy as np
# import time
# from main import LeapNode  # Assuming main.py contains the LeapNode class

# # Define reasonable limits for the joint positions (modify as needed)
# LEAP_MIN = np.zeros(16)  # Assuming the minimum joint position is 0
# LEAP_MAX = np.full(16, 1.57)  # Assuming the maximum joint position is 180

# def generate_random_pose():
#     """Generates a random pose within the LEAP Hand's joint limits."""
#     return np.random.uniform(LEAP_MIN, LEAP_MAX)

# def main():
#     leap_hand = LeapNode()
#     print("Starting random pose execution...")
    
#     try:
#         while True:
#             random_pose = generate_random_pose()
#             leap_hand.set_allegro(random_pose)
#             print("Sent pose:", random_pose)
#             time.sleep(1.5)  # Adjust the delay as needed
#     except KeyboardInterrupt:
#         print("Stopping random pose execution.")

# if __name__ == "__main__":
#     main()

import numpy as np
import time
from main import LeapNode  # Assuming main.py contains the LeapNode class

class LeapHandController:
    def __init__(self):
        self.leap_hand = LeapNode()
        self.states = {
            "rock": np.array([3.1416, 4.1888, 4.5553, 4.4157, 3.1416, 4.1190, 5.1487, 4.2412, 3.1416, 4.2237, 4.7124, 4.4506, 3.14, 1.5184, 0.76, 0.76]),
            "paper": np.array([3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416]),
            "scissors": np.array([3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 4.2237, 4.7124, 4.4506, 3.14, 1.5184, 0.76, 0.76])
        }
    
    def send_random_state(self):
        state_name = np.random.choice(list(self.states.keys()))
        pose = self.states[state_name]
        self.leap_hand.set_leap(pose)
        print(f"Sent pose: {state_name}")


def main():
    controller = LeapHandController()
    print("Starting state-based pose execution...")
    
    try:
        while True:
            controller.send_random_state()
            time.sleep(1.5)  # Adjust the delay as needed
    except KeyboardInterrupt:
        print("Stopping state-based pose execution.")

if __name__ == "__main__":
    main()
