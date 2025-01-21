import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

#parameters
SPACE_SIZE = 50  # nxnxn
NUM_DRONES = 4   # Number of Drones
NUM_TAGS = 20    # Number of ArUco tags
STEPS = 100      # Number of simulation steps(can change it to try something)
DRONE_RANGE = 10 # Drone detection range (for the aruco)

#randomly spawn aruco
aruco_tags = np.random.uniform(0, SPACE_SIZE, size=(NUM_TAGS, 3))

#randomly
drones = np.random.uniform(0, SPACE_SIZE, size=(NUM_DRONES, 3))
drones_velocities = np.zeros_like(drones)

# Function to detect tags within a drone's range (3D)(kind of simulating a camera)
def detect_tags(drone_pos, tags, range_limit):
    distances = np.linalg.norm(tags - drone_pos, axis=1)
    detected_indices = np.where(distances <= range_limit)[0]
    return detected_indices

def is_explored(map, x, y, z):
    if 0 <= x < map.shape[0] and 0 <= y < map.shape[1] and 0 <= z < map.shape[2]:
        return map[x, y, z] == 1
    return True 
#frontier based explorartion
def find_frontiers(map):
    frontiers = []
    for x in range(1, map.shape[0] - 1):
        for y in range(1, map.shape[1] - 1):
            for z in range(1, map.shape[2] - 1):
                if is_explored(map, x, y, z):
                    unexplored_neighbors = 0
                    for dx, dy, dz in [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]:
                        if not is_explored(map, x + dx, y + dy, z + dz):
                            unexplored_neighbors += 1
                    if unexplored_neighbors > 0:
                        frontiers.append((x, y, z, unexplored_neighbors))
    frontiers.sort(key=lambda x: x[3], reverse=True)
    return frontiers


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([0, SPACE_SIZE])
ax.set_ylim([0, SPACE_SIZE])
ax.set_zlim([0, SPACE_SIZE])

# Initialize a 3D map
map = np.zeros((SPACE_SIZE, SPACE_SIZE, SPACE_SIZE), dtype=int)


detected_tags = set()

#main loop
for step in range(STEPS):
    ax.cla()
    ax.set_xlim([0, SPACE_SIZE])
    ax.set_ylim([0, SPACE_SIZE])
    ax.set_zlim([0, SPACE_SIZE])
    ax.set_title(f"Step: {step + 1}")

    for i, drone in enumerate(drones):
        # Update map based on drone positions (3D)
        x = int(np.floor(drone[0]))
        y = int(np.floor(drone[1]))
        z = int(np.floor(drone[2]))
        if 0 <= x < map.shape[0] and 0 <= y < map.shape[1] and 0 <= z < map.shape[2]:
            map[x, y, z] = 1

        frontiers = find_frontiers(map)

        if frontiers:
            frontier_target = frontiers[0][:3]
            target_pos = np.array([frontier_target[0] + 0.5, frontier_target[1] + 0.5, frontier_target[2] + 0.5])
            target_pos += np.random.uniform(-2,2, size = 3) #sample around the frontier
            target_pos = np.clip(target_pos, [0,0,0], [SPACE_SIZE,SPACE_SIZE,SPACE_SIZE]) #keep within map
        else:
            target_pos = np.random.uniform(0, SPACE_SIZE, size=3)

        direction = target_pos - drone
        if np.linalg.norm(direction) > 0:
            drones_velocities[i] = direction / np.linalg.norm(direction) * 2
            drones[i] += drones_velocities[i]
            drones[i] = np.clip(drones[i], 0, SPACE_SIZE)

        # Detect ArUco tags
        detected = detect_tags(drone, aruco_tags, DRONE_RANGE)
        detected_tags.update(detected)

        # Plot drone
        ax.scatter(*drone, color=f'C{i}', label=f'Drone {i + 1}')

    # Plot detected and undetected tags
    detected_indices = list(detected_tags)
    if detected_indices:
        ax.scatter(*aruco_tags[detected_indices].T, c='g', label='Detected Tags', s=50)
    undetected_tags = np.delete(aruco_tags, list(detected_tags), axis=0)
    if undetected_tags.size > 0:
        ax.scatter(*undetected_tags.T, c='r', label='Undetected Tags', s=50)

    # Plot explored voxels (3D) - This can be very slow for large maps
    x, y, z = np.where(map == 1)
    ax.scatter(x, y, z, c='b', marker='o', alpha=0.1, s=1) #alpha for transparency

    ax.legend()
    plt.pause(0.01)

# Print summary
print(f"Simulation complete. Total tags detected: {len(detected_tags)}/{NUM_TAGS}")
plt.show()