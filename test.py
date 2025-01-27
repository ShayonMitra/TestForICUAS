import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Voronoi
import random

# Parameters
SPACE_SIZE = 30
NUM_DRONES = 4
NUM_TAGS = 20
STEPS = 100
DRONE_RANGE = 10
STEP_SIZE = 1.5

# Randomly spawn ArUco tags
aruco_tags = np.random.uniform(0, SPACE_SIZE, size=(NUM_TAGS, 3))

# Initialize drones
drones = [{"position": np.random.uniform(0, SPACE_SIZE, size=3), "voronoi_region": None} for _ in range(NUM_DRONES)]
detected_tags = set()
total_reward = 0

# Functions
def detect_tags(drone_pos, tags, range_limit):
    distances = np.linalg.norm(tags - drone_pos, axis=1)
    detected_indices = np.where(distances <= range_limit)[0]
    return detected_indices

def is_valid_point(point):
    return all(0 <= coord <= SPACE_SIZE for coord in point)

def get_voronoi_regions(points):
    vor = Voronoi(points)
    regions = []
    for r in vor.regions:
        if not -1 in r and len(r) > 0: #valid region
            polygon = []
            for i in r:
                v = vor.vertices[i]
                if is_valid_point(v):
                    polygon.append(v)
                else:
                    polygon = None
                    break
            if polygon:
                regions.append(np.array(polygon))
    return regions

def get_centroid(region):
    if region is not None and len(region) > 0:
        return np.mean(region, axis=0)
    return None

# Plotting setup
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([0, SPACE_SIZE])
ax.set_ylim([0, SPACE_SIZE])
ax.set_zlim([0, SPACE_SIZE])

# Main loop
for step in range(STEPS):
    ax.cla()
    ax.set_xlim([0, SPACE_SIZE])
    ax.set_ylim([0, SPACE_SIZE])
    ax.set_zlim([0, SPACE_SIZE])
    ax.set_title(f"Step: {step + 1}")

    drone_positions = [drone["position"] for drone in drones]
    voronoi_regions = get_voronoi_regions(drone_positions)

    for i, drone in enumerate(drones):
        drone["voronoi_region"] = voronoi_regions[i] if i < len(voronoi_regions) else None

        # Move towards centroid of Voronoi region
        if drone["voronoi_region"] is not None:
            target_pos = get_centroid(drone["voronoi_region"])
            if target_pos is not None:
                direction = target_pos - drone["position"]
                norm = np.linalg.norm(direction)
                if norm > 0:
                    drone["position"] += direction / norm * min(norm, STEP_SIZE)
                    drone["position"] = np.clip(drone["position"], 0, SPACE_SIZE)
        else: #if no voronoi region, move randomly
            target_pos = np.random.uniform(0, SPACE_SIZE, size=3)
            direction = target_pos - drone["position"]
            norm = np.linalg.norm(direction)
            if norm > 0:
                drone["position"] += direction / norm * min(norm, STEP_SIZE)
                drone["position"] = np.clip(drone["position"], 0, SPACE_SIZE)


        # Tag detection and reward
        detected = detect_tags(drone["position"], aruco_tags, DRONE_RANGE)
        newly_detected = [tag for tag in detected if tag not in detected_tags]
        total_reward += len(newly_detected)
        detected_tags.update(detected)

        # Plot drone
        ax.scatter(*drone["position"], color=f'C{i}', label=f'Drone {i + 1}')

    # Plot tags
    detected_indices = list(detected_tags)
    if detected_indices:
        ax.scatter(*aruco_tags[detected_indices].T, c='g', label='Detected Tags', s=50)
    undetected_tags = np.delete(aruco_tags, list(detected_tags), axis=0)
    if undetected_tags.size > 0:
        ax.scatter(*undetected_tags.T, c='r', label='Undetected Tags', s=50)

    plt.pause(0.01)

# Print summary
print(f"Simulation complete. Total tags detected: {len(detected_tags)}/{NUM_TAGS}")

plt.show()