import numpy as np
from models import *

import torch
import torch.nn as nn
import torch.optim as optim
# -----------------------------
# Synthetic Data Generator
# -----------------------------

def generate_sample():

    img = np.zeros((64,64))

    center_shift = np.random.uniform(-8,8)

    y = np.linspace(5,58,10)

    left_lane = 20 + center_shift + 0.02*(y-32)**2
    right_lane = 40 + center_shift + 0.02*(y-32)**2

    for i in range(len(y)):
        x = int(left_lane[i])
        img[int(y[i]), x - 1:x + 2] = 1
        x = int(right_lane[i])
        img[int(y[i]), x - 1:x + 2] = 1

    lane_points = np.concatenate([left_lane, right_lane]) / 64.0
    steering = -center_shift / 8

    return img.astype(np.float32), lane_points.astype(np.float32), np.array([steering]).astype(np.float32)


def generate_dataset(n):

    images = []
    lanes = []
    steerings = []

    for _ in range(n):
        img, lane, steer = generate_sample()
        images.append(img)
        lanes.append(lane)
        steerings.append(steer)

    images = np.array(images)[:,None,:,:]
    lanes = np.array(lanes)
    steerings = np.array(steerings)

    return torch.tensor(images), torch.tensor(lanes), torch.tensor(steerings)


# -----------------------------
# Generate Data
# -----------------------------

train_images, train_lanes, train_steer = generate_dataset(2000)
test_images, test_lanes, test_steer = generate_dataset(200)


# -----------------------------
# Train CNN (Lane Detection)
# -----------------------------

lane_model = LaneCNN()

criterion = nn.MSELoss()
optimizer = optim.Adam(lane_model.parameters(), lr=0.001)

print("Training lane detector...")

for epoch in range(50):

    pred = lane_model(train_images)
    loss = criterion(pred, train_lanes)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 2 == 0:
        print("epoch", epoch, "loss", loss.item())


# -----------------------------
# Train Steering Model
# -----------------------------

steer_model = SteeringModel()

optimizer = optim.Adam(steer_model.parameters(), lr=0.001)

print("\nTraining steering model...")

for epoch in range(1000):

    with torch.no_grad():
        lane_pred = lane_model(train_images)   # CNN output

    center = (lane_pred[:, :10] + lane_pred[:, 10:]) / 2

    steering_pred = steer_model(center)

    loss = criterion(steering_pred, train_steer)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print("epoch", epoch, "loss", loss.item())


torch.save({
    "train_images": train_images,
    "train_lanes": train_lanes,
    "train_steer": train_steer,
    "test_images": test_images,
    "test_lanes": test_lanes,
    "test_steer": test_steer
}, "dataset.pth")
torch.save(lane_model.state_dict(), "lane_model.pth")
torch.save(steer_model.state_dict(), "steer_model.pth")
print("Data and Models saved!")