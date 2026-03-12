
import numpy as np

import matplotlib.pyplot as plt
from models import *



# -----------------------------
# Evaluation + Visualization
# -----------------------------
lane_model = LaneCNN()
steer_model = SteeringModel()

# load data
data = torch.load("dataset.pth")
test_images = data["test_images"]
test_lanes = data["test_lanes"]
test_steer = data["test_steer"]

# load models
lane_model.load_state_dict(torch.load("lane_model.pth"))
steer_model.load_state_dict(torch.load("steer_model.pth"))

lane_model.eval()
steer_model.eval()

img = test_images[0:1]

with torch.no_grad():

    pred_lanes = lane_model(img)

    center_pred = (pred_lanes[:, :10] + pred_lanes[:, 10:]) / 2
    steering = steer_model(center_pred)

num_point = 8 # max is 10 in the testing data
# convert to numpy for plotting
pred_lanes_plot = pred_lanes.numpy().reshape(2,10) * 64
center_plot = (pred_lanes_plot[0] + pred_lanes_plot[1]) / 2
center_plot = center_plot[:num_point]

pred_lanes = pred_lanes.numpy().reshape(2,10) * 64
pred_lanes = pred_lanes[:, :num_point]
gt = test_lanes[0].numpy().reshape(2,10) * 64
gt = gt[:, :num_point]
y = np.linspace(5,58,num_point)

dx = steering.item() * 8
dy = -4

center_line = np.full_like(y, 32)

# create blank background instead of showing pixel squares
background = np.zeros((64,64))

print(center_pred.min(), center_pred.max())

fig, axs = plt.subplots(2,2, figsize=(10,10))
ax1, ax2, ax3, ax4 = axs.flatten()

# -------------------------------------------------
# 1 Ground Truth
# -------------------------------------------------

ax1.imshow(background, cmap="gray")

ax1.scatter(gt[0], y, c="white", marker="x", label="GT left lane")
ax1.scatter(gt[1], y, c="yellow", marker="x", label="GT right lane")

ax1.set_title("Ground Truth Lanes")
ax1.legend()

# -------------------------------------------------
# 2 CNN Predictions
# -------------------------------------------------

ax2.imshow(background, cmap="gray")

ax2.scatter(pred_lanes[0], y, c="red", label="pred left lane")
ax2.scatter(pred_lanes[1], y, c="blue", label="pred right lane")

ax2.scatter(gt[0], y, c="white", marker="x")
ax2.scatter(gt[1], y, c="yellow", marker="x")

ax2.set_title("CNN Lane Predictions")
ax2.legend()

# -------------------------------------------------
# 3 Steering Output
# -------------------------------------------------
ax3.imshow(background, cmap="gray")

vehicle_x = center_plot
vehicle_y = y

ax3.scatter(vehicle_x, vehicle_y, c="white", s=25, label="vehicle positions")
ax3.scatter(pred_lanes[0], y, c="red", label="pred left lane")
ax3.scatter(pred_lanes[1], y, c="blue", label="pred right lane")

ax3.scatter(gt[0], y, c="white", marker="x")
ax3.scatter(gt[1], y, c="yellow", marker="x")


ax3.set_title("vehicle Testing Positions")
ax3.legend()
# -------------------------------------------------
# 4 Full Pipeline
# -------------------------------------------------

ax4.imshow(background, cmap="gray")

ax4.scatter(pred_lanes[0], y, c="red", label="pred left lane")
ax4.scatter(pred_lanes[1], y, c="blue", label="pred right lane")

ax4.scatter(gt[0], y, c="white", marker="x")
ax4.scatter(gt[1], y, c="yellow", marker="x")
ax4.scatter(vehicle_x, vehicle_y, c="white", s=25, label="vehicle positions")

for i in range(len(y)-1):

    # build partial centerline input
    center_partial = center_pred.clone()
    center_partial[:, i+1:] = center_partial[:, i].unsqueeze(1)

    # predict steering for this position
    steering_i = steer_model(center_partial).item()
    dx = steering.item()

    ax4.arrow(
        vehicle_x[i],
        vehicle_y[i],
        dx,
        -y[i] /6,
        color="cyan",
        width=0.1,
        head_width=1,
        length_includes_head=True,
    )

ax4.arrow(
        vehicle_x[-1],
        vehicle_y[-1],
        dx,
        -y[i] /6,
        color="cyan",
        width=0.1,
        head_width=1,
        length_includes_head=True,
        label="vehicle steering output"
    )
ax4.set_title("Full Output ")

plt.tight_layout()
plt.show()