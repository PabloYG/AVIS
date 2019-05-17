import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_number = 2

v1_frames = ["V1 Frame 0", "V1 Frame 1", "V1 Frame 2", "V1 Frame 3",
              "V1 Frame 4", "V1 Frame 5", "V1 Frame 6"]
v2_frames = ["V2 Frame 0", "V2 Frame 1", "V2 Frame 2",
           "V2 Frame 3", "V2 Frame 4", "V2 Frame 5", "V2 Frame 6"]

harvest = np.array([[80, 240, 250, 390, 100, 40, 100],
                    [240, 90, 450, 160, 227, 80, 120],
                    [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                    [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                    [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                    [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])


fig, ax = plt.subplots()
im = ax.imshow(harvest)

# We want to show all ticks...
ax.set_xticks(np.arange(len(v1_frames)))
ax.set_yticks(np.arange(len(v2_frames)))
# ... and label them with the respective list entries
ax.set_xticklabels(v1_frames)
ax.set_yticklabels(v2_frames)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(v2_frames)):
    for j in range(len(v1_frames)):
        text = ax.text(j, i, harvest[i, j],
                       ha="center", va="center", color="w")

ax.set_title("Sequence to Sequence Alignment Cost Matrix")
fig.tight_layout()
plt.show()


