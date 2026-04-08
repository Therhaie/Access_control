import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Create two random 1024-dimension base vectors
np.random.seed(42)
vec_1 = np.random.randn(1024)
vec_2 = np.random.randn(1024)

steps = 40
cos_similarities = []
cos_sim_diff_index = []
cos_sim_2_same_index = []
cos_sim_1_high_value_untargeted = []
cos_sim_2_index_1_same_1_diff = []
evolution_values = []


for i in range(1, steps+1):
    # Modify same index
    evolving_values_1 = np.zeros(100)
    evolving_values_2 = np.zeros(100)
    evolving_values_1[0] = 10 ** (i/4)
    evolving_values_2[0] = 10 ** (i/4)

    v1 = np.concatenate([vec_1, evolving_values_1])
    v2 = np.concatenate([vec_2, evolving_values_2])
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    
    sim = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0, 0]
    cos_similarities.append(sim)
    evolution_values.append(10 ** (i/4))

    # Modify different index
    evolving_values_1 = np.zeros(100)
    evolving_values_2 = np.zeros(100)
    evolving_values_1[0] = 10 ** (i/4)
    evolving_values_2[1] = 10 ** (i/4)

    v1 = np.concatenate([vec_1, evolving_values_1])
    v2 = np.concatenate([vec_2, evolving_values_2])
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    sim_diff_index = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0, 0]
    cos_sim_diff_index.append(sim_diff_index)

    # 2 Same index modified and an extra one for 
    evolving_values_1 = np.zeros(100)
    evolving_values_2 = np.zeros(100)
    evolving_values_1[0] = 10 ** (i/4)
    evolving_values_1[1] = 10 ** (i/4)
    evolving_values_2[0] = 10 ** (i/4)
    evolving_values_2[1] = 10 ** (i/4)

    v1 = np.concatenate([vec_1, evolving_values_1])
    v2 = np.concatenate([vec_2, evolving_values_2])
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    sim_2_index = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0, 0]
    cos_sim_2_same_index.append(sim_2_index)

    # 2 index modified, 1 in common
    evolving_values_1 = np.zeros(100)
    evolving_values_2 = np.zeros(100)
    evolving_values_1[0] = 10 ** (i/4)
    evolving_values_1[1] = 10 ** (i/4)
    evolving_values_2[0] = 10 ** (i/4)
    evolving_values_2[2] = 10 ** (i/4)

    v1 = np.concatenate([vec_1, evolving_values_1])
    v2 = np.concatenate([vec_2, evolving_values_2])
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    sim_2_index = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0, 0]
    cos_sim_2_index_1_same_1_diff.append(sim_2_index)


    # 2 Same index modified and an extra one for 
    evolving_values_1 = np.zeros(100)
    evolving_values_2 = np.zeros(100)
    # evolving_values_1[0] = 10 ** (i/4)


    v1 = np.concatenate([vec_1, evolving_values_1])
    v2 = np.concatenate([vec_2, evolving_values_2])
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    sim_2_index = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0, 0]
    cos_sim_1_high_value_untargeted.append(sim_2_index)

# Step 4: Plot with log scale
plt.figure(figsize=(8, 5))
plt.plot(evolution_values, cos_similarities, marker='o', label='Same index modified')
plt.plot(evolution_values, cos_sim_diff_index, marker='x', label='Different index modified')
plt.plot(evolution_values, cos_sim_2_same_index, marker='s', label='Two same indices modified')
plt.plot(evolution_values, cos_sim_1_high_value_untargeted, marker='^', label='High value vs untargeted')
plt.plot(evolution_values, cos_sim_2_index_1_same_1_diff, marker='d', label='Two indices modified, 1 same 1 diff')
plt.xscale('log')
plt.xlabel('Evolving Value (log scale)')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarity Evolution (Log Scale)')
plt.grid(True, which="both", ls="-", linewidth=0.5)
plt.tight_layout()
plt.legend()
plt.savefig("AAcosine_similarity_evolution.png")
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics.pairwise import cosine_similarity

# # Step 1: Create two random 1024-dimension base vectors
# np.random.seed(42)
# vec_1 = np.random.randn(1024)
# vec_2 = np.random.randn(1024)
# # vec_2 = vec_1.copy()  # Identical vectors to start

# # Number of steps for the evolving 100-dim vector (skip 0 for log scale)
# steps = 10
# cos_similarities = []
# evolution_values = []

# # Step 2/3: Loop over values for the evolving 100-dim array
# for i in range(1, steps+1):  # start at 1!
#     # evolving_values_1 = np.zeros(100) 
#     # evolving_values_1 = np.ones(100) * i * 2  # Linearly increasing values (10, 20, ..., 100)
#     # evolving_values_2 = np.zeros(100) 
#     evolving_values_1 = np.zeros(100) 
#     evolving_values_1[0] = i * 10  
#     evolving_values_2 = np.zeros(100) 
#     evolving_values_2[0] = i * 10

#     v1 = np.concatenate([vec_1, evolving_values_1])
#     v2 = np.concatenate([vec_2, evolving_values_2])

#     v1 /= np.linalg.norm(v1)  # Normalize to prevent magnitude from dominating similarity
#     v2 /= np.linalg.norm(v2)

#     sim = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0, 0]
#     cos_similarities.append(sim)
#     evolution_values.append(i * 10)  # This is the value assigned to the evolving part

#     # plot for different index modified
#     evolving_values_1 = np.zeros(100)
#     evolving_values_1[0] = i * 10
#     evolving_values_2 = np.zeros(100)
#     evolving_values_2[1] = i * 10  # Different index modified
#     v1 = np.concatenate([vec_1, evolving_values_1])
#     v2 = np.concatenate([vec_2, evolving_values_2])
#     v1 /= np.linalg.norm(v1)
#     v2 /= np.linalg.norm(v2)
#     sim_diff_index = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0, 0]
#     # print(f"Value: {i*10}, Cosine Similarity (same index): {sim:.4f}, Cosine Similarity

# # Step 4: Plot with log scale
# plt.figure(figsize=(8, 5))
# plt.plot(evolution_values, cos_similarities, marker='o', label='Cosine Similarity same index modified')
# plt.plot(evolution_values, sim_diff_index, marker='x', label='Cosine Similarity different index modified')
# plt.xscale('log')
# plt.xlabel('Evolving Value (log scale)')
# plt.ylabel('Cosine Similarity')
# plt.title('Cosine Similarity Evolution (Log Scale)')
# plt.grid(True, which="both", ls="-", linewidth=0.5)
# plt.tight_layout()
# plt.legend()
# plt.savefig("AAcosine_similarity_evolution.png")
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics.pairwise import cosine_similarity

# from mpl_toolkits.mplot3d import Axes3D

# # Base vectors
# np.random.seed(42)
# vec_1 = np.random.randn(1024)
# vec_2 = vec_1.copy()

# # Settings
# steps_val = 20  # Number of distinct values to try (spread over a log scale)
# steps_zeros = 80  # Up to 100 zeros, in steps

# values = np.logspace(1, 5, steps_val)  # e.g., from 10^1 to 10^5
# num_zeros_list = np.linspace(0, 100, steps_zeros, dtype=int)

# X, Y = np.meshgrid(values, num_zeros_list)
# Z = np.zeros_like(X, dtype=float)

# for i, num_zeros in enumerate(num_zeros_list):
#     for j, val in enumerate(values):
#         evolving_values_1 = np.ones(100) * val
#         evolving_values_1[:num_zeros] = 0  # Set the first `num_zeros` elements to zero
#         evolving_values_2 = np.ones(100) * val

#         v1 = np.concatenate([vec_1, evolving_values_1])
#         v2 = np.concatenate([vec_2, evolving_values_2])

#         sim = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0, 0]
#         Z[i, j] = sim

# # Plot
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# # Surface plot
# surf = ax.plot_surface(np.log10(X), Y, Z, cmap='viridis', edgecolor='none')
# ax.set_xlabel('log10(Evolving Value)')
# ax.set_ylabel('Number of zeros in evolving_values')
# ax.set_zlabel('Cosine Similarity')
# ax.set_title('3D Impact of Zero Count and Value on Cosine Similarity')
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.tight_layout()
# plt.show()
# plt.savefig("AAcosine_similarity_3D.png")



# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# import plotly.graph_objs as go

# # --- Parameters ---
# vec_dim = 1024
# evolving_dim = 100
# num_value_steps = 90
# num_zero_steps = 91  # 0 to 50 inclusive
# max_value = 25
# # --- Vary: Value (avoid exact 0 for log scale) ---
# values = np.logspace(0, max_value, num_value_steps)  # From 1 to 1e9

# # --- Vary: Number of zeros ---
# num_zeros_list = np.arange(0, num_zero_steps)  # 0 to 50 inclusive

# # --- Meshgrid for plotting ---
# X, Y = np.meshgrid(values, num_zeros_list)
# Z = np.zeros_like(X, dtype=float)

# # --- Base vectors ---
# np.random.seed(42)
# vec_1 = np.random.randn(vec_dim)
# vec_2 = vec_1.copy()

# # --- Compute cosine similarities ---
# for i, num_zeros in enumerate(num_zeros_list):
#     for j, val in enumerate(values):
#         ev_1 = np.ones(evolving_dim) * val
#         ev_1[:num_zeros] = 0
#         ev_2 = np.ones(evolving_dim) * val
#         v1 = np.concatenate([vec_1, ev_1])
#         v2 = np.concatenate([vec_2, ev_2])
#         # normalize
#         v1 /= np.linalg.norm(v1)
#         v2 /= np.linalg.norm(v2)
#         sim = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0, 0]
#         Z[i, j] = sim

# # --- Plotly 3D Surface ---
# fig = go.Figure(
#     data=[
#         go.Surface(
#             z=Z,
#             x=np.log10(X),  # log10 scale for values
#             y=Y,
#             colorscale='Viridis'
#         )
#     ]
# )

# fig.update_layout(
#     title="3D Impact of Zero Count and Value on Cosine Similarity",
#     scene=dict(
#         xaxis_title='log10(Value in evolving_values_1)',
#         yaxis_title='Number of zeros in evolving_values_1',
#         zaxis_title='Cosine Similarity'
#     ),
# )

# fig.write_html("cosine_similarity_3d_norm.html")
# print("Interactive plot saved as cosine_similarity_3d.html.")