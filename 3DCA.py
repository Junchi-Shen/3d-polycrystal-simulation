import cupy as cp
import random # Keep standard random for single choices if needed, but prefer cp.random
import os
import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy # For loading the initial grid if it's a numpy file

# Constants for cell types (use integers)
CELL_TYPE_M0 = 0  # Metal Grain
CELL_TYPE_M1 = 1  # Metal Boundary
CELL_TYPE_M2 = 2  # Metal Precipitate
CELL_TYPE_M_OTHER = 3 # Assuming this was another metal type in original probabilities? Or is it N? The description says N=3, but probabilities list 3. Let's assume N=4 based on description/init.
CELL_TYPE_N = 4   # Neutral Solution
CELL_TYPE_S = 5   # Acidic Solute
CELL_TYPE_P = 6   # Corrosion Product

# --- GPU Helper Functions ---

# Note: get_moore_neighbors is implicitly handled by index calculations below.

def corrosion_occurs_gpu(neighbor_types, probabilities_map, grid_shape):
    """
    Determines corrosion based on neighbor types using GPU arrays.

    Args:
        neighbor_types (cp.ndarray): Array of neighbor cell types at potential new locations.
        probabilities_map (dict): Dictionary mapping metal types to probabilities.
        grid_shape: The shape of the corrosion grid. Needed for random number generation.

    Returns:
        cp.ndarray: Boolean array indicating where corrosion occurs.
        cp.ndarray: Array of the types of cells that will be corroded.
    """
    # Create probability array based on neighbor types
    probs = cp.zeros_like(neighbor_types, dtype=cp.float32)
    metal_mask = cp.zeros_like(neighbor_types, dtype=bool)

    for metal_type, prob in probabilities_map.items():
         # Check if the neighbor type is one of the metals susceptible to corrosion
         # Assuming M types are 0, 1, 2. Adjust if M_OTHER (3) is also a corrodible metal.
        if metal_type in [CELL_TYPE_M0, CELL_TYPE_M1, CELL_TYPE_M2]:
            type_mask = (neighbor_types == metal_type)
            probs[type_mask] = prob
            metal_mask |= type_mask # Mark as a metal interaction

    # Generate random numbers for comparison
    random_values = cp.random.rand(*neighbor_types.shape, dtype=cp.float32)

    # Corrosion occurs where the neighbor is metal AND random value < probability
    corrodes = (random_values < probs) & metal_mask
    corroded_cell_types = neighbor_types[corrodes] # Get types of cells that actually corroded

    return corrodes, corroded_cell_types


def turn_random_n_to_s_gpu(corr_grid, solution_thickness, num_to_convert):
    """
    Randomly converts 'num_to_convert' N cells to S cells in the solution layer.
    """
    if num_to_convert <= 0:
        return

    # Find indices of N cells ONLY in the solution part
    solution_slice = corr_grid[:solution_thickness, :, :]
    n_indices_local = cp.argwhere(solution_slice == CELL_TYPE_N)

    num_available_n = n_indices_local.shape[0]
    num_to_actually_convert = min(num_to_convert, num_available_n)

    if num_to_actually_convert > 0:
        # Choose random indices from the list of N cells
        chosen_indices_flat = cp.random.choice(
            cp.arange(num_available_n), size=num_to_actually_convert, replace=False
        )
        chosen_indices_3d = n_indices_local[chosen_indices_flat]

        # Convert chosen N cells to S cells
        # Need to access the original grid with these indices
        corr_grid[chosen_indices_3d[:, 0], chosen_indices_3d[:, 1], chosen_indices_3d[:, 2]] = CELL_TYPE_S

# --- Main Simulation Functions (GPU Version) ---

def initialize_corr_grid_gpu(bound_grid_np, concentration_ratio, solution_thickness):
    """Initializes the corrosion grid on the GPU."""
    n_metal, m, l = bound_grid_np.shape
    n_total = n_metal + solution_thickness
    grid_shape = (n_total, m, l)

    # Create grid on GPU, initialize upper part to N
    corr_grid = cp.full(grid_shape, CELL_TYPE_N, dtype=cp.int32)

    # Calculate number of S cells
    total_solution_cells = solution_thickness * m * l
    num_s_cells = int(total_solution_cells * concentration_ratio)

    if num_s_cells > 0:
        # Generate random flat indices for S cells within the solution layer volume
        s_indices_flat = cp.random.choice(cp.arange(total_solution_cells), size=num_s_cells, replace=False)

        # Convert flat indices to 3D coordinates within the solution layer
        x_indices = s_indices_flat // (m * l)
        remainder = s_indices_flat % (m * l)
        y_indices = remainder // l
        z_indices = remainder % l

        # Place S cells
        corr_grid[x_indices, y_indices, z_indices] = CELL_TYPE_S

    # Place metal part - transfer numpy array to GPU
    bound_grid_gpu = cp.asarray(bound_grid_np, dtype=cp.int32)
    corr_grid[solution_thickness:, :, :] = bound_grid_gpu

    return corr_grid


def random_walk_s_gpu(corr_grid, probabilities_map, solution_thickness):
    """Performs the random walk and corrosion step for S cells on the GPU."""
    n_total, m, l = corr_grid.shape
    grid_shape = corr_grid.shape

    # 1. Find all S cells
    s_indices = cp.argwhere(corr_grid == CELL_TYPE_S) # Shape: (num_s, 3)
    num_s = s_indices.shape[0]
    if num_s == 0:
        return 0 # No S cells to move

    x, y, z = s_indices[:, 0], s_indices[:, 1], s_indices[:, 2]

    # 2. Generate random directions for all S cells
    # 0:l, 1:r, 2:u, 3:d, 4:f, 5:b
    directions = cp.random.randint(0, 6, size=num_s, dtype=cp.int32)

    # 3. Calculate potential new positions
    new_x, new_y, new_z = x.copy(), y.copy(), z.copy()

    # Note: CuPy doesn't directly support fancy indexing assignment with conditions easily like numpy.
    # We calculate potential new coordinates based on direction.
    new_y[directions == 0] = (y[directions == 0] - 1) % m # Left
    new_y[directions == 1] = (y[directions == 1] + 1) % m # Right
    new_x[directions == 2] = cp.maximum(0, x[directions == 2] - 1) # Up
    new_x[directions == 3] = cp.minimum(n_total - 1, x[directions == 3] + 1) # Down (careful with boundary)
    new_z[directions == 4] = (z[directions == 4] - 1) % l # Front
    new_z[directions == 5] = (z[directions == 5] + 1) % l # Back

    # --- Interaction Logic ---
    # This is where race conditions are tricky with simple array methods.
    # We'll do a simplified version: check target cell, decide action.
    # A two-buffer approach or atomics would be more robust.

    # Get types of cells at the *potential* new locations
    target_cell_types = corr_grid[new_x, new_y, new_z]

    # A. Identify S cells attempting to move into N or S cells (potential swap)
    swap_mask = (target_cell_types == CELL_TYPE_N) | (target_cell_types == CELL_TYPE_S)

    # B. Identify S cells attempting to move into Metal cells (potential corrosion)
    # Assuming M types are 0, 1, 2. Adjust if M_OTHER (3) is also corrodible.
    metal_interaction_mask = (target_cell_types == CELL_TYPE_M0) | \
                             (target_cell_types == CELL_TYPE_M1) | \
                             (target_cell_types == CELL_TYPE_M2)
                             # Add | (target_cell_types == CELL_TYPE_M_OTHER) if needed

    # C. Determine where corrosion actually occurs for metal interactions
    corrosion_mask = cp.zeros(num_s, dtype=bool)
    num_corroded = 0
    if cp.any(metal_interaction_mask):
         # Get types of metal cells targeted
        targeted_metal_types = target_cell_types[metal_interaction_mask]
        # Check probabilities only for those interacting with metals
        corrodes_subset, _ = corrosion_occurs_gpu(targeted_metal_types, probabilities_map, grid_shape)
         # Update the overall corrosion mask
        corrosion_mask[metal_interaction_mask] = corrodes_subset
        num_corroded = int(cp.sum(corrosion_mask)) # Count how many corrosions happened


    # --- Update Grid ---
    # Apply updates. WARNING: Potential race conditions here if multiple S update same cell!
    # A more robust method would write results to a *new* grid.

    # Apply Swaps: Where swap_mask is true, swap S with N/S at target
    # This is hard without atomics or buffers. Simple approach:
    # S becomes target type, target becomes S. This might overwrite another S's move.
    if cp.any(swap_mask):
        original_s_types = corr_grid[x[swap_mask], y[swap_mask], z[swap_mask]] # Should be S=5
        target_types_for_swap = corr_grid[new_x[swap_mask], new_y[swap_mask], new_z[swap_mask]] # N=4 or S=5

        corr_grid[x[swap_mask], y[swap_mask], z[swap_mask]] = target_types_for_swap
        corr_grid[new_x[swap_mask], new_y[swap_mask], new_z[swap_mask]] = original_s_types # Put S=5 in new location

    # Apply Corrosion: Where corrosion_mask is true
    # S -> N (at original position), M -> P (at new position)
    if cp.any(corrosion_mask):
        corr_grid[x[corrosion_mask], y[corrosion_mask], z[corrosion_mask]] = CELL_TYPE_N  # Original S becomes N
        corr_grid[new_x[corrosion_mask], new_y[corrosion_mask], new_z[corrosion_mask]] = CELL_TYPE_P  # Target M becomes P

    # Return the count of corrosion events for N->S conversion later
    return num_corroded


def move_corrosion_products_gpu(corr_grid, p_Pmove):
    """Moves corrosion product (P) cells randomly to adjacent N or S cells."""
    n_total, m, l = corr_grid.shape

    # 1. Find all P cells
    p_indices = cp.argwhere(corr_grid == CELL_TYPE_P)
    num_p = p_indices.shape[0]
    if num_p == 0:
        return

    x, y, z = p_indices[:, 0], p_indices[:, 1], p_indices[:, 2]

    # 2. Decide which P cells will attempt to move
    move_attempt_mask = cp.random.rand(num_p, dtype=cp.float32) < p_Pmove
    num_to_move = int(cp.sum(move_attempt_mask))

    if num_to_move == 0:
        return

    # Get indices of P cells that will attempt to move
    move_indices = p_indices[move_attempt_mask]
    x_move, y_move, z_move = move_indices[:, 0], move_indices[:, 1], move_indices[:, 2]

    # 3. For each moving P, find valid neighbors (N or S)
    # This is the trickiest part with pure array ops. Finding neighbors and
    # randomly choosing one per P cell is complex.
    # We will use a simplified approach: randomly choose *one* direction
    # and check if that neighbor is valid (N or S).

    # Generate random directions (0-5) for P cells attempting to move
    directions = cp.random.randint(0, 6, size=num_to_move, dtype=cp.int32)

    # Calculate potential neighbor coordinates
    nx, ny, nz = x_move.copy(), y_move.copy(), z_move.copy()
    nx[directions == 0] = (x_move[directions == 0] - 1) # Use cp.maximum(0, ...) if boundary needed
    nx[directions == 1] = (x_move[directions == 1] + 1) # Use cp.minimum(n-1, ...) if boundary needed
    ny[directions == 2] = (y_move[directions == 2] - 1) % m
    ny[directions == 3] = (y_move[directions == 3] + 1) % m
    nz[directions == 4] = (z_move[directions == 4] - 1) % l
    nz[directions == 5] = (z_move[directions == 5] + 1) % l

    # Check boundary conditions for x (up/down) explicitly
    nx = cp.clip(nx, 0, n_total - 1) # Ensure x stays within grid bounds

    # Get types of the randomly chosen neighbors
    neighbor_types = corr_grid[nx, ny, nz]

    # 4. Identify successful moves (neighbor is N or S)
    successful_move_mask = (neighbor_types == CELL_TYPE_N) | (neighbor_types == CELL_TYPE_S)

    # 5. Perform the swap for successful moves
    # Again, potential race conditions if multiple P target the same N/S.
    if cp.any(successful_move_mask):
        # Indices of P cells that moved successfully
        x_success = x_move[successful_move_mask]
        y_success = y_move[successful_move_mask]
        z_success = z_move[successful_move_mask]

        # Indices of the target N/S cells
        nx_success = nx[successful_move_mask]
        ny_success = ny[successful_move_mask]
        nz_success = nz[successful_move_mask]

        # Get the values to swap
        original_p_types = corr_grid[x_success, y_success, z_success] # Should be P=6
        target_types_for_swap = corr_grid[nx_success, ny_success, nz_success] # N=4 or S=5

        # Perform the swap
        corr_grid[x_success, y_success, z_success] = target_types_for_swap # Old P location gets N/S
        corr_grid[nx_success, ny_success, nz_success] = original_p_types  # Target N/S location gets P

# --- Visualization (Stays mostly the same, needs data transfer) ---

def visualize_corr_grid(corr_grid_gpu, step, z_slice_index=50, show_plot=False, save_plot=True, save_dir="corrosion_plots_gpu"):
    """Visualizes a slice of the grid. Transfers data from GPU to CPU."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(6, 6))
    # Define the colormap based on cell type values
    # 0:M0(gray), 1:M1(blue), 2:M2(yellow), 3:M_OTHER?(red), 4:N(white), 5:S(purple), 6:P(black)
    # Adjust colors as needed
    cmap = ListedColormap(['gray', 'blue', 'yellow', 'red', 'white', 'purple', 'black'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5] # Define boundaries for discrete colors
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    # Get the slice from GPU -> Transfer to CPU using .get() or cp.asnumpy()
    grid_slice_cpu = cp.asnumpy(corr_grid_gpu[:, :, z_slice_index])

    plt.imshow(grid_slice_cpu, cmap=cmap, norm=norm, interpolation='nearest')
    plt.title(f"GPU Corrosion Step: {step} (Slice Z={z_slice_index})")

    # Create a colorbar with labels
    cbar = plt.colorbar(ticks=[0, 1, 2, 3, 4, 5, 6])
    cbar.set_ticklabels(['M0', 'M1', 'M2', 'M?', 'N', 'S', 'P']) # Adjust 'M?' label if type 3 is used

    if save_plot:
        plt.savefig(os.path.join(save_dir, f"corrosion_step_{step:05d}.png"))

    if show_plot:
        plt.show()
    else:
        plt.close()

# --- Main Simulation Loop (GPU Version) ---

def simulate_corrosion_gpu(bound_grid_np, steps, concentration_ratio, solution_thickness, corrosion_base_probabilities, p_Pmove, vis_interval=100, vis_slice=50):
    """Main simulation loop using GPU acceleration."""

    print("Initializing grid on GPU...")
    corr_grid_gpu = initialize_corr_grid_gpu(bound_grid_np, concentration_ratio, solution_thickness)
    print("Grid initialized.")

    n_total, m, l = corr_grid_gpu.shape
    # Ensure probabilities map uses integer keys defined earlier
    prob_map = {int(k): v for k, v in corrosion_base_probabilities.items()}

    print("Starting simulation...")
    for step in tqdm.tqdm(range(steps)):
        # Step 1: S cell random walk and corrosion
        num_corroded = random_walk_s_gpu(corr_grid_gpu, prob_map, solution_thickness)

        # Step 2: Convert N cells to S cells based on corrosion count
        # This happens *after* all walks/corrosions for the step are done.
        if num_corroded > 0:
             turn_random_n_to_s_gpu(corr_grid_gpu, solution_thickness, num_corroded)

        # Step 3: Move corrosion products (P cells)
        move_corrosion_products_gpu(corr_grid_gpu, p_Pmove)

        # Step 4: Visualization (periodically)
        if step % vis_interval == 0:
            visualize_corr_grid(corr_grid_gpu, step, z_slice_index=vis_slice, show_plot=False, save_plot=True) # Show plot can be True for debugging

    print("Simulation finished.")
    visualize_corr_grid(corr_grid_gpu, steps, z_slice_index=vis_slice, show_plot=True, save_plot=True) # Show final state


# --- Parameters and Execution ---
if __name__ == '__main__':
    # Ensure CuPy is installed and a compatible GPU is available

    # Parameters (adjust as needed)
    concentration_ratio = 0.2
    solution_thickness = 10 # Increased thickness for better visualization/effect
    # Probabilities: M0, M1, M2, M_OTHER/N? -> Check what type 3 represents
    # If type 3 is N (Neutral), it shouldn't corrode. If it's another metal, assign probability.
    # Original code had {0: 0.0002, 1: 0.6, 2: 0.05, 3: 0.8}. Let's assume 3 was intended to be N=4 initially and had 0 prob.
    # And let's add probability for N->S conversion if S hits N? No, description says S+M->P+N
    # So, only metal types 0, 1, 2 should have non-zero probability.
    corrosion_base_probabilities = {
        CELL_TYPE_M0: 0.0002, # Grain Al
        CELL_TYPE_M1: 0.6,    # Boundary Al
        CELL_TYPE_M2: 0.05    # Precipitate
        # CELL_TYPE_M_OTHER: 0.0 # If there's another metal type
    }
    p_Pmove = 0.2 # Probability for P cell to move
    simulation_steps = 5000 # Number of simulation steps
    visualization_interval = 100 # Visualize every N steps
    visualization_slice_z = 50 # Z-slice index to visualize

    # Load the initial metal grid (NumPy array)
    # Make sure the path is correct and the file exists
    try:
        # Corrected path assuming 'G:' drive and '接活' folder
        bound_grid_path = r"G:\接活\grid_alloy_100_20_0.8.npy"
        bound_grid_np = numpy.load(bound_grid_path)
        print(f"Loaded metal grid from {bound_grid_path} with shape: {bound_grid_np.shape}")

        # Adjust visualization slice if grid dimensions are smaller
        if visualization_slice_z >= bound_grid_np.shape[2]:
            visualization_slice_z = bound_grid_np.shape[2] // 2 # Use middle slice if specified is out of bounds
            print(f"Adjusted visualization Z slice to: {visualization_slice_z}")

    except FileNotFoundError:
        print(f"Error: Metal grid file not found at {bound_grid_path}")
        print("Please ensure the .npy file exists and the path is correct.")
        exit()
    except Exception as e:
        print(f"Error loading grid: {e}")
        exit()


    # Run the simulation
    simulate_corrosion_gpu(
        bound_grid_np,
        steps=simulation_steps,
        concentration_ratio=concentration_ratio,
        solution_thickness=solution_thickness,
        corrosion_base_probabilities=corrosion_base_probabilities,
        p_Pmove=p_Pmove,
        vis_interval=visualization_interval,
        vis_slice=visualization_slice_z
    )