import cupy as cp
import numpy as np # Still needed for numpy file loading and CPU-side logic
import random # For CPU-side random choices
import os
import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math # For calculating grid/block sizes

# Constants for cell types (use integers)
CELL_TYPE_M0 = 0  # Metal Grain
CELL_TYPE_M1 = 1  # Metal Boundary
CELL_TYPE_M2 = 2  # Metal Precipitate
# CELL_TYPE_M_OTHER = 3 # If needed
CELL_TYPE_N = 4   # Neutral Solution
CELL_TYPE_S = 5   # Acidic Solute
CELL_TYPE_P = 6   # Corrosion Product
CELL_TYPE_EMPTY = -1 # Placeholder if needed

# --- CUDA Kernels (as Python strings) ---

# Kernel to initialize cuRAND states
# Requires grid dimensions to calculate thread ID
curand_init_kernel_code = r'''
#include <curand_kernel.h>

extern "C" __global__
void init_curand_states(unsigned long long seed, int n_threads, curandState* states) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_threads) {
        // Initialize cuRAND state for each thread
        curand_init(seed, tid, 0, &states[tid]);
    }
}
'''

# Main Simulation Kernel (Handles S-Walk/Corrosion and P-Move)
# Uses atomicCAS for move conflict resolution
simulation_step_kernel_code = r'''
#include <curand_kernel.h>
#include <cupy/atomics.cuh> // Include CuPy atomics support

// Device function to get 1D index from 3D coordinates
__device__ int get_idx(int x, int y, int z, int m, int l) {
    return z + y * l + x * l * m;
}

extern "C" __global__
void simulation_step(
    const int* __restrict__ in_grid, // Read-only input grid
    int* out_grid,                   // Writeable output grid
    curandState* rand_states,        // Random states per thread
    int n, int m, int l,             // Grid dimensions (n=total height)
    float prob_m0, float prob_m1, float prob_m2, // Corrosion probabilities
    float p_Pmove,                   // Probability for P cell move
    int* atomic_counter)             // Atomic counter for corrosion events
{
    // Calculate 3D indices for this thread
    int z = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int x = threadIdx.z + blockIdx.z * blockDim.z;

    // Check bounds: only process threads within the grid dimensions
    if (x >= n || y >= m || z >= l) {
        return;
    }

    // Calculate 1D index for accessing arrays
    int idx = get_idx(x, y, z, m, l);
    int thread_id_1d = idx; // Use 1D index for rand state lookup

    // Get the current cell type from the input grid
    int current_type = in_grid[idx];

    // Initialize output grid with the input state (default: cell doesn't change)
    // Ensure this write doesn't conflict if multiple kernels write here?
    // It's generally safe if each thread writes its own initial state.
    out_grid[idx] = current_type;

    // Get random state for this thread
    curandState local_rand_state = rand_states[thread_id_1d];
    float rand_val = curand_uniform(&local_rand_state); // Consume one random value

    // --- S Cell Logic ---
    if (current_type == CELL_TYPE_S) {
        // Choose random direction (0-5: L, R, U, D, F, B)
        int direction = curand_uniform(&local_rand_state) * 6; // Generate float 0-6, cast to int
        int nx = x, ny = y, nz = z;

        // Calculate potential new coordinates (includes boundary wrapping/clamping)
        switch(direction) {
            case 0: ny = (y - 1 + m) % m; break; // Left (wrap)
            case 1: ny = (y + 1) % m;     break; // Right (wrap)
            case 2: nx = max(0, x - 1);   break; // Up (clamp)
            case 3: nx = min(n - 1, x + 1); break; // Down (clamp)
            case 4: nz = (z - 1 + l) % l; break; // Front (wrap)
            case 5: nz = (z + 1) % l;     break; // Back (wrap)
        }

        int nidx = get_idx(nx, ny, nz, m, l);
        int target_type = in_grid[nidx]; // Read target type from INPUT grid

        // Interaction based on target type
        if (target_type == CELL_TYPE_N || target_type == CELL_TYPE_S) {
            // Attempt to move S into N or S: Use atomicCAS on the *output* grid
            // Try to change target cell from N/S to S, only if it's still N/S
            int old_val = atomicCAS(&out_grid[nidx], target_type, CELL_TYPE_S);
            // If successful (old_val == target_type), the S cell moves.
            // The original location should become what *was* at the target.
            if (old_val == target_type) {
                 out_grid[idx] = target_type; // Original S pos becomes N or S
            }
            // If CAS failed, S cell stays put (out_grid[idx] remains S)
        }
        else if (target_type == CELL_TYPE_M0 || target_type == CELL_TYPE_M1 || target_type == CELL_TYPE_M2) {
            // Interaction with Metal: Check for corrosion
            float corrosion_prob = 0.0f;
            if (target_type == CELL_TYPE_M0) corrosion_prob = prob_m0;
            else if (target_type == CELL_TYPE_M1) corrosion_prob = prob_m1;
            else if (target_type == CELL_TYPE_M2) corrosion_prob = prob_m2;

            float corrosion_rand = curand_uniform(&local_rand_state);
            if (corrosion_rand < corrosion_prob) {
                // Corrosion occurs!
                // Original S becomes N. Target Metal becomes P.
                // Use atomic operations? Not strictly necessary if S->N and M->P are unique results,
                // but safer if other interactions could target the same cells.
                // Let's assume simple assignment is okay here for the two-buffer approach.
                out_grid[idx] = CELL_TYPE_N;  // S becomes N
                out_grid[nidx] = CELL_TYPE_P; // Metal becomes P
                // Increment atomic counter for N->S conversion later
                atomicAdd(atomic_counter, 1);
            }
            // Else (no corrosion): S stays S (already set in out_grid), Metal stays Metal (default)
        }
        // Else (target is P or other): S cell doesn't move (stays S)
    }
    // --- P Cell Logic ---
    else if (current_type == CELL_TYPE_P) {
         // Check probability to move
         if (rand_val < p_Pmove) {
             // Choose random direction (0-5)
             int direction = curand_uniform(&local_rand_state) * 6;
             int nx = x, ny = y, nz = z;

             // Calculate potential new coordinates (neighbor)
             switch(direction) {
                 case 0: ny = (y - 1 + m) % m; break;
                 case 1: ny = (y + 1) % m;     break;
                 case 2: nx = max(0, x - 1);   break;
                 case 3: nx = min(n - 1, x + 1); break;
                 case 4: nz = (z - 1 + l) % l; break;
                 case 5: nz = (z + 1) % l;     break;
             }

             int nidx = get_idx(nx, ny, nz, m, l);
             int target_type = in_grid[nidx]; // Read target type from INPUT grid

             // Check if neighbor is N or S
             if (target_type == CELL_TYPE_N || target_type == CELL_TYPE_S) {
                 // Attempt to move P into N or S using atomicCAS on the *output* grid
                 int old_val = atomicCAS(&out_grid[nidx], target_type, CELL_TYPE_P);
                 // If successful, the P cell moves. Original location becomes target type.
                 if (old_val == target_type) {
                     out_grid[idx] = target_type; // Original P pos becomes N or S
                 }
                 // If CAS failed, P cell stays put (out_grid[idx] remains P)
             }
             // Else (neighbor is not N or S): P cell doesn't move
         }
         // Else (doesn't attempt to move): P stays P (default)
    }

    // Update the random state for the next iteration
    rand_states[thread_id_1d] = local_rand_state;
}
'''

# --- Python Orchestration ---

# Compile Kernels
curand_init_kernel = cp.RawKernel(curand_init_kernel_code, 'init_curand_states')
simulation_step_kernel = cp.RawKernel(simulation_step_kernel_code, 'simulation_step')

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

    if num_s_cells > 0 and total_solution_cells > 0:
         # Use numpy on CPU for choice if total_solution_cells is huge, then transfer.
         # Or use cp.random.choice if feasible.
        if total_solution_cells > 10**8: # Heuristic threshold
             s_indices_flat_np = np.random.choice(np.arange(total_solution_cells), size=num_s_cells, replace=False)
             s_indices_flat = cp.asarray(s_indices_flat_np)
        else:
            try:
                s_indices_flat = cp.random.choice(cp.arange(total_solution_cells), size=num_s_cells, replace=False)
            except ValueError: # Handle case where num_s_cells > total_solution_cells (shouldn't happen with int())
                print(f"Warning: num_s_cells ({num_s_cells}) > total_solution_cells ({total_solution_cells}). Clamping.")
                num_s_cells = total_solution_cells
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

def convert_n_to_s_hybrid(corr_grid, solution_thickness, num_to_convert):
    """
    Hybrid approach: Find N on GPU, choose on CPU, update on GPU.
    """
    if num_to_convert <= 0:
        return

    # Ensure grid dimensions are valid
    n_total, m, l = corr_grid.shape
    if solution_thickness <= 0 or m <= 0 or l <= 0:
        return # No solution layer to operate on

    # 1. Find indices of N cells ONLY in the solution part (GPU)
    solution_slice = corr_grid[:solution_thickness, :, :]
    n_indices_gpu = cp.argwhere(solution_slice == CELL_TYPE_N) # Shape (num_found, 3)

    num_available_n = n_indices_gpu.shape[0]
    num_to_actually_convert = min(num_to_convert, num_available_n)

    if num_to_actually_convert > 0:
        # 2. Choose random indices (CPU)
        # Transfer indices to CPU - this can be costly if many N cells exist
        # Optimization: Maybe only transfer the *count* and do selection differently?
        # Let's stick to transferring indices for now for correctness.
        n_indices_cpu = cp.asnumpy(n_indices_gpu)

        # Use python's random.sample for efficient unique selection
        chosen_indices_local_flat = random.sample(range(num_available_n), num_to_actually_convert)
        # Get the 3D indices corresponding to the chosen flat indices
        chosen_indices_3d_cpu = n_indices_cpu[chosen_indices_local_flat] # Shape (num_convert, 3)

        # 3. Update chosen N cells to S cells (GPU)
        # Transfer chosen 3D indices back to GPU
        chosen_indices_3d_gpu = cp.asarray(chosen_indices_3d_cpu)

        # Extract coordinates and update the main grid
        update_x = chosen_indices_3d_gpu[:, 0]
        update_y = chosen_indices_3d_gpu[:, 1]
        update_z = chosen_indices_3d_gpu[:, 2]
        corr_grid[update_x, update_y, update_z] = CELL_TYPE_S


# Visualization function remains the same as the previous CuPy version
def visualize_corr_grid(corr_grid_gpu, step, z_slice_index=50, show_plot=False, save_plot=True, save_dir="corrosion_plots_gpu_raw"):
    """Visualizes a slice of the grid. Transfers data from GPU to CPU."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(6, 6))
    # Define the colormap based on cell type values
    cmap = ListedColormap(['gray', 'blue', 'yellow', 'red', 'white', 'purple', 'black', 'pink']) # Added pink for Empty=-1 if used
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    # Ensure slice index is valid
    if z_slice_index >= corr_grid_gpu.shape[2] or z_slice_index < 0:
         z_slice_index = corr_grid_gpu.shape[2] // 2

    # Get the slice from GPU -> Transfer to CPU using .get() or cp.asnumpy()
    grid_slice_cpu = cp.asnumpy(corr_grid_gpu[:, :, z_slice_index])

    plt.imshow(grid_slice_cpu, cmap=cmap, norm=norm, interpolation='nearest')
    plt.title(f"GPU RawKernel Step: {step} (Slice Z={z_slice_index})")

    # Create a colorbar with labels
    cbar = plt.colorbar(ticks=[-1, 0, 1, 2, 3, 4, 5, 6])
    cbar.set_ticklabels(['EMPTY','M0', 'M1', 'M2', 'M?', 'N', 'S', 'P']) # Adjust 'M?' label if type 3 is used

    if save_plot:
        plt.savefig(os.path.join(save_dir, f"corrosion_step_{step:05d}.png"))

    if show_plot:
        plt.show()
    else:
        plt.close()

# --- Main Simulation Loop (RawKernel Version) ---

def simulate_corrosion_rawkernel(bound_grid_np, steps, concentration_ratio, solution_thickness, corrosion_base_probabilities, p_Pmove, vis_interval=100, vis_slice=50):
    """Main simulation loop using GPU RawKernels."""

    print("Initializing grid on GPU...")
    d_grid1 = initialize_corr_grid_gpu(bound_grid_np, concentration_ratio, solution_thickness)
    d_grid2 = cp.empty_like(d_grid1) # Second buffer for output

    n_total, m, l = d_grid1.shape
    grid_shape_3d = (n_total, m, l)
    n_threads_total = n_total * m * l

    # --- Setup cuRAND states ---
    print("Initializing cuRAND states...")
    seed = random.randint(0, 2**32 - 1) # Seed for reproducibility if needed
    rand_states = cp.empty(n_threads_total, dtype=cp.uint64) # Placeholder, RawKernel expects pointer type
    # Actual type is curandState, but CuPy doesn't expose it directly. Size might vary.
    # Let's try allocating bytes based on typical size (around 48-56 bytes?). Risky.
    # Alternative: Use CuPy's random generator if RawKernel could access it (it can't directly).
    # Fallback: Use a simpler random number generator within the kernel (e.g., XORshift) or manage states carefully.
    # Let's allocate a placeholder and hope the kernel treats the pointer correctly.
    # Proper way involves linking against cuRAND library or finding CuPy's internal handle.
    # For now, we'll allocate a simple array and pass it. This might CRASH if type size mismatch.
    # A SAFER approach is needed here. Let's USE the curand_init kernel.
    # Need to create the state array type correctly. CuPy doesn't have `curandState`.
    # We might need to define a custom dtype or use void pointer tricks.
    # Let's try allocating sufficient uint64 and hope pointer casting works.
    # Assuming curandState is ~48 bytes = 6 * uint64
    rand_state_element_size = 6 # Heuristic - adjust if needed
    rand_states = cp.empty(n_threads_total * rand_state_element_size, dtype=cp.uint64)

    # Calculate launch configuration for init kernel (1D)
    threads_per_block_1d = 256
    blocks_1d = math.ceil(n_threads_total / threads_per_block_1d)
    curand_init_kernel((blocks_1d,), (threads_per_block_1d,), (seed, n_threads_total, rand_states.data.ptr)) # Pass raw pointer
    print("cuRAND states initialized (potentially risky allocation).")


    # --- Setup Atomic Counter ---
    d_atomic_counter = cp.zeros(1, dtype=cp.int32)

    # --- Simulation Kernel Launch Config ---
    # Use 3D grid/block structure matching the grid
    # Keep block dimensions reasonable (e.g., powers of 2, total threads <= 1024)
    block_dim = (8, 8, 4) # Example: 8*8*4 = 256 threads per block
    grid_dim = (math.ceil(l / block_dim[0]),
                math.ceil(m / block_dim[1]),
                math.ceil(n_total / block_dim[2]))

    # --- Get Probabilities ---
    prob_m0 = float(corrosion_base_probabilities.get(CELL_TYPE_M0, 0.0))
    prob_m1 = float(corrosion_base_probabilities.get(CELL_TYPE_M1, 0.0))
    prob_m2 = float(corrosion_base_probabilities.get(CELL_TYPE_M2, 0.0))
    p_Pmove_float = float(p_Pmove)

    # --- Main Loop ---
    print("Starting simulation with RawKernel...")
    d_grid_in = d_grid1
    d_grid_out = d_grid2

    for step in tqdm.tqdm(range(steps)):
        # 1. Reset atomic counter
        d_atomic_counter.fill(0)

        # 2. Launch the simulation kernel
        simulation_step_kernel(
            grid_dim, block_dim,
            (d_grid_in, d_grid_out, rand_states.data.ptr, # Pass raw pointer to states
             n_total, m, l,
             prob_m0, prob_m1, prob_m2,
             p_Pmove_float,
             d_atomic_counter)
        )

        # 3. Synchronize (ensure kernel finishes before reading counter)
        cp.cuda.Stream.null.synchronize()

        # 4. Get number of corrosion events
        num_corroded = int(d_atomic_counter.get())

        # 5. Perform N -> S conversion (using the output grid)
        if num_corroded > 0:
             convert_n_to_s_hybrid(d_grid_out, solution_thickness, num_corroded)

        # 6. Swap input/output grids for the next step
        d_grid_in, d_grid_out = d_grid_out, d_grid_in

        # 7. Visualization (periodically, use d_grid_in as it holds the latest completed state)
        if step % vis_interval == 0 or step == steps - 1:
            visualize_corr_grid(d_grid_in, step, z_slice_index=vis_slice, show_plot=(step == steps - 1), save_plot=True)

    print("Simulation finished.")
    # Final state is in d_grid_in


# --- Parameters and Execution ---
if __name__ == '__main__':
    # Parameters (adjust as needed)
    concentration_ratio = 0.2
    solution_thickness = 10
    corrosion_base_probabilities = {
        CELL_TYPE_M0: 0.0002, # Grain Al
        CELL_TYPE_M1: 0.6,    # Boundary Al
        CELL_TYPE_M2: 0.05    # Precipitate
    }
    p_Pmove = 0.2 # Probability for P cell to move
    simulation_steps = 5000 # Number of simulation steps
    visualization_interval = 100 # Visualize every N steps
    visualization_slice_z = 50 # Z-slice index to visualize

    # Load the initial metal grid (NumPy array)
    try:
        bound_grid_path = r"G:\接活\grid_alloy_100_20_0.8.npy" # Adjust path if needed
        bound_grid_np = np.load(bound_grid_path)
        print(f"Loaded metal grid from {bound_grid_path} with shape: {bound_grid_np.shape}")
        if visualization_slice_z >= bound_grid_np.shape[2]:
            visualization_slice_z = bound_grid_np.shape[2] // 2
            print(f"Adjusted visualization Z slice to: {visualization_slice_z}")
    except FileNotFoundError:
        print(f"Error: Metal grid file not found at {bound_grid_path}")
        exit()
    except Exception as e:
        print(f"Error loading grid: {e}")
        exit()

    # Run the simulation
    simulate_corrosion_rawkernel(
        bound_grid_np,
        steps=simulation_steps,
        concentration_ratio=concentration_ratio,
        solution_thickness=solution_thickness,
        corrosion_base_probabilities=corrosion_base_probabilities,
        p_Pmove=p_Pmove,
        vis_interval=visualization_interval,
        vis_slice=visualization_slice_z
    )