#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <string>
#include <atomic>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <omp.h> // For OpenMP
#include <filesystem> // For directory operations (C++17)
#include <iomanip> // For formatting output

// --- Cell Type Definitions ---
constexpr int CELL_GRAIN = 0;    // M: Grain interior
constexpr int CELL_GB = 1;       // M: Grain boundary
constexpr int CELL_PRECIP = 2;   // M: Precipitate
constexpr int CELL_GB_P = 3;     // Type 3 (Not M, inert based on rules)
constexpr int CELL_N_SOL = 4;    // N: Neutral solution
constexpr int CELL_S_SOL = 5;    // S: Corrosive solution
constexpr int CELL_PRODUCT = 6;  // P: Corrosion product
constexpr int CELL_S_SWAPPING = -5; // Temporary state for S<->S atomic swap

// Helper to get 1D index from 3D coordinates
inline size_t idx(int x, int y, int z, int m, int l) {
    return static_cast<size_t>(z) + static_cast<size_t>(l) * (static_cast<size_t>(y) + static_cast<size_t>(m) * static_cast<size_t>(x));
}

// Helper to check if a cell type is swappable solution (for product movement)
inline bool is_swappable_solution(int cell_type) {
    return (cell_type == CELL_N_SOL || cell_type == CELL_S_SOL);
}

// --- Simulation Functions ---

// Placeholder: Creates a simple metal grid instead of loading from .npy
std::vector<int> create_placeholder_metal_grid(int n_metal, int m, int l) {
    std::vector<int> metal_grid(static_cast<size_t>(n_metal) * m * l);
    // Example: Fill mostly with grain, add some boundaries/precipitates
    for (int x = 0; x < n_metal; ++x) {
        for (int y = 0; y < m; ++y) {
            for (int z = 0; z < l; ++z) {
                size_t index = idx(x, y, z, m, l);
                if (x < 2 || y < 2 || z < 2 || x > n_metal - 3 || y > m - 3 || z > l - 3) {
                     if ((x+y+z) % 10 == 0) metal_grid[index] = CELL_PRECIP; // Some precipitates near edges
                     else metal_grid[index] = CELL_GB; // Grain boundary near edges
                } else {
                     metal_grid[index] = CELL_GRAIN; // Mostly grain inside
                }
            }
        }
    }
     // Add a distinct feature for visualization
     if (n_metal > 5 && m > 10 && l > 10) {
         for(int y = m/2 - 5; y < m/2 + 5; ++y) {
            for(int z = l/2 - 5; z < l/2 + 5; ++z) {
                metal_grid[idx(n_metal/2, y, z, m, l)] = CELL_PRECIP;
            }
         }
     }
    std::cout << "Created placeholder metal grid (" << n_metal << "x" << m << "x" << l << ")" << std::endl;
    return metal_grid;
}


// Rule 1: Initialize corrosion grid
std::vector<int> initialize_corr_grid(const std::vector<int>& bound_grid_metal,
                                      double concentration_ratio,
                                      int solution_thickness,
                                      int n_metal, int m, int l) {
    int n_total = n_metal + solution_thickness;
    size_t total_size = static_cast<size_t>(n_total) * m * l;
    std::cout << "Creating grid of shape (" << n_total << ", " << m << ", " << l << ")" << std::endl;

    std::vector<int> corr_grid(total_size, CELL_N_SOL); // Initialize all as Neutral Solution

    // Populate solution layer
    size_t solution_volume_size = static_cast<size_t>(solution_thickness) * m * l;
    size_t num_S_cells_target = static_cast<size_t>(solution_volume_size * concentration_ratio);

    std::cout << "Targeting " << num_S_cells_target << " S-cells in solution layer (size " << solution_volume_size << ")." << std::endl;

    if (num_S_cells_target > 0 && solution_volume_size > 0) {
        std::vector<size_t> solution_indices(solution_volume_size);
        std::iota(solution_indices.begin(), solution_indices.end(), 0); // Fill with 0, 1, ..., N-1

        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(solution_indices.begin(), solution_indices.end(), g);

        size_t num_S_cells_actual = std::min(num_S_cells_target, solution_volume_size);
        for (size_t i = 0; i < num_S_cells_actual; ++i) {
            corr_grid[solution_indices[i]] = CELL_S_SOL;
        }
        std::cout << "Placed " << num_S_cells_actual << " S-cells." << std::endl;
    }

    // Place metal grid
    std::cout << "Placing metal grid..." << std::endl;
    size_t metal_start_offset = static_cast<size_t>(solution_thickness) * m * l;
    if (bound_grid_metal.size() == static_cast<size_t>(n_metal) * m * l) {
         std::copy(bound_grid_metal.begin(), bound_grid_metal.end(), corr_grid.begin() + metal_start_offset);
    } else {
         std::cerr << "Warning: Metal grid size mismatch! Metal part might be incorrect." << std::endl;
    }


    std::cout << "Grid initialization complete." << std::endl;
    return corr_grid;
}

// Rule 3 (Concentration Maintenance)
void convert_n_to_s(std::vector<int>& corr_grid,
                    int count_s_converted_to_n, // How many S became N due to corrosion
                    int solution_thickness, int m, int l,
                    int absolute_cap, std::vector<std::mt19937>& rngs) {

    if (count_s_converted_to_n <= 0 || absolute_cap <= 0) return;

    std::vector<size_t> N_positions_indices;
    size_t solution_volume_size = static_cast<size_t>(solution_thickness) * m * l;

    // Find available N cells IN THE SOLUTION LAYER ONLY
    // This part is sequential but could be parallelized if it becomes a bottleneck
    for (size_t i = 0; i < solution_volume_size; ++i) {
        if (corr_grid[i] == CELL_N_SOL) {
            N_positions_indices.push_back(i);
        }
    }

    size_t num_available_N = N_positions_indices.size();
    if (num_available_N > 0) {
        size_t num_to_convert = std::min({static_cast<size_t>(num_available_N),
                                          static_cast<size_t>(absolute_cap),
                                          static_cast<size_t>(count_s_converted_to_n)}); // Replenish up to the number lost or the cap

        if (num_to_convert > 0) {
            // Get a random number generator (use the first one for this sequential part)
             std::mt19937& g = rngs[0]; // Or distribute if parallelizing this selection
             std::shuffle(N_positions_indices.begin(), N_positions_indices.end(), g);

            for (size_t i = 0; i < num_to_convert; ++i) {
                 // Use atomic reference for the update, although less contention expected here
                 std::atomic_ref<int> cell_ref(corr_grid[N_positions_indices[i]]);
                 cell_ref.store(CELL_S_SOL); // Directly store, less need for CAS here
            }
           // std::cout << "Converted " << num_to_convert << " N cells to S." << std::endl; // Optional debug
        }
    }
}

// Rule 7: Save data slice (replaces matplotlib visualization)
bool save_slice_to_file(const std::vector<int>& grid, int step, int z_slice,
                       int n_total, int m, int l, const std::string& frame_dir) {
    if (!std::filesystem::exists(frame_dir)) {
        try {
            std::filesystem::create_directories(frame_dir);
        } catch (const std::exception& e) {
            std::cerr << "Error creating directory '" << frame_dir << "': " << e.what() << std::endl;
            return false;
        }
    }

    std::string filename = frame_dir + "/frame_" + std::to_string(step) + "_slice_" + std::to_string(z_slice) + ".txt";
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return false;
    }

     if (z_slice < 0 || z_slice >= l) {
         std::cerr << "Error: Invalid z_slice index " << z_slice << std::endl;
         return false;
     }


    // Write the 2D slice (x rows, y columns)
    for (int x = 0; x < n_total; ++x) {
        for (int y = 0; y < m; ++y) {
            outfile << grid[idx(x, y, z_slice, m, l)] << (y == m - 1 ? "" : " ");
        }
        outfile << "\n";
    }

    outfile.close();
    return true;
}


// --- Main simulation step functions (Parallelized) ---

// Rule 2a: S cell random walk and corrosion
void random_walk_s(std::vector<int>& grid, int n_total, int m, int l,
                   float p0, float p1, float p2,
                   std::atomic<int>& s_to_n_counter,
                   std::atomic<int>& prob_check_passed_counter,
                   std::vector<std::mt19937>& rngs, // Per-thread RNGs
                   std::vector<std::uniform_real_distribution<float>>& dists // Per-thread distributions
                   ) {
    #pragma omp parallel for collapse(3) // Parallelize over the 3D grid
    for (int x = 0; x < n_total; ++x) {
        for (int y = 0; y < m; ++y) {
            for (int z = 0; z < l; ++z) {
                size_t current_idx = idx(x, y, z, m, l);
                std::atomic_ref<int> current_cell_ref(grid[current_idx]); // Atomic reference to current cell
                int current_cell_type = current_cell_ref.load(); // Read current type

                if (current_cell_type == CELL_S_SOL) {
                    int thread_id = omp_get_thread_num();
                    std::mt19937& engine = rngs[thread_id]; // Get thread-local RNG engine
                    std::uniform_real_distribution<float>& dist = dists[thread_id]; // Thread-local distribution
                    std::uniform_int_distribution<int> dir_dist(0, 5); // Distribution for direction


                    int direction = dir_dist(engine); // Choose random direction
                    int nx = x, ny = y, nz = z;

                    // Calculate neighbor coordinates (Von Neumann neighborhood)
                    // Periodic boundaries for y and z, clamping for x
                    if (direction == 0) { ny = (y - 1 + m) % m; }      // Back
                    else if (direction == 1) { ny = (y + 1) % m; }      // Front
                    else if (direction == 2) { nx = std::max(0, x - 1); } // Down (towards solution)
                    else if (direction == 3) { nx = std::min(n_total - 1, x + 1); } // Up (towards metal)
                    else if (direction == 4) { nz = (z - 1 + l) % l; }      // Left
                    else { nz = (z + 1) % l; }                      // Right

                    // Check if neighbor is within valid *logical* bounds (x is clamped already)
                     // No need for nx bounds check due to clamping above
                    size_t neighbor_idx = idx(nx, ny, nz, m, l);
                    std::atomic_ref<int> neighbor_cell_ref(grid[neighbor_idx]); // Atomic reference to neighbor

                    int target_type = neighbor_cell_ref.load(); // Read neighbor type

                    // --- Apply Rules ---

                    if (target_type == CELL_N_SOL) { // S<->N Swap Attempt
                        int expected = CELL_N_SOL;
                        // Try to change N at neighbor_idx to S
                        if (neighbor_cell_ref.compare_exchange_strong(expected, CELL_S_SOL)) {
                            // If successful, change current cell S to N
                            current_cell_ref.store(CELL_N_SOL);
                        }
                        // If CAS fails, another thread likely changed the neighbor, so do nothing

                    } else if (target_type >= CELL_GRAIN && target_type <= CELL_PRECIP) { // S+M -> P+N Corrosion Attempt
                        float corrosion_prob = 0.0f;
                        if (target_type == CELL_GRAIN) corrosion_prob = p0;
                        else if (target_type == CELL_GB) corrosion_prob = p1;
                        else if (target_type == CELL_PRECIP) corrosion_prob = p2;

                        float rand_val_for_corrosion = dist(engine); // Generate random number JUST before check

                        if (rand_val_for_corrosion < corrosion_prob) { // Corrosion occurs
                             int expected_metal = target_type; // Expect the specific metal type
                             // Try to change M at neighbor_idx to P
                             if (neighbor_cell_ref.compare_exchange_strong(expected_metal, CELL_PRODUCT)) {
                                 // If successful:
                                 prob_check_passed_counter.fetch_add(1); // Increment diagnostic counter
                                 current_cell_ref.store(CELL_N_SOL);    // Change current cell S to N
                                 s_to_n_counter.fetch_add(1);          // Increment counter for replenishment
                             }
                             // If CAS fails, target was already changed (e.g., to P by another thread), do nothing
                        }
                        // Else (no corrosion): S + M -> S + M (no change)

                    } else if (target_type == CELL_S_SOL) { // S<->S Swap Attempt (More complex)
                         int expected_s = CELL_S_SOL;
                         // Try to change target S to temporary SWAPPING state
                         if (neighbor_cell_ref.compare_exchange_strong(expected_s, CELL_S_SWAPPING)) {
                             // If successful, we 'own' the target cell temporarily.
                             // Change current cell to S (it remains S, but confirms the swap start)
                             current_cell_ref.store(CELL_S_SOL); // No real change needed here? Or set to N if source moves? Check Python logic.
                             // Python logic: grid[idx] = CELL_S_SOL (redundant), atomicExch(&grid[nidx], CELL_S_SOL) -> Changes target back
                             // Let's assume the source S moves *into* the target position logically
                             // So, the source cell *should* become what the target *was* if it were empty. But it's S.
                             // The goal is just to swap positions. The CAS ensures only one thread swaps with this target.
                             // Final step: change the target back from SWAPPING to S.
                             neighbor_cell_ref.store(CELL_S_SOL); // Could use exchange, but store is fine here as we 'own' it.
                             // The *source* cell (current_idx) should logically be empty now, but since we are iterating
                             // through S cells, another S might move into it later, or maybe it should become N?
                             // The Python code seems to imply the original S at current_idx just stays put if the swap partner is S.
                             // Let's re-examine Python:
                             //  int old_target = atomicCAS(&grid[nidx], CELL_S_SOL, CELL_S_SWAPPING);
                             //  if (old_target == CELL_S_SOL) {
                             //      grid[idx] = CELL_S_SOL; // Source remains S
                             //      atomicExch(&grid[nidx], CELL_S_SOL); // Target becomes S again
                             //  }
                             // This CUDA code doesn't actually *swap* the cells' identities, it just ensures
                             // that if two S cells try to move into the same spot, only one interaction happens,
                             // effectively acting like a bump/no movement scenario handled atomically.
                             // So the C++ code above correctly reflects this: if CAS succeeds, we just revert target to S.

                         }
                         // If CAS fails, another S is interacting with the target, so this thread does nothing.
                    }
                    // Implicit: S + P or S + Type3 -> S + P / S + Type3 (no reaction, no movement)
                } // end if (current_cell_type == CELL_S_SOL)
            } // end z loop
        } // end y loop
    } // end x loop
}


// Rule 2b: P cell random walk
void move_products(std::vector<int>& grid, int n_total, int m, int l, float p_Pmove,
                   std::vector<std::mt19937>& rngs, // Per-thread RNGs
                   std::vector<std::uniform_real_distribution<float>>& dists // Per-thread distributions
                  ) {

    #pragma omp parallel for collapse(3) // Parallelize over the 3D grid
    for (int x = 0; x < n_total; ++x) {
        for (int y = 0; y < m; ++y) {
            for (int z = 0; z < l; ++z) {
                 size_t current_idx = idx(x, y, z, m, l);
                 // Non-atomic read is okay here, as we only care if *this* thread sees a P.
                 // If it changes before we act, our atomic operations later will handle it.
                 if (grid[current_idx] == CELL_PRODUCT) {
                     int thread_id = omp_get_thread_num();
                     std::mt19937& engine = rngs[thread_id];
                     std::uniform_real_distribution<float>& dist = dists[thread_id];

                     if (dist(engine) < p_Pmove) { // Check move probability
                         // Find swappable neighbors (N or S)
                         int dx[] = {0, 0, -1, 1, 0, 0};
                         int dy[] = {-1, 1, 0, 0, 0, 0};
                         int dz[] = {0, 0, 0, 0, -1, 1};
                         std::vector<size_t> swappable_neighbor_indices;
                         swappable_neighbor_indices.reserve(6); // Reserve space

                         for (int i = 0; i < 6; ++i) {
                             int nx = x + dx[i];
                             int ny = y + dy[i];
                             int nz = z + dz[i];

                             // Check bounds (x clamped, y/z periodic)
                             if (nx >= 0 && nx < n_total) {
                                 ny = (ny % m + m) % m; // Ensure positive modulo result
                                 nz = (nz % l + l) % l; // Ensure positive modulo result
                                 size_t neighbor_idx = idx(nx, ny, nz, m, l);
                                 // Read neighbor type non-atomically (will re-check with atomic)
                                 if (is_swappable_solution(grid[neighbor_idx])) {
                                     swappable_neighbor_indices.push_back(neighbor_idx);
                                 }
                             }
                         }

                         if (!swappable_neighbor_indices.empty()) {
                             // Choose a random swappable neighbor
                             std::uniform_int_distribution<size_t> neighbor_dist(0, swappable_neighbor_indices.size() - 1);
                             size_t target_idx = swappable_neighbor_indices[neighbor_dist(engine)];

                             // --- Atomic Swap Attempt ---
                             std::atomic_ref<int> current_cell_ref(grid[current_idx]);
                             std::atomic_ref<int> target_cell_ref(grid[target_idx]);

                             // Try to take the solution cell, putting P there
                             int old_target_val = target_cell_ref.exchange(CELL_PRODUCT);

                             // Check if we actually got a solution cell (N or S)
                             if (is_swappable_solution(old_target_val)) {
                                 // Success! Put the original solution type (N or S) into the current cell
                                 current_cell_ref.store(old_target_val);
                             } else {
                                 // Failure: We grabbed something else (maybe another P moved first).
                                 // Put the original value back into the target cell.
                                 target_cell_ref.store(old_target_val);
                                 // The current cell remains P.
                             }
                         } // end if swappable neighbors found
                     } // end if move probability check passes
                 } // end if cell is product
            } // end z
        } // end y
    } // end x
}

// --- Main Simulation Loop ---
std::vector<std::string> simulate_corrosion(
    std::vector<int>& corr_grid, // Grid passed by reference
    int n_total, int m, int l, int solution_thickness,
    float p0, float p1, float p2, float p_Pmove,
    int steps,
    int absolute_replenish_cap,
    int frame_interval,
    const std::string& frame_dir,
    int z_slice_anim)
{
    std::vector<std::string> frame_files; // To potentially store filenames if needed later

    // --- Initialize Random Number Generators (one per thread) ---
    int max_threads = omp_get_max_threads();
    std::vector<std::mt19937> rngs(max_threads);
    std::vector<std::uniform_real_distribution<float>> dists(max_threads);
    std::random_device rd;
    for (int i = 0; i < max_threads; ++i) {
        rngs[i].seed(rd() + i); // Seed each engine differently
        dists[i] = std::uniform_real_distribution<float>(0.0f, 1.0f);
    }
    std::cout << "Initialized " << max_threads << " RNGs for parallel execution." << std::endl;


    // --- Atomic Counters ---
    std::atomic<int> s_to_n_counter(0);
    std::atomic<int> prob_check_passed_counter(0);

    // --- Setup Frame Directory ---
     if (std::filesystem::exists(frame_dir)) {
         try {
             std::filesystem::remove_all(frame_dir);
         } catch (const std::exception& e) {
              std::cerr << "Warning: Could not remove existing frame directory: " << e.what() << std::endl;
         }
     }
     try {
          std::filesystem::create_directories(frame_dir);
          std::cout << "Saving animation frame data every " << frame_interval << " steps to '" << frame_dir << "/'" << std::endl;
     } catch (const std::exception& e) {
          std::cerr << "Error creating frame directory '" << frame_dir << "': " << e.what() << std::endl;
          return frame_files; // Return empty list if directory fails
     }


    std::cout << "Using ABSOLUTE N->S Replenish Cap = " << absolute_replenish_cap << " per step (activated by corrosion)" << std::endl;

    // --- Simulation Loop ---
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < steps; ++step) {
        // Reset counters for the step
        s_to_n_counter.store(0);
        prob_check_passed_counter.store(0);

        // 1. S cell walk/corrosion kernel call
        random_walk_s(corr_grid, n_total, m, l, p0, p1, p2,
                      s_to_n_counter, prob_check_passed_counter, rngs, dists);

        // 2. P cell move kernel call
        move_products(corr_grid, n_total, m, l, p_Pmove, rngs, dists);

        // --- Synchronization Point (Implicit in OpenMP barrier at end of parallel regions) ---
        // No explicit sync needed like cp.cuda.Device().synchronize() unless using async tasks

        // 3. Concentration Maintenance (Sequential part, uses RNG[0])
        int s_converted_count = s_to_n_counter.load();
        if (s_converted_count > 0) {
             convert_n_to_s(corr_grid, s_converted_count, solution_thickness, m, l, absolute_replenish_cap, rngs);
        }


        // 4. Visualization & Cell Counts
        if (step % frame_interval == 0 || step == steps - 1) {
            long long s_count = 0, n_count = 0, p_count = 0, m0_count = 0, m1_count = 0, m2_count = 0;
            // Simple sequential count for verification/logging
            for(int cell_val : corr_grid) {
                switch(cell_val) {
                    case CELL_S_SOL: s_count++; break;
                    case CELL_N_SOL: n_count++; break;
                    case CELL_PRODUCT: p_count++; break;
                    case CELL_GRAIN: m0_count++; break;
                    case CELL_GB: m1_count++; break;
                    case CELL_PRECIP: m2_count++; break;
                }
            }
            int prob_checks_ok = prob_check_passed_counter.load();

             auto current_time = std::chrono::high_resolution_clock::now();
             std::chrono::duration<double> elapsed = current_time - start_time;

            std::cout << "\nStep " << step << " [" << std::fixed << std::setprecision(2) << elapsed.count() << "s]: "
                      << "S=" << s_count << ", N=" << n_count << ", P=" << p_count
                      << ", M0=" << m0_count << ", M1=" << m1_count << ", M2=" << m2_count
                      << ", ProbChecksOK=" << prob_checks_ok;


            bool saved = save_slice_to_file(corr_grid, step, z_slice_anim, n_total, m, l, frame_dir);
            if (!saved) {
                std::cerr << "Warning: Failed to save frame data at step " << step << std::endl;
            } else {
                 // Optionally store filename if needed later for GIF script
                 // frame_files.push_back(frame_dir + "/...");
            }
        } else {
             // Progress indication for steps without full output
             if (step % 10 == 0) { // Print a dot every 10 steps
                 std::cout << "." << std::flush;
             }
        }


    } // End simulation loop

     auto end_time = std::chrono::high_resolution_clock::now();
     std::chrono::duration<double> total_elapsed = end_time - start_time;
     std::cout << "\n\nSimulation finished " << steps << " steps in "
               << std::fixed << std::setprecision(2) << total_elapsed.count() << " seconds." << std::endl;


    return frame_files; // Return list (currently empty, but could hold filenames)
}


// --- Main Execution Logic ---
int main() {
    // --- Parameters ---
    // Metal grid dimensions (using placeholder generator)
    int grid_size_m = 100; // Example dimension for metal part y/z
    int grid_size_n = 50;  // Example dimension for metal part x (depth)
    // Parameters for placeholder grid generation (if used)
    // int num_grains = 20; // Not directly used in simple placeholder
    // float precipitate_ratio = 0.8; // Not directly used in simple placeholder

    // Simulation Parameters
    double concentration_ratio = 0.2;
    int solution_thickness = 10;
    // Corrosion Probabilities (matched to Python example)
    float p0_grain = 0.1f;
    float p1_gb = 0.8f;
    float p2_precip = 0.4f;
    float p_Pmove = 0.9f; // Probability for Product movement
    int simulation_steps = 1500;
    int ABSOLUTE_REPLENISH_CAP = 50; // Max N->S conversion per step

    // Animation/Output Parameters
    int save_frame_interval = 100;
    std::string animation_dir = "corrosion_frames_cpp_cap_" + std::to_string(ABSOLUTE_REPLENISH_CAP);
    // Output GIF generation would need an external script reading the .txt frames
    int animation_z_slice; // Will be set after getting metal grid dimensions

    // --- Setup ---
    // Load or Generate Metal Grid
    // Option 1: Generate placeholder
     std::vector<int> bound_grid_metal = create_placeholder_metal_grid(grid_size_n, grid_size_m, grid_size_m);
     int n_metal = grid_size_n;
     int m = grid_size_m;
     int l = grid_size_m;
     animation_z_slice = l / 2; // Set slice based on actual dimension

    // Option 2: Load from file (requires a library like cnpy)
    // std::string metal_filepath = "grid_alloy_100_20_0.8.npy"; // Example filename
    // try {
    //     // cnpy::NpyArray metal_arr = cnpy::npy_load(metal_filepath);
    //     // n_metal = metal_arr.shape[0]; m = metal_arr.shape[1]; l = metal_arr.shape[2];
    //     // bound_grid_metal = metal_arr.as_vec<int>(); // Or correct type
    //     // animation_z_slice = l / 2;
    //     // std::cout << "Loaded metal grid from " << metal_filepath << ", shape: " << n_metal << "x" << m << "x" << l << std::endl;
    // } catch (const std::exception& e) {
    //     std::cerr << "Error loading metal grid file '" << metal_filepath << "': " << e.what() << std::endl;
    //     return 1;
    // }

    int n_total = n_metal + solution_thickness;

    // --- Print Setup Info ---
    std::cout << std::string(30, '-') << std::endl;
    std::cout << "Simulation Setup (C++ / OpenMP):" << std::endl;
    std::cout << "  Metal Grid Dimensions:  " << n_metal << "x" << m << "x" << l << std::endl;
    std::cout << "  Solution Thickness:     " << solution_thickness << std::endl;
    std::cout << "  Total Grid Dimensions:  " << n_total << "x" << m << "x" << l << std::endl;
    std::cout << "  Initial S ratio:        " << concentration_ratio << std::endl;
    std::cout << "  Corrosion Probs:        M0=" << p0_grain << ", M1=" << p1_gb << ", M2=" << p2_precip << std::endl;
    std::cout << "  Product Move Prob:      " << p_Pmove << std::endl;
    std::cout << "  Simulation Steps:       " << simulation_steps << std::endl;
    std::cout << "  Replenish Cap/Step:     " << ABSOLUTE_REPLENISH_CAP << std::endl;
    std::cout << "  Output Slice z:         " << animation_z_slice << std::endl;
    std::cout << "  Output Frame Dir:       " << animation_dir << std::endl;
    std::cout << std::string(30, '-') << std::endl;

    // --- Execute Simulation ---
    try {
        std::cout << "Initializing corrosion grid..." << std::endl;
        std::vector<int> corr_grid = initialize_corr_grid(bound_grid_metal, concentration_ratio, solution_thickness, n_metal, m, l);
        // Metal grid no longer needed after copy
        bound_grid_metal.clear();
        bound_grid_metal.shrink_to_fit();


        std::cout << "\nStarting C++/OpenMP simulation for " << simulation_steps << " steps..." << std::endl;
        std::vector<std::string> frame_files = simulate_corrosion(
            corr_grid, n_total, m, l, solution_thickness,
            p0_grain, p1_gb, p2_precip, p_Pmove,
            simulation_steps,
            ABSOLUTE_REPLENISH_CAP,
            save_frame_interval,
            animation_dir,
            animation_z_slice);

        std::cout << "\nSimulation completed." << std::endl;
        std::cout << "Frame data saved in: " << animation_dir << std::endl;
        std::cout << "Use an external tool (Python/Matplotlib, Gnuplot, etc.) to visualize the .txt slices." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\nAn unexpected error occurred: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\nAn unknown error occurred." << std::endl;
        return 1;
    }


    std::cout << "\nProgram finished." << std::endl;
    return 0;
}#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <string>
#include <atomic>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <omp.h> // For OpenMP
#include <filesystem> // For directory operations (C++17)
#include <iomanip> // For formatting output

// --- Cell Type Definitions ---
constexpr int CELL_GRAIN = 0;    // M: Grain interior
constexpr int CELL_GB = 1;       // M: Grain boundary
constexpr int CELL_PRECIP = 2;   // M: Precipitate
constexpr int CELL_GB_P = 3;     // Type 3 (Not M, inert based on rules)
constexpr int CELL_N_SOL = 4;    // N: Neutral solution
constexpr int CELL_S_SOL = 5;    // S: Corrosive solution
constexpr int CELL_PRODUCT = 6;  // P: Corrosion product
constexpr int CELL_S_SWAPPING = -5; // Temporary state for S<->S atomic swap

// Helper to get 1D index from 3D coordinates
inline size_t idx(int x, int y, int z, int m, int l) {
    return static_cast<size_t>(z) + static_cast<size_t>(l) * (static_cast<size_t>(y) + static_cast<size_t>(m) * static_cast<size_t>(x));
}

// Helper to check if a cell type is swappable solution (for product movement)
inline bool is_swappable_solution(int cell_type) {
    return (cell_type == CELL_N_SOL || cell_type == CELL_S_SOL);
}

// --- Simulation Functions ---

// Placeholder: Creates a simple metal grid instead of loading from .npy
std::vector<int> create_placeholder_metal_grid(int n_metal, int m, int l) {
    std::vector<int> metal_grid(static_cast<size_t>(n_metal) * m * l);
    // Example: Fill mostly with grain, add some boundaries/precipitates
    for (int x = 0; x < n_metal; ++x) {
        for (int y = 0; y < m; ++y) {
            for (int z = 0; z < l; ++z) {
                size_t index = idx(x, y, z, m, l);
                if (x < 2 || y < 2 || z < 2 || x > n_metal - 3 || y > m - 3 || z > l - 3) {
                     if ((x+y+z) % 10 == 0) metal_grid[index] = CELL_PRECIP; // Some precipitates near edges
                     else metal_grid[index] = CELL_GB; // Grain boundary near edges
                } else {
                     metal_grid[index] = CELL_GRAIN; // Mostly grain inside
                }
            }
        }
    }
     // Add a distinct feature for visualization
     if (n_metal > 5 && m > 10 && l > 10) {
         for(int y = m/2 - 5; y < m/2 + 5; ++y) {
            for(int z = l/2 - 5; z < l/2 + 5; ++z) {
                metal_grid[idx(n_metal/2, y, z, m, l)] = CELL_PRECIP;
            }
         }
     }
    std::cout << "Created placeholder metal grid (" << n_metal << "x" << m << "x" << l << ")" << std::endl;
    return metal_grid;
}


// Rule 1: Initialize corrosion grid
std::vector<int> initialize_corr_grid(const std::vector<int>& bound_grid_metal,
                                      double concentration_ratio,
                                      int solution_thickness,
                                      int n_metal, int m, int l) {
    int n_total = n_metal + solution_thickness;
    size_t total_size = static_cast<size_t>(n_total) * m * l;
    std::cout << "Creating grid of shape (" << n_total << ", " << m << ", " << l << ")" << std::endl;

    std::vector<int> corr_grid(total_size, CELL_N_SOL); // Initialize all as Neutral Solution

    // Populate solution layer
    size_t solution_volume_size = static_cast<size_t>(solution_thickness) * m * l;
    size_t num_S_cells_target = static_cast<size_t>(solution_volume_size * concentration_ratio);

    std::cout << "Targeting " << num_S_cells_target << " S-cells in solution layer (size " << solution_volume_size << ")." << std::endl;

    if (num_S_cells_target > 0 && solution_volume_size > 0) {
        std::vector<size_t> solution_indices(solution_volume_size);
        std::iota(solution_indices.begin(), solution_indices.end(), 0); // Fill with 0, 1, ..., N-1

        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(solution_indices.begin(), solution_indices.end(), g);

        size_t num_S_cells_actual = std::min(num_S_cells_target, solution_volume_size);
        for (size_t i = 0; i < num_S_cells_actual; ++i) {
            corr_grid[solution_indices[i]] = CELL_S_SOL;
        }
        std::cout << "Placed " << num_S_cells_actual << " S-cells." << std::endl;
    }

    // Place metal grid
    std::cout << "Placing metal grid..." << std::endl;
    size_t metal_start_offset = static_cast<size_t>(solution_thickness) * m * l;
    if (bound_grid_metal.size() == static_cast<size_t>(n_metal) * m * l) {
         std::copy(bound_grid_metal.begin(), bound_grid_metal.end(), corr_grid.begin() + metal_start_offset);
    } else {
         std::cerr << "Warning: Metal grid size mismatch! Metal part might be incorrect." << std::endl;
    }


    std::cout << "Grid initialization complete." << std::endl;
    return corr_grid;
}

// Rule 3 (Concentration Maintenance)
void convert_n_to_s(std::vector<int>& corr_grid,
                    int count_s_converted_to_n, // How many S became N due to corrosion
                    int solution_thickness, int m, int l,
                    int absolute_cap, std::vector<std::mt19937>& rngs) {

    if (count_s_converted_to_n <= 0 || absolute_cap <= 0) return;

    std::vector<size_t> N_positions_indices;
    size_t solution_volume_size = static_cast<size_t>(solution_thickness) * m * l;

    // Find available N cells IN THE SOLUTION LAYER ONLY
    // This part is sequential but could be parallelized if it becomes a bottleneck
    for (size_t i = 0; i < solution_volume_size; ++i) {
        if (corr_grid[i] == CELL_N_SOL) {
            N_positions_indices.push_back(i);
        }
    }

    size_t num_available_N = N_positions_indices.size();
    if (num_available_N > 0) {
        size_t num_to_convert = std::min({static_cast<size_t>(num_available_N),
                                          static_cast<size_t>(absolute_cap),
                                          static_cast<size_t>(count_s_converted_to_n)}); // Replenish up to the number lost or the cap

        if (num_to_convert > 0) {
            // Get a random number generator (use the first one for this sequential part)
             std::mt19937& g = rngs[0]; // Or distribute if parallelizing this selection
             std::shuffle(N_positions_indices.begin(), N_positions_indices.end(), g);

            for (size_t i = 0; i < num_to_convert; ++i) {
                 // Use atomic reference for the update, although less contention expected here
                 std::atomic_ref<int> cell_ref(corr_grid[N_positions_indices[i]]);
                 cell_ref.store(CELL_S_SOL); // Directly store, less need for CAS here
            }
           // std::cout << "Converted " << num_to_convert << " N cells to S." << std::endl; // Optional debug
        }
    }
}

// Rule 7: Save data slice (replaces matplotlib visualization)
bool save_slice_to_file(const std::vector<int>& grid, int step, int z_slice,
                       int n_total, int m, int l, const std::string& frame_dir) {
    if (!std::filesystem::exists(frame_dir)) {
        try {
            std::filesystem::create_directories(frame_dir);
        } catch (const std::exception& e) {
            std::cerr << "Error creating directory '" << frame_dir << "': " << e.what() << std::endl;
            return false;
        }
    }

    std::string filename = frame_dir + "/frame_" + std::to_string(step) + "_slice_" + std::to_string(z_slice) + ".txt";
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return false;
    }

     if (z_slice < 0 || z_slice >= l) {
         std::cerr << "Error: Invalid z_slice index " << z_slice << std::endl;
         return false;
     }


    // Write the 2D slice (x rows, y columns)
    for (int x = 0; x < n_total; ++x) {
        for (int y = 0; y < m; ++y) {
            outfile << grid[idx(x, y, z_slice, m, l)] << (y == m - 1 ? "" : " ");
        }
        outfile << "\n";
    }

    outfile.close();
    return true;
}


// --- Main simulation step functions (Parallelized) ---

// Rule 2a: S cell random walk and corrosion
void random_walk_s(std::vector<int>& grid, int n_total, int m, int l,
                   float p0, float p1, float p2,
                   std::atomic<int>& s_to_n_counter,
                   std::atomic<int>& prob_check_passed_counter,
                   std::vector<std::mt19937>& rngs, // Per-thread RNGs
                   std::vector<std::uniform_real_distribution<float>>& dists // Per-thread distributions
                   ) {
    #pragma omp parallel for collapse(3) // Parallelize over the 3D grid
    for (int x = 0; x < n_total; ++x) {
        for (int y = 0; y < m; ++y) {
            for (int z = 0; z < l; ++z) {
                size_t current_idx = idx(x, y, z, m, l);
                std::atomic_ref<int> current_cell_ref(grid[current_idx]); // Atomic reference to current cell
                int current_cell_type = current_cell_ref.load(); // Read current type

                if (current_cell_type == CELL_S_SOL) {
                    int thread_id = omp_get_thread_num();
                    std::mt19937& engine = rngs[thread_id]; // Get thread-local RNG engine
                    std::uniform_real_distribution<float>& dist = dists[thread_id]; // Thread-local distribution
                    std::uniform_int_distribution<int> dir_dist(0, 5); // Distribution for direction


                    int direction = dir_dist(engine); // Choose random direction
                    int nx = x, ny = y, nz = z;

                    // Calculate neighbor coordinates (Von Neumann neighborhood)
                    // Periodic boundaries for y and z, clamping for x
                    if (direction == 0) { ny = (y - 1 + m) % m; }      // Back
                    else if (direction == 1) { ny = (y + 1) % m; }      // Front
                    else if (direction == 2) { nx = std::max(0, x - 1); } // Down (towards solution)
                    else if (direction == 3) { nx = std::min(n_total - 1, x + 1); } // Up (towards metal)
                    else if (direction == 4) { nz = (z - 1 + l) % l; }      // Left
                    else { nz = (z + 1) % l; }                      // Right

                    // Check if neighbor is within valid *logical* bounds (x is clamped already)
                     // No need for nx bounds check due to clamping above
                    size_t neighbor_idx = idx(nx, ny, nz, m, l);
                    std::atomic_ref<int> neighbor_cell_ref(grid[neighbor_idx]); // Atomic reference to neighbor

                    int target_type = neighbor_cell_ref.load(); // Read neighbor type

                    // --- Apply Rules ---

                    if (target_type == CELL_N_SOL) { // S<->N Swap Attempt
                        int expected = CELL_N_SOL;
                        // Try to change N at neighbor_idx to S
                        if (neighbor_cell_ref.compare_exchange_strong(expected, CELL_S_SOL)) {
                            // If successful, change current cell S to N
                            current_cell_ref.store(CELL_N_SOL);
                        }
                        // If CAS fails, another thread likely changed the neighbor, so do nothing

                    } else if (target_type >= CELL_GRAIN && target_type <= CELL_PRECIP) { // S+M -> P+N Corrosion Attempt
                        float corrosion_prob = 0.0f;
                        if (target_type == CELL_GRAIN) corrosion_prob = p0;
                        else if (target_type == CELL_GB) corrosion_prob = p1;
                        else if (target_type == CELL_PRECIP) corrosion_prob = p2;

                        float rand_val_for_corrosion = dist(engine); // Generate random number JUST before check

                        if (rand_val_for_corrosion < corrosion_prob) { // Corrosion occurs
                             int expected_metal = target_type; // Expect the specific metal type
                             // Try to change M at neighbor_idx to P
                             if (neighbor_cell_ref.compare_exchange_strong(expected_metal, CELL_PRODUCT)) {
                                 // If successful:
                                 prob_check_passed_counter.fetch_add(1); // Increment diagnostic counter
                                 current_cell_ref.store(CELL_N_SOL);    // Change current cell S to N
                                 s_to_n_counter.fetch_add(1);          // Increment counter for replenishment
                             }
                             // If CAS fails, target was already changed (e.g., to P by another thread), do nothing
                        }
                        // Else (no corrosion): S + M -> S + M (no change)

                    } else if (target_type == CELL_S_SOL) { // S<->S Swap Attempt (More complex)
                         int expected_s = CELL_S_SOL;
                         // Try to change target S to temporary SWAPPING state
                         if (neighbor_cell_ref.compare_exchange_strong(expected_s, CELL_S_SWAPPING)) {
                             // If successful, we 'own' the target cell temporarily.
                             // Change current cell to S (it remains S, but confirms the swap start)
                             current_cell_ref.store(CELL_S_SOL); // No real change needed here? Or set to N if source moves? Check Python logic.
                             // Python logic: grid[idx] = CELL_S_SOL (redundant), atomicExch(&grid[nidx], CELL_S_SOL) -> Changes target back
                             // Let's assume the source S moves *into* the target position logically
                             // So, the source cell *should* become what the target *was* if it were empty. But it's S.
                             // The goal is just to swap positions. The CAS ensures only one thread swaps with this target.
                             // Final step: change the target back from SWAPPING to S.
                             neighbor_cell_ref.store(CELL_S_SOL); // Could use exchange, but store is fine here as we 'own' it.
                             // The *source* cell (current_idx) should logically be empty now, but since we are iterating
                             // through S cells, another S might move into it later, or maybe it should become N?
                             // The Python code seems to imply the original S at current_idx just stays put if the swap partner is S.
                             // Let's re-examine Python:
                             //  int old_target = atomicCAS(&grid[nidx], CELL_S_SOL, CELL_S_SWAPPING);
                             //  if (old_target == CELL_S_SOL) {
                             //      grid[idx] = CELL_S_SOL; // Source remains S
                             //      atomicExch(&grid[nidx], CELL_S_SOL); // Target becomes S again
                             //  }
                             // This CUDA code doesn't actually *swap* the cells' identities, it just ensures
                             // that if two S cells try to move into the same spot, only one interaction happens,
                             // effectively acting like a bump/no movement scenario handled atomically.
                             // So the C++ code above correctly reflects this: if CAS succeeds, we just revert target to S.

                         }
                         // If CAS fails, another S is interacting with the target, so this thread does nothing.
                    }
                    // Implicit: S + P or S + Type3 -> S + P / S + Type3 (no reaction, no movement)
                } // end if (current_cell_type == CELL_S_SOL)
            } // end z loop
        } // end y loop
    } // end x loop
}


// Rule 2b: P cell random walk
void move_products(std::vector<int>& grid, int n_total, int m, int l, float p_Pmove,
                   std::vector<std::mt19937>& rngs, // Per-thread RNGs
                   std::vector<std::uniform_real_distribution<float>>& dists // Per-thread distributions
                  ) {

    #pragma omp parallel for collapse(3) // Parallelize over the 3D grid
    for (int x = 0; x < n_total; ++x) {
        for (int y = 0; y < m; ++y) {
            for (int z = 0; z < l; ++z) {
                 size_t current_idx = idx(x, y, z, m, l);
                 // Non-atomic read is okay here, as we only care if *this* thread sees a P.
                 // If it changes before we act, our atomic operations later will handle it.
                 if (grid[current_idx] == CELL_PRODUCT) {
                     int thread_id = omp_get_thread_num();
                     std::mt19937& engine = rngs[thread_id];
                     std::uniform_real_distribution<float>& dist = dists[thread_id];

                     if (dist(engine) < p_Pmove) { // Check move probability
                         // Find swappable neighbors (N or S)
                         int dx[] = {0, 0, -1, 1, 0, 0};
                         int dy[] = {-1, 1, 0, 0, 0, 0};
                         int dz[] = {0, 0, 0, 0, -1, 1};
                         std::vector<size_t> swappable_neighbor_indices;
                         swappable_neighbor_indices.reserve(6); // Reserve space

                         for (int i = 0; i < 6; ++i) {
                             int nx = x + dx[i];
                             int ny = y + dy[i];
                             int nz = z + dz[i];

                             // Check bounds (x clamped, y/z periodic)
                             if (nx >= 0 && nx < n_total) {
                                 ny = (ny % m + m) % m; // Ensure positive modulo result
                                 nz = (nz % l + l) % l; // Ensure positive modulo result
                                 size_t neighbor_idx = idx(nx, ny, nz, m, l);
                                 // Read neighbor type non-atomically (will re-check with atomic)
                                 if (is_swappable_solution(grid[neighbor_idx])) {
                                     swappable_neighbor_indices.push_back(neighbor_idx);
                                 }
                             }
                         }

                         if (!swappable_neighbor_indices.empty()) {
                             // Choose a random swappable neighbor
                             std::uniform_int_distribution<size_t> neighbor_dist(0, swappable_neighbor_indices.size() - 1);
                             size_t target_idx = swappable_neighbor_indices[neighbor_dist(engine)];

                             // --- Atomic Swap Attempt ---
                             std::atomic_ref<int> current_cell_ref(grid[current_idx]);
                             std::atomic_ref<int> target_cell_ref(grid[target_idx]);

                             // Try to take the solution cell, putting P there
                             int old_target_val = target_cell_ref.exchange(CELL_PRODUCT);

                             // Check if we actually got a solution cell (N or S)
                             if (is_swappable_solution(old_target_val)) {
                                 // Success! Put the original solution type (N or S) into the current cell
                                 current_cell_ref.store(old_target_val);
                             } else {
                                 // Failure: We grabbed something else (maybe another P moved first).
                                 // Put the original value back into the target cell.
                                 target_cell_ref.store(old_target_val);
                                 // The current cell remains P.
                             }
                         } // end if swappable neighbors found
                     } // end if move probability check passes
                 } // end if cell is product
            } // end z
        } // end y
    } // end x
}

// --- Main Simulation Loop ---
std::vector<std::string> simulate_corrosion(
    std::vector<int>& corr_grid, // Grid passed by reference
    int n_total, int m, int l, int solution_thickness,
    float p0, float p1, float p2, float p_Pmove,
    int steps,
    int absolute_replenish_cap,
    int frame_interval,
    const std::string& frame_dir,
    int z_slice_anim)
{
    std::vector<std::string> frame_files; // To potentially store filenames if needed later

    // --- Initialize Random Number Generators (one per thread) ---
    int max_threads = omp_get_max_threads();
    std::vector<std::mt19937> rngs(max_threads);
    std::vector<std::uniform_real_distribution<float>> dists(max_threads);
    std::random_device rd;
    for (int i = 0; i < max_threads; ++i) {
        rngs[i].seed(rd() + i); // Seed each engine differently
        dists[i] = std::uniform_real_distribution<float>(0.0f, 1.0f);
    }
    std::cout << "Initialized " << max_threads << " RNGs for parallel execution." << std::endl;


    // --- Atomic Counters ---
    std::atomic<int> s_to_n_counter(0);
    std::atomic<int> prob_check_passed_counter(0);

    // --- Setup Frame Directory ---
     if (std::filesystem::exists(frame_dir)) {
         try {
             std::filesystem::remove_all(frame_dir);
         } catch (const std::exception& e) {
              std::cerr << "Warning: Could not remove existing frame directory: " << e.what() << std::endl;
         }
     }
     try {
          std::filesystem::create_directories(frame_dir);
          std::cout << "Saving animation frame data every " << frame_interval << " steps to '" << frame_dir << "/'" << std::endl;
     } catch (const std::exception& e) {
          std::cerr << "Error creating frame directory '" << frame_dir << "': " << e.what() << std::endl;
          return frame_files; // Return empty list if directory fails
     }


    std::cout << "Using ABSOLUTE N->S Replenish Cap = " << absolute_replenish_cap << " per step (activated by corrosion)" << std::endl;

    // --- Simulation Loop ---
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < steps; ++step) {
        // Reset counters for the step
        s_to_n_counter.store(0);
        prob_check_passed_counter.store(0);

        // 1. S cell walk/corrosion kernel call
        random_walk_s(corr_grid, n_total, m, l, p0, p1, p2,
                      s_to_n_counter, prob_check_passed_counter, rngs, dists);

        // 2. P cell move kernel call
        move_products(corr_grid, n_total, m, l, p_Pmove, rngs, dists);

        // --- Synchronization Point (Implicit in OpenMP barrier at end of parallel regions) ---
        // No explicit sync needed like cp.cuda.Device().synchronize() unless using async tasks

        // 3. Concentration Maintenance (Sequential part, uses RNG[0])
        int s_converted_count = s_to_n_counter.load();
        if (s_converted_count > 0) {
             convert_n_to_s(corr_grid, s_converted_count, solution_thickness, m, l, absolute_replenish_cap, rngs);
        }


        // 4. Visualization & Cell Counts
        if (step % frame_interval == 0 || step == steps - 1) {
            long long s_count = 0, n_count = 0, p_count = 0, m0_count = 0, m1_count = 0, m2_count = 0;
            // Simple sequential count for verification/logging
            for(int cell_val : corr_grid) {
                switch(cell_val) {
                    case CELL_S_SOL: s_count++; break;
                    case CELL_N_SOL: n_count++; break;
                    case CELL_PRODUCT: p_count++; break;
                    case CELL_GRAIN: m0_count++; break;
                    case CELL_GB: m1_count++; break;
                    case CELL_PRECIP: m2_count++; break;
                }
            }
            int prob_checks_ok = prob_check_passed_counter.load();

             auto current_time = std::chrono::high_resolution_clock::now();
             std::chrono::duration<double> elapsed = current_time - start_time;

            std::cout << "\nStep " << step << " [" << std::fixed << std::setprecision(2) << elapsed.count() << "s]: "
                      << "S=" << s_count << ", N=" << n_count << ", P=" << p_count
                      << ", M0=" << m0_count << ", M1=" << m1_count << ", M2=" << m2_count
                      << ", ProbChecksOK=" << prob_checks_ok;


            bool saved = save_slice_to_file(corr_grid, step, z_slice_anim, n_total, m, l, frame_dir);
            if (!saved) {
                std::cerr << "Warning: Failed to save frame data at step " << step << std::endl;
            } else {
                 // Optionally store filename if needed later for GIF script
                 // frame_files.push_back(frame_dir + "/...");
            }
        } else {
             // Progress indication for steps without full output
             if (step % 10 == 0) { // Print a dot every 10 steps
                 std::cout << "." << std::flush;
             }
        }


    } // End simulation loop

     auto end_time = std::chrono::high_resolution_clock::now();
     std::chrono::duration<double> total_elapsed = end_time - start_time;
     std::cout << "\n\nSimulation finished " << steps << " steps in "
               << std::fixed << std::setprecision(2) << total_elapsed.count() << " seconds." << std::endl;


    return frame_files; // Return list (currently empty, but could hold filenames)
}


// --- Main Execution Logic ---
int main() {
    // --- Parameters ---
    // Metal grid dimensions (using placeholder generator)
    int grid_size_m = 100; // Example dimension for metal part y/z
    int grid_size_n = 50;  // Example dimension for metal part x (depth)
    // Parameters for placeholder grid generation (if used)
    // int num_grains = 20; // Not directly used in simple placeholder
    // float precipitate_ratio = 0.8; // Not directly used in simple placeholder

    // Simulation Parameters
    double concentration_ratio = 0.2;
    int solution_thickness = 10;
    // Corrosion Probabilities (matched to Python example)
    float p0_grain = 0.1f;
    float p1_gb = 0.8f;
    float p2_precip = 0.4f;
    float p_Pmove = 0.9f; // Probability for Product movement
    int simulation_steps = 1500;
    int ABSOLUTE_REPLENISH_CAP = 50; // Max N->S conversion per step

    // Animation/Output Parameters
    int save_frame_interval = 100;
    std::string animation_dir = "corrosion_frames_cpp_cap_" + std::to_string(ABSOLUTE_REPLENISH_CAP);
    // Output GIF generation would need an external script reading the .txt frames
    int animation_z_slice; // Will be set after getting metal grid dimensions

    // --- Setup ---
    // Load or Generate Metal Grid
    // Option 1: Generate placeholder
     std::vector<int> bound_grid_metal = create_placeholder_metal_grid(grid_size_n, grid_size_m, grid_size_m);
     int n_metal = grid_size_n;
     int m = grid_size_m;
     int l = grid_size_m;
     animation_z_slice = l / 2; // Set slice based on actual dimension

    // Option 2: Load from file (requires a library like cnpy)
    // std::string metal_filepath = "grid_alloy_100_20_0.8.npy"; // Example filename
    // try {
    //     // cnpy::NpyArray metal_arr = cnpy::npy_load(metal_filepath);
    //     // n_metal = metal_arr.shape[0]; m = metal_arr.shape[1]; l = metal_arr.shape[2];
    //     // bound_grid_metal = metal_arr.as_vec<int>(); // Or correct type
    //     // animation_z_slice = l / 2;
    //     // std::cout << "Loaded metal grid from " << metal_filepath << ", shape: " << n_metal << "x" << m << "x" << l << std::endl;
    // } catch (const std::exception& e) {
    //     std::cerr << "Error loading metal grid file '" << metal_filepath << "': " << e.what() << std::endl;
    //     return 1;
    // }

    int n_total = n_metal + solution_thickness;

    // --- Print Setup Info ---
    std::cout << std::string(30, '-') << std::endl;
    std::cout << "Simulation Setup (C++ / OpenMP):" << std::endl;
    std::cout << "  Metal Grid Dimensions:  " << n_metal << "x" << m << "x" << l << std::endl;
    std::cout << "  Solution Thickness:     " << solution_thickness << std::endl;
    std::cout << "  Total Grid Dimensions:  " << n_total << "x" << m << "x" << l << std::endl;
    std::cout << "  Initial S ratio:        " << concentration_ratio << std::endl;
    std::cout << "  Corrosion Probs:        M0=" << p0_grain << ", M1=" << p1_gb << ", M2=" << p2_precip << std::endl;
    std::cout << "  Product Move Prob:      " << p_Pmove << std::endl;
    std::cout << "  Simulation Steps:       " << simulation_steps << std::endl;
    std::cout << "  Replenish Cap/Step:     " << ABSOLUTE_REPLENISH_CAP << std::endl;
    std::cout << "  Output Slice z:         " << animation_z_slice << std::endl;
    std::cout << "  Output Frame Dir:       " << animation_dir << std::endl;
    std::cout << std::string(30, '-') << std::endl;

    // --- Execute Simulation ---
    try {
        std::cout << "Initializing corrosion grid..." << std::endl;
        std::vector<int> corr_grid = initialize_corr_grid(bound_grid_metal, concentration_ratio, solution_thickness, n_metal, m, l);
        // Metal grid no longer needed after copy
        bound_grid_metal.clear();
        bound_grid_metal.shrink_to_fit();


        std::cout << "\nStarting C++/OpenMP simulation for " << simulation_steps << " steps..." << std::endl;
        std::vector<std::string> frame_files = simulate_corrosion(
            corr_grid, n_total, m, l, solution_thickness,
            p0_grain, p1_gb, p2_precip, p_Pmove,
            simulation_steps,
            ABSOLUTE_REPLENISH_CAP,
            save_frame_interval,
            animation_dir,
            animation_z_slice);

        std::cout << "\nSimulation completed." << std::endl;
        std::cout << "Frame data saved in: " << animation_dir << std::endl;
        std::cout << "Use an external tool (Python/Matplotlib, Gnuplot, etc.) to visualize the .txt slices." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\nAn unexpected error occurred: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\nAn unknown error occurred." << std::endl;
        return 1;
    }


    std::cout << "\nProgram finished." << std::endl;
    return 0;
}