import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from pydmd import BOPDMD
from scipy.interpolate import interp1d
import random

# Full dataset range
scan_dict = {f'{i:04d}': i / 100 for i in range(10, 50)}

# Function to randomly select training and testing points
def split_dataset(scan_dict, N, M, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)

    # Get all keys and values from the scan_dict
    all_keys = list(scan_dict.keys())
    all_values = list(scan_dict.values())

    # Randomly select N training points
    training_indices = random.sample(all_keys, N)
    training_values = [scan_dict[key] for key in training_indices]

    # Determine the range of the training set
    min_train = min(training_values)
    max_train = max(training_values)

    # Filter keys to ensure testing points are within the range of the training set
    remaining_keys = [key for key in all_keys if key not in training_indices]
    remaining_values = {key: scan_dict[key] for key in remaining_keys}
    valid_testing_keys = [
        key for key, value in remaining_values.items() 
        if min_train <= value <= max_train
    ]

    # Randomly select M testing points from the valid keys
    testing_indices = random.sample(valid_testing_keys, M)

    return training_indices, testing_indices

# Example usage
N = 10  # Number of training points
M = 5   # Number of testing points
random_seed = 5  # For reproducibility
interp_kind = 'cubic' # 'slinear', 'quadratic'

training_indices, testing_indices = split_dataset(scan_dict, N, M, random_seed)

# Load the data for the selected indices
X_data = {}
t_data = {}

# Load in GENE simulation data downsampled by factor of 100 (10*10)
for key in training_indices + testing_indices:
    X_data[key] = np.load(f'/global/cfs/cdirs/m3586/CBC_ROM/linear/ITG_adiabatic_ky_scan_dky_0.01/g1_{key}.npy')[:, ::10]
    t_data[key] = np.load(f'/global/cfs/cdirs/m3586/CBC_ROM/linear/ITG_adiabatic_ky_scan_dky_0.01/g1_{key}_times.npy')[::10]

# Print the selected training and testing indices
print("Training indices:", training_indices)
print("Testing indices:", testing_indices)

# Extract ky values for training and testing from scan_dict
ky_train = np.array([scan_dict[i] for i in training_indices])
ky_test = [scan_dict[i] for i in testing_indices]
print(f"ky train: {ky_train}")
print(f"ky test: {ky_test}")

# --- Compute GLOBAL basis from all TRAINING snapshots ---
big_matrix = np.hstack([X_data[idx] for idx in training_indices])
U, S, Vh = np.linalg.svd(big_matrix, full_matrices=False)
r = 1  # your chosen rank
V_global = U[:, :r]

A_list = []
b_list = []

# --- Fit BOPDMD for each training parameter with the GLOBAL basis ---
for idx in training_indices:
    dmd_model = BOPDMD(
        svd_rank=r,
        num_trials=10,
        varpro_opts_dict={"tol": 0.015},
        proj_basis=V_global,   # supply the same basis
        use_proj=True,
    )
    dmd_model.fit(X_data[idx], t=t_data[idx])
    
    # Now Atilde, b, etc. are consistent across parameters (same reduced coords)
    A_list.append(dmd_model.atilde)         # shape (r, r)
    b_list.append(dmd_model._b)             # shape (r,)

### CHANGES: Stack arrays AFTER we finish the loop
A_array = np.stack(A_list, axis=0)              # shape: (N, r, r)
b_array = np.stack(b_list, axis=0)              # shape: (N, r)

# --- Build interpolation functions for A and b ---
A_interp_funcs = []
for i in range(r):
    row_funcs = []
    for j in range(r):
        # Extract the (i,j) element across all parameters
        A_ij_values = A_array[:, i, j]
        # Create a 1D interpolant
        f_ij = interp1d(ky_train, A_ij_values, kind=interp_kind)
        row_funcs.append(f_ij)
    A_interp_funcs.append(row_funcs)

b_interp_funcs = []
for i in range(r):
    b_i_values = b_array[:, i]
    f_b_i = interp1d(ky_train, b_i_values, kind=interp_kind)
    b_interp_funcs.append(f_b_i)

# --- Forecast at TEST parameters ---
for test_idx_idx, test_idx in enumerate(testing_indices):

    # Interpolate A at the test parameter
    A_p = np.zeros((r, r), dtype=complex)
    for row in range(r):
        for col in range(r):
            A_p[row, col] = A_interp_funcs[row][col](ky_test[test_idx_idx])

    # Interpolate b
    b_p = np.zeros(r, dtype=complex)
    for i in range(r):
        b_p[i] = b_interp_funcs[i](ky_test[test_idx_idx])

    # Option A: Recompute eigenvalues from A_p
    eigs_p, vecs_p = np.linalg.eig(A_p)

    modes_full = V_global @ vecs_p  # shape (n, r), these are the DMD modes in full space

    print(f"\n--- Testing param ky={ky_test[test_idx_idx]} ---")
    print(f"Atilde shape={A_p.shape}, eigenvalues={eigs_p}")

    test_dmd_model = dmd_model

    # 3) Overwrite with our interpolated operator, eigenvalues, etc.
    test_dmd_model.operator._Atilde      = A_p
    test_dmd_model.operator._b           = b_p
    test_dmd_model._amplitudes           = b_p # same as b vector
    test_dmd_model.operator._eigenvalues = eigs_p
    test_dmd_model.operator._modes       = modes_full

    # Now forecast for these test times
    t_test = t_data[test_idx]
    pred_mean, pred_var = test_dmd_model.forecast(t_test)
    print("Forecast shapes:", pred_mean.shape, pred_var.shape)

    # Optionally get the reconstructed data (which will reconstruct
    # over the 'dummy' times by default). 
    results = test_dmd_model.reconstructed_data

    # Save reconstruction
    filename_modes = f'/global/cfs/cdirs/m3586/CBC_ROM/linear/ITG_adiabatic_ky_scan_dky_0.01/bopdmd_rank{r}_{test_idx}_modes.npy'
    filename_mean  = f'/global/cfs/cdirs/m3586/CBC_ROM/linear/ITG_adiabatic_ky_scan_dky_0.01/bopdmd_rank{r}_{test_idx}_mean.npy'
    filename_var   = f'/global/cfs/cdirs/m3586/CBC_ROM/linear/ITG_adiabatic_ky_scan_dky_0.01/bopdmd_rank{r}_{test_idx}_var.npy'
    filename_recon = f'/global/cfs/cdirs/m3586/CBC_ROM/linear/ITG_adiabatic_ky_scan_dky_0.01/bopdmd_rank{r}_{test_idx}_recon.npy'

    np.save(filename_modes, modes_full)
    np.save(filename_mean,  pred_mean)
    np.save(filename_var,   pred_var)
    np.save(filename_recon, results)

