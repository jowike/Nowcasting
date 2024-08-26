import math
import cmath
import inspect
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh
from scipy.linalg import kron, block_diag  # inv,
from pysrc.remNaNs_spline import remNaNs_spline


def safe_round(
    matrix,
    precision=64,
    clip_threshold=1e64,
):
    matrix = np.clip(matrix, -clip_threshold, clip_threshold)
    return np.round(matrix, precision)


def round_inputs(func, precision=16):
    def wrapper(*args, **kwargs):
        # Get the signature of the function
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Round all arguments
        for name, value in bound_args.arguments.items():
            if isinstance(value, np.ndarray) and value.dtype == float:
                bound_args.arguments[name] = safe_round(value, precision=precision)
            elif isinstance(value, dict):
                for key in value.copy().keys():
                    if isinstance(value[key], np.ndarray) and value[key].dtype == float:
                        arr = value.pop(key)
                        value[key] = safe_round(arr, precision=precision)
                bound_args.arguments[name] = value

        # Call the original function with rounded arguments
        return func(*bound_args.args, **bound_args.kwargs)

    return wrapper


# def round_outputs(func, precision=64):
#     def wrapper(*args, **kwargs):
#         results = func(*args, **kwargs)

#         # If the result is not a tuple or list, convert it to a tuple for consistent processing
#         if not isinstance(results, (tuple, list)):
#             results = (results,)
#             single_output = True
#         else:
#             single_output = False

#         # Round all results
#         outputs = []
#         for item in results:
#             if isinstance(item, np.ndarray) and item.dtype == float:
#                 item = safe_round(item, precision=precision)
#             elif isinstance(item, dict):
#                 for key, value in item.items():
#                     if isinstance(value, np.ndarray) and value.dtype == float:
#                         item[key] = safe_round(value, precision=precision)
#                     elif isinstance(value, (float, int)):
#                         item[key] = round(value, precision)
#             elif isinstance(item, (float, int)):
#                 item = round(item, precision)
#             outputs.append(item)

#         # If the function originally returned a single item, return it as a single item
#         if single_output:
#             return outputs[0]

#         return tuple(outputs)

#     return wrapper


# Function for matrix inversion with handling singular matrices
def safe_inv(matrix):
    # if np.isnan(matrix).any():
    #     matrix[~np.isfinite(matrix)] = 0

    try:
        inv_matrix = np.linalg.inv(matrix)
        if np.isnan(inv_matrix).any():
            raise ValueError("Matrix inversion resulted in NaNs.")
        return inv_matrix
    except np.linalg.LinAlgError:
        # print("Matrix is singular or nearly singular, using pseudoinverse.")
        return np.linalg.pinv(matrix)


def dfm(X, Spec, threshold=1e-5, max_iter=5000):
    """
    Runs the dynamic factor model.

    Parameters
    ----------
    X : array_like
        Kalman-smoothed data where missing values are replaced by their expectations.
    Par : dict
        A dictionary containing the following parameters:
        - `blocks`: Block loadings.
        - `nQ`: Number of quarterly series.
        - `p`: Number of lags in the transition matrix.
        - `r`: Number of common factors for each block.

    Returns
    -------
    dict
        A structure of model results with the following fields:
        - `X_sm` : array_like
            Kalman-smoothed data where missing values are replaced by their expectations.
        - `Z` : array_like
            Smoothed states. Rows give time, and columns are organized according to `Res.C`.
        - `C` : array_like
            Observation matrix. The rows correspond to each series, and the columns are organized as follows:
            - Columns 1-5: Factor loadings for the first block in reverse-chronological order
              (f^G_t, f^G_t-1, f^G_t-2, f^G_t-3, f^G_t-4).
            - Columns 6-10, 11-15, and 16-20: Loadings for the second, third, and fourth blocks, respectively.
        - `R` : array_like
            Covariance for observation matrix residuals.
        - `A` : array_like
            Transition matrix. This is a square matrix following the same organization scheme as `Res.C`'s columns.
            Identity matrices are used to account for matching terms on both sides.
        - `Q` : array_like
            Covariance for transition equation residuals.
        - `Mx` : array_like
            Series mean.
        - `Wx` : array_like
            Series standard deviation.
        - `Z_0` : array_like
            Initial value of the state.
        - `V_0` : array_like
            Initial value of the covariance matrix.
        - `r` : int
            Number of common factors for each block.
        - `p` : int
            Number of lags in the transition equation.

    References
    ----------
    Marta Banbura, Domenico Giannone, and Lucrezia Reichlin,
    "Nowcasting," 2010, in Michael P. Clements and David F. Hendry (eds.),
    Oxford Handbook on Economic Forecasting.
    """

    # Store model parameters
    # DFM input specifications: See documentation for details
    Par = {
        "blocks": np.array(Spec["blocks"]),
        "nQ": sum(freq == "q" for freq in Spec["frequency"]),
        "p": 1,
        "r": np.ones(np.array(Spec["blocks"]).shape[1], dtype=int),
    }

    print("\n\n\nTable 3: Block Loading Structure")
    print(
        pd.DataFrame(
            Spec["blocks"],
            index=[name.replace(" ", "_") for name in Spec["seriesname"]],
            columns=Spec["blocknames"],
        )
    )

    print("Estimating the dynamic factor model (DFM) ... \n\n")

    T, N = X.shape
    nQ = Par["nQ"]  # Number of quarterly series
    r = Par["r"]
    p = Par["p"]
    blocks = Par["blocks"]

    i_idio = np.array([1] * (N - nQ) + [0] * nQ, dtype=bool).reshape(-1, 1)

    # R*Lambda = q; Contraints on the loadings of the quartrly variables
    R_mat = np.array(
        [[2, -1, 0, 0, 0], [3, 0, -1, 0, 0], [2, 0, 0, -1, 0], [1, 0, 0, 0, -1]]
    )

    q = np.zeros(4).reshape(-1, 1)

    # Prepare data -----------------------------------------------------------
    Mx = np.nanmean(X, axis=0)
    Wx = np.nanstd(X, axis=0)
    xNaN = (X - Mx) / Wx  # Standardize series

    # Initial Conditions -----------------------------------------------------
    optNaN = {
        "method": 2,  # Remove leading and closing zeros
        "k": 3,  # Setting for filter(): See remNaN_spline
    }  # options

    A, C, Q, R, Z_0, V_0 = InitCond(xNaN, r, p, blocks, optNaN, R_mat, q, nQ, i_idio)

    # Initialize EM loop values
    previous_loglik = -np.inf
    num_iter = 0
    LL = [-np.inf]
    converged = 0

    # y for the estimation is WITH missing data
    y = xNaN.T

    # EM LOOP ----------------------------------------------------------------

    # The model can be written as
    # y = C*Z + e;
    # Z = A*Z(-1) + v
    # where y is NxT, Z is (pr)xT, etc

    # Remove the leading and ending nans
    optNaN = {"method": 3, "k": optNaN["k"]}
    y_est, _ = remNaNs_spline(xNaN, optNaN)
    y_est = y_est.T

    while num_iter < max_iter and not converged:  # Loop until converges or max iter.
        C_new, R_new, A_new, Q_new, Z_0, V_0, loglik = EMstep(  # Applying EM algorithm
            y_est, A, C, Q, R, Z_0, V_0, r, p, R_mat, q, nQ, i_idio, blocks
        )

        C, R, A, Q = C_new, R_new, A_new, Q_new

        if num_iter > 2:  # Checking convergence
            converged, decrease = em_converged(
                loglik, previous_loglik, threshold, check_decreased=False
            )

        if num_iter % 10 == 0 and num_iter > 0:  # Print updates to command window
            print(f"Now running the {num_iter}th iteration of max {max_iter}")
            # change = 100 * (
            #     (loglik - previous_loglik) / (previous_loglik + np.finfo(float).eps)
            # )  # Avoid division by zero
            # print(f"  Loglik: {loglik}   (% Change: {change:.2f}%)")
            print(f"  Loglik: {loglik}   ")

        LL.append(loglik)
        previous_loglik = loglik
        num_iter = num_iter + 1

    if num_iter < max_iter:
        print(f"Successful: Convergence at {num_iter} iterations")
    else:
        print("Stopped because maximum iterations reached")

    # Final run of the Kalman filter
    Zsmooth, _, _, _ = runKF(y, A, C, Q, R, Z_0, V_0)
    Zsmooth = Zsmooth.T
    x_sm = Zsmooth[1:] @ C.T  # Get smoothed X

    # Loading the structure with the results --------------------------------
    Res = {
        "x_sm": x_sm,
        "X_sm": Wx * x_sm + Mx,  # Unstandardized, smoothed
        "Z": Zsmooth[1:],
        "C": C,
        "R": R,
        "A": A,
        "Q": Q,
        "Mx": Mx,
        "Wx": Wx,
        "Z_0": Z_0,
        "V_0": V_0,
        "r": r,
        "p": p,
    }

    # Display output
    # Table with names and factor loadings

    nM = len(Spec["seriesname"]) - nQ  # Number monthly series
    nLags = max(Par["p"], 5)  # 5 comes from monthly-quarterly aggregation
    nFactors = sum(Par["r"])

    # Slicing to select rows and specific columns
    rows = slice(0, nM)  # 0 to nM-1 in Python
    cols = np.arange(0, nFactors * 5, 5)  # Only select lag(0) terms

    try:
        print("Table 4: Factor Loadings for Monthly Series")
        # Create a DataFrame from the selected rows and columns
        monthly_loadings = pd.DataFrame(
            Res["C"][rows, cols],  # Apply the slicing
            index=[
                name.replace(" ", "_") for name in Spec["seriesname"][:nM]
            ],  # RowNames
            columns=Spec["blocknames"],  # VariableNames
        )
        print(monthly_loadings)
        print("\n\n\n")

        print("Table 5: Quarterly Loadings Sample (Global Factor)")
        # Slicing for quarterly loadings (select last nQ rows and first 5 columns)
        quarterly_loadings = pd.DataFrame(
            Res["C"][-nQ:, :5],  # Select only quarterly series
            index=[
                name.replace(" ", "_") for name in Spec["seriesname"][-nQ:]
            ],  # RowNames
            columns=[
                "f1_lag0",
                "f1_lag1",
                "f1_lag2",
                "f1_lag3",
                "f1_lag4",
            ],  # VariableNames
        )
        print(quarterly_loadings)
        print("\n\n\n")
    except Exception as e:
        print(f"Error displaying factor loadings: {e}")

    # Table with AR model on factors (factors with AR parameter and variance of residuals)
    A_terms = np.diag(Res["A"])  # Transition equation terms
    Q_terms = np.diag(Res["Q"])  # Covariance matrix terms

    try:
        print("Table 6: Autoregressive Coefficients on Factors")
        ar_coefficients_factors = pd.DataFrame(
            {  # Only select lag(0) terms
                "AR_Coefficient": A_terms[0 : nFactors * 5 : 5],
                "Variance_Residual": Q_terms[0 : nFactors * 5 : 5],
            },
            index=[name.replace(" ", "_") for name in Spec["blocknames"]],
        )
        print(ar_coefficients_factors)
        print("\n\n\n")
    except Exception as e:
        print(f"Error displaying autoregressive coefficients on factors: {e}")

    # Table with AR model idiosyncratic errors (factors with AR parameter and variance of residuals)
    try:
        print("Table 7: Autoregressive Coefficients on Idiosyncratic Component")
        # Construct the indices similar to MATLAB
        idiosyncratic_indices = np.concatenate(
            [
                np.arange(
                    nFactors * 5, nFactors * 5 + nM
                ),  # MATLAB: nFactors*5+1 to nFactors*5+nM
                np.arange(
                    nFactors * 5 + nM, len(A_terms), 5
                ),  # MATLAB: nFactors*5+nM+1 to end, step 5
            ]
        )  # 21:50 give monthly, 51:5:61 give quarterly

        # Create the DataFrame
        idiosyncratic_coefficients = pd.DataFrame(
            {
                "AR_Coefficient": A_terms[idiosyncratic_indices],
                "Variance_Residual": Q_terms[idiosyncratic_indices],
            },
            index=[name.replace(" ", "_") for name in Spec["seriesname"]],
        )

        print(idiosyncratic_coefficients)
    except Exception as e:
        print(
            f"Error displaying autoregressive coefficients on idiosyncratic component: {e}"
        )

    return Res


# PROCEDURES -------------------------------------------------------------
# def EMstep(y, A, C, Q, R, Z_0, V_0, r,p,R_mat,q,nQ,i_idio,blocks):
#     """
#     Applies EM algorithm for parameter reestimation

#     Description:
#       EMstep reestimates parameters based on the Estimation Maximization (EM)
#       algorithm. This is a two-step procedure:
#         (1) E-step: the expectation of the log-likelihood is calculated using
#             previous parameter estimates.
#         (2) M-step: Parameters are re-estimated through the maximisation of
#             the log-likelihood (maximize result from (1)).

#       See "Maximum likelihood estimation of factor models on data sets with
#       arbitrary pattern of missing data" for details about parameter
#       derivation (Banbura & Modugno, 2010). This procedure is in much the
#       same spirit.

#     Input:
#         y:      Series data
#         A:      Transition matrix
#         C:      Observation matrix
#         Q:      Covariance for transition equation residuals
#         R:      Covariance for observation matrix residuals
#         Z_0:    Initial values of factors
#         V_0:    Initial value of factor covariance matrix
#         r:      Number of common factors for each block (e.g. vector [1 1 1 1])
#         p:      Number of lags in transition equation
#         R_mat:  Estimation structure for quarterly variables (i.e. "tent")
#         q:      Constraints on loadings
#         nQ:     Number of quarterly series
#         i_idio: Indices for monthly variables
#         blocks: Block structure for each series (i.e. for a series, the structure
#                 [1 0 0 1] indicates loadings on the first and fourth factors)

#     Output:
#         C_new: Updated observation matrix
#         R_new: Updated covariance matrix for residuals of observation matrix
#         A_new: Updated transition matrix
#         Q_new: Updated covariance matrix for residuals for transition matrix
#         Z_0:   Initial value of state
#         V_0:   Initial value of covariance matrix
#         loglik: Log likelihood

#     References:
#       "Maximum likelihood estimation of factor models on data sets with
#       arbitrary pattern of missing data" by Banbura & Modugno (2010).
#       Abbreviated as BM2010
#     """
#     # Initialize preliminary values

#     # Store series/model values
#     n, T = y.shape
#     nM = n - nQ  # Number of monthly series
#     pC = R_mat.shape[1]
#     ppC = max(p,pC)
#     num_blocks = blocks.shape[1]  # Number of blocks

#     # ESTIMATION STEP: Compute the (expected) sufficient statistics for a single
#     # Kalman filter sequence

#     # Running the Kalman filter and smoother with current parameters
#     # Note that log-liklihood is NOT re-estimated after the runKF step: This
#     # effectively gives the previous iteration's log-likelihood
#     # For more information on output, see runKF
#     Zsmooth, Vsmooth, VVsmooth, loglik = runKF(y, A, C, Q, R, Z_0, V_0)
#     # MAXIMIZATION STEP (TRANSITION EQUATION)
#     # See (Banbura & Modugno, 2010) for details.

#     # Initialize output
#     A_new = A.copy()
#     Q_new = Q.copy()
#     V_0_new = V_0.copy()

#     # 2A. UPDATE FACTOR PARAMETERS INDIVIDUALLY ----------------------------

#     for i in range(num_blocks):  # Loop for each block: factors are uncorrelated
#         # SETUP INDEXING
#         r_i = r[i]  # r_i = 1 if block is loaded
#         rp = r_i*p
#         rp1 = sum(r[:i]) * ppC
#         b_subset = slice(rp1, rp1 + rp)  # Subset blocks: Helps for subsetting Zsmooth, Vsmooth
#         t_start = rp1+1  # Transition matrix factor idx start
#         t_end = rp1+r_i*ppC  # Transition matrix factor idx end

#         # ESTIMATE FACTOR PORTION OF Q, A
#         # Note: EZZ, EZZ_BB, EZZ_FB are parts of equations 6 and 8 in BM 2010

#         # E[f_t*f_t' | Omega_T]
#         EZZ = Zsmooth[b_subset, 1:] @ Zsmooth[b_subset, 1:].T + np.sum(Vsmooth[b_subset, b_subset, 1:], axis=2)


#         # E[f_{t-1}*f_{t-1}' | Omega_T]
#         EZZ_BB = Zsmooth[b_subset, :-1] @ Zsmooth[b_subset, :-1].T + np.sum(Vsmooth[b_subset, b_subset, :-1], axis=2)

#         # E[f_t*f_{t-1}' | Omega_T]
#         EZZ_FB = Zsmooth[b_subset, 1:] @ Zsmooth[b_subset, :-1].T + np.sum(VVsmooth[b_subset, b_subset, :], axis=2)

#         # Select transition matrix/covariance matrix for block i
#         A_i = A[t_start:t_end, t_start:t_end]
#         Q_i = Q[t_start:t_end, t_start:t_end]

#         # Equation 6: Estimate VAR(p) for factor
#         A_i[:r_i, :rp] = EZZ_FB[:r_i, :rp] @ safe_inv(EZZ_BB[:rp, :rp])

#         # Equation 8: Covariance matrix of residuals of VAR
#         Q_i[:r_i, :r_i] = (EZZ[:r_i, :r_i] - A_i[:r_i, :rp] @ EZZ_FB[:r_i, :rp].T) / T

#         # Place updated results in output matrix
#         A_new[t_start:t_end, t_start:t_end] = A_i
#         Q_new[t_start:t_end, t_start:t_end] = Q_i
#         V_0_new[t_start:t_end, t_start:t_end] = Vsmooth[t_start:t_end, t_start:t_end, 0]

#     # 2B. UPDATING PARAMETERS FOR IDIOSYNCRATIC COMPONENT ------------------
#     rp1 = sum(r) * ppC  # Col size of factor portion
#     niM = np.sum(i_idio[:nM])  # Number of monthly values
#     t_start = rp1  # Start of idiosyncratic component index
#     i_subset = slice(t_start, rp1 + niM)  # Gives indices for monthly idiosyncratic component values

#     # Below 3 estimate the idiosyncratic component (for eqns 6, 8 BM 2010)
#     # E[f_t*f_t' | \Omega_T]
#     EZZ = np.diag(np.diag(Zsmooth[t_start:, 1:] @ Zsmooth[t_start:, 1:].T)) + \
#           np.diag(np.diag(np.sum(Vsmooth[t_start:, t_start:, 1:], axis=2)))

#     # E[f_{t-1}*f_{t-1}' | \Omega_T]
#     EZZ_BB = np.diag(np.diag(Zsmooth[t_start:, :-1] @ Zsmooth[t_start:, :-1].T)) + \
#              np.diag(np.diag(np.sum(Vsmooth[t_start:, t_start:, :-1], axis=2)))

#     # E[f_t*f_{t-1}' | \Omega_T]
#     EZZ_FB = np.diag(np.diag(Zsmooth[t_start:, 1:] @ Zsmooth[t_start:, :-1].T)) + \
#              np.diag(np.diag(np.sum(VVsmooth[t_start:, t_start:, :], axis=2)))

#     A_i = EZZ_FB @ np.diag(1.0 / np.diag(EZZ_BB))  # Equation 6
#     Q_i = (EZZ - A_i @ EZZ_FB.T) / T  # Equation 8

#     # Place updated results in output matrix
#     A_new[i_subset, i_subset] = A_i[:niM, :niM]
#     Q_new[i_subset, i_subset] = Q_i[:niM, :niM]
#     V_0_new[i_subset, i_subset] = np.diag(np.diag(Vsmooth[i_subset, i_subset, 0]))

#     #  3 MAXIMIZATION STEP (observation equation)

#     # INITIALIZATION AND SETUP ----------------------------------------------
#     Z_0 = Zsmooth[:, 0]  # zeros(size(Zsmooth,1),1);

#     # Set missing data series values to 0
#     nanY = np.isnan(y)
#     y[nanY] = 0

#     # LOADINGS
#     C_new = C.copy()

#     # Blocks
#     bl = np.unique(blocks, axis=0)  # Gives unique loadings
#     n_bl = bl.shape[0]  # Number of unique loadings

#     # Initialize indices: These later help with subsetting
#     bl_idxM = np.empty((n_bl, 0), dtype=bool)  # Indicator for monthly factor loadings
#     bl_idxQ = np.empty((n_bl, 0), dtype=bool)  # Indicator for quarterly factor loadings
#     R_con = None  # Block diagonal matrix giving monthly-quarterly aggreg scheme
#     q_con = np.empty((0, 1))  # Empty column vector

#     # Loop through each block
#     for i in range(num_blocks):
#         # Create the repeated block matrices
#         repeated_bl_Q = np.tile(bl[:, i:i+1].reshape(-1, 1), (1, r[i] * ppC))
#         bl_idxQ = np.hstack([bl_idxQ, repeated_bl_Q]) if bl_idxQ.size else repeated_bl_Q

#         matrices_to_concat = [bl_idxM, np.tile(bl[:, i:i+1].reshape(-1, 1), (1, r[i])), np.zeros((n_bl, r[i] * (ppC - 1)))]  # Monthly
#         matrices_to_concat = [matrix for matrix in matrices_to_concat if matrix.size]

#         # Update bl_idxQ and bl_idxM by concatenating horizontally
#         bl_idxM = np.hstack(matrices_to_concat)

#         # Construct block diagonal matrix R_con
#         block_kron = kron(R_mat, np.eye(r[i]))
#         R_con = block_diag(R_con, block_kron) if R_con is not None else block_kron

#         # Append zeros to q_con
#         zeros_to_append = np.zeros((r[i] * R_mat.shape[0], 1))
#         q_con = np.vstack([q_con, zeros_to_append])


#     # Indicator for monthly/quarterly blocks in observation matrix
#     bl_idxM = np.array(bl_idxM, dtype=bool)
#     bl_idxQ = np.array(bl_idxQ, dtype=bool)

#     i_idio_M = i_idio[:nM]  # Gives 1 for monthly series
#     n_idio_M = np.count_nonzero(i_idio_M)  # Number of monthly series
#     c_i_idio = np.cumsum(i_idio)  # Cumulative number of monthly series

#     for i in range(n_bl):  # Loop through unique loadings (e.g. [1 0 0 0], [1 1 0 0])
#         bl_i = bl[i, :]
#         rs = sum(r[bl_i.astype(bool)])  # Total num of blocks loaded

#         idx_i = np.where((blocks == bl_i).all(axis=1))[0]  # Indices for bl_i
#         idx_iM = idx_i[idx_i < nM]  # Only monthly
#         n_i = len(idx_iM)  # Number of monthly series

#         # Initialize sums in equation 13 of BGR 2010
#         denom = np.zeros((n_i * rs, n_i * rs))
#         nom = np.zeros((n_i, rs))

#         # Stores monthly indicies. These are done for input robustness
#         i_idio_i = i_idio_M[idx_iM]
#         i_idio_ii = c_i_idio[idx_iM]
#         i_idio_ii = i_idio_ii[i_idio_i.flatten()]

#         # UPDATE MONTHLY VARIABLES: Loop through each period ----------------
#         for t in range(T):
#             Wt = np.diag(~nanY[idx_iM, t])  # Gives selection matrix (1 for nonmissing values)

#             denom = (
#                 denom +  # E[f_t*t_t' | Omega_T]
#                 kron(
#                     (Zsmooth[np.where(bl_idxM[i, :]), t+1].reshape(-1,1) @ Zsmooth[np.where(bl_idxM[i, :]), t+1].reshape(-1,1).T + Vsmooth[np.where(bl_idxM[i, :]), np.where(bl_idxM[i, :]), t+1].reshape(-1,1)),
#                     Wt
#                 )
#             )

#             # denom = denom +...
#             #         kron(Zsmooth(bl_idxM(i, :), t+1) * Zsmooth(bl_idxM(i, :), t+1)' + ...
#             #         Vsmooth(bl_idxM(i, :), bl_idxM(i, :), t+1), Wt);
#             nom = (
#                 nom +   # E[y_t*f_t' | \Omega_T]
#                 y[idx_iM, t][:, np.newaxis] @ Zsmooth[np.where(bl_idxM[i, :]), t + 1].reshape(-1,1).T -
#                     Wt[:, np.where(i_idio_i)[0]] @ (Zsmooth[rp1 + i_idio_ii, t + 1].reshape(-1,1) @
#                                         Zsmooth[np.where(bl_idxM[i, :]), t + 1].reshape(-1,1).T +
#                                         Vsmooth[rp1 + i_idio_ii, :, :][:, np.where(bl_idxM[i, :])[0], t + 1])
#             )

#             # nom = nom + ...  E[y_t*f_t' | \Omega_T]
#             #       y(idx_iM, t) * Zsmooth(bl_idxM(i, :), t+1)' - ...
#             #       Wt(:, i_idio_i) * (Zsmooth(rp1 + i_idio_ii, t+1) * ...
#             #       Zsmooth(bl_idxM(i, :), t+1)' + ...
#             #       Vsmooth(rp1 + i_idio_ii, bl_idxM(i, :), t+1));

#         vec_C = safe_inv(denom) @ nom.flatten().reshape(-1,1)  # Eqn 13 BGR 2010

#         # Place updated monthly results in output matrix
#         C_new[idx_iM[:, None], np.where(bl_idxM[i, :])] = vec_C.reshape(n_i, rs)

#         # UPDATE QUARTERLY VARIABLES -----------------------------------------

#         idx_iQ = idx_i[idx_i >= nM]  # Index for quarterly series
#         rps = rs * ppC

#        # Monthly-quarterly aggregation scheme
#         R_con_i = R_con[:, bl_idxQ[i, :]]
#         q_con_i = q_con

#         no_c = ~np.any(R_con_i, axis=1)
#         R_con_i = R_con_i[~no_c, :]  # R_con_i(no_c,:) = [];
#         q_con_i = q_con_i[~no_c, :]  # q_con_i(no_c,:) = [];

#         # Loop through quarterly series in loading. This parallels monthly code
#         for j in idx_iQ:
#             # Initialization
#             denom = np.zeros((rps, rps))
#             nom = np.zeros((1, rps))

#             idx_jQ = j - nM  # Ordinal position of quarterly variable
#             # Loc of factor structure corresponding to quarterly var residuals
#             # i_idio_jQ = (rp1 + n_idio_M + 5*(idx_jQ-1)+1:rp1+ n_idio_M + 5*idx_jQ);
#             i_idio_jQ = np.arange(rp1 + n_idio_M + 5 * (idx_jQ - 1),
#                                   rp1 + n_idio_M + 5 * idx_jQ)

#             # Place quarterly values in output matrix
#             V_0_new[i_idio_jQ[:, None], i_idio_jQ] = Vsmooth[i_idio_jQ[:, None], i_idio_jQ, 0]
#             A_new[i_idio_jQ[0], i_idio_jQ[0]] = A_i[i_idio_jQ[0] - rp1, i_idio_jQ[0] - rp1]
#             Q_new[i_idio_jQ[0], i_idio_jQ[0]] = Q_i[i_idio_jQ[0] - rp1, i_idio_jQ[0] - rp1]
#             for t in range(T):
#                 Wt = np.diag([~nanY[j, t]])  # Selection matrix for quarterly values

#                 # Intermediate steps in BGR equation 13
#                 denom = (
#                     denom +
#                     kron(
#                         Zsmooth[np.where(bl_idxQ[i, :]), t + 1].reshape(-1,1) @ Zsmooth[np.where(bl_idxQ[i, :]), t + 1].reshape(-1,1).T +
#                                 Vsmooth[np.where(bl_idxQ[i, :]), np.where(bl_idxQ[i, :]), t + 1].reshape(-1,1),
#                         Wt
#                     )
#                 )
#                 # denom = denom + ...
#                 #         kron(Zsmooth(bl_idxQ(i,:), t+1) * Zsmooth(bl_idxQ(i,:), t+1)'...
#                 #         + Vsmooth(bl_idxQ(i,:), bl_idxQ(i,:), t+1), Wt);

#                 nom = nom + y[j, t] * Zsmooth[np.where(bl_idxQ[i, :]), t + 1].reshape(-1,1).T
#                 # nom = nom + y(j, t)*Zsmooth(bl_idxQ(i,:), t+1)';

#                 nom = (
#                     nom -
#                     Wt * (
#                         [1, 2, 3, 2, 1] @ Zsmooth[i_idio_jQ, t + 1].reshape(-1, 1) @
#                              Zsmooth[np.where(bl_idxQ[i, :]), t + 1].reshape(-1, 1).T +
#                              np.array([1, 2, 3, 2, 1]) @ Vsmooth[i_idio_jQ[:, None], np.where(bl_idxQ[i, :]), t + 1]
#                              )
#                 )
#                 # nom = nom -...
#                 #         Wt * ([1 2 3 2 1] * Zsmooth(i_idio_jQ,t+1) * ...
#                 #         Zsmooth(bl_idxQ(i,:),t+1)'+...
#                 #         [1 2 3 2 1]*Vsmooth(i_idio_jQ,bl_idxQ(i,:),t+1));

#             C_i = safe_inv(denom) @ nom.T

#             C_i_constr = (
#                 C_i -  # BGR equation 13
#                 safe_inv(denom) @ R_con_i.T @ safe_inv(R_con_i @ safe_inv(denom) @ R_con_i.T) @ (R_con_i @ C_i - q_con_i)
#             )

#             # Place updated values in output structure
#             C_new[j, np.where(bl_idxQ[i, :])] = C_i_constr.reshape(1, -1)
#             # C_new(j,bl_idxQ(i,:)) = C_i_constr;

#     # 3B. UPDATE COVARIANCE OF RESIDUALS FOR OBSERVATION EQUATION -----------
#     # Initialize covariance of residuals of observation equation
#     R_new = np.zeros((n, n))
#     for t in range(T):
#         Wt = np.diag(~nanY[:, t])  # Selection matrix

#         R_new = (
#             R_new +
#             (y[:, t][:, np.newaxis] - Wt @ C_new @ Zsmooth[:, t + 1][:, np.newaxis]) @ (y[:, t][:, np.newaxis] - Wt @ C_new @ Zsmooth[:, t + 1][:, np.newaxis]).T +
#             Wt @ C_new @ Vsmooth[:, :, t + 1] @ C_new.T @ Wt +
#             (np.eye(n) - Wt) @ R @ (np.eye(n) - Wt)
#             )

#         # R_new = R_new + (y(:,t) - ...  % BGR equation 15
#         #         Wt * C_new * Zsmooth(:, t+1)) * (y(:,t) - Wt*C_new*Zsmooth(:,t+1))'...
#         #       + Wt*C_new*Vsmooth(:,:,t+1)*C_new'*Wt + (eye(n)-Wt)*R*(eye(n)-Wt);

#     R_new = R_new/T
#     RR = np.diag(R_new).copy()  # RR(RR<1e-2) = 1e-2;
#     RR[np.where(i_idio_M.flatten())] = 1e-04  # Ensure non-zero measurement error. See Doz, Giannone, Reichlin (2012) for reference.
#     RR[nM:] = 1e-04
#     R_new = np.diag(RR)

#     return C_new, R_new, A_new, Q_new, Z_0, V_0_new, loglik


@round_inputs
def EMstep(y, A, C, Q, R, Z_0, V_0, r, p, R_mat, q, nQ, i_idio, blocks):
    """
    Applies EM algorithm for parameter reestimation

    Description:
      EMstep reestimates parameters based on the Estimation Maximization (EM)
      algorithm. This is a two-step procedure:
        (1) E-step: the expectation of the log-likelihood is calculated using
            previous parameter estimates.
        (2) M-step: Parameters are re-estimated through the maximisation of
            the log-likelihood (maximize result from (1)).

      See "Maximum likelihood estimation of factor models on data sets with
      arbitrary pattern of missing data" for details about parameter
      derivation (Banbura & Modugno, 2010). This procedure is in much the
      same spirit.

    Input:
        y:      Series data
        A:      Transition matrix
        C:      Observation matrix
        Q:      Covariance for transition equation residuals
        R:      Covariance for observation matrix residuals
        Z_0:    Initial values of factors
        V_0:    Initial value of factor covariance matrix
        r:      Number of common factors for each block (e.g. vector [1 1 1 1])
        p:      Number of lags in transition equation
        R_mat:  Estimation structure for quarterly variables (i.e. "tent")
        q:      Constraints on loadings
        nQ:     Number of quarterly series
        i_idio: Indices for monthly variables
        blocks: Block structure for each series (i.e. for a series, the structure
                [1 0 0 1] indicates loadings on the first and fourth factors)

    Output:
        C_new: Updated observation matrix
        R_new: Updated covariance matrix for residuals of observation matrix
        A_new: Updated transition matrix
        Q_new: Updated covariance matrix for residuals for transition matrix
        Z_0:   Initial value of state
        V_0:   Initial value of covariance matrix
        loglik: Log likelihood

    References:
      "Maximum likelihood estimation of factor models on data sets with
      arbitrary pattern of missing data" by Banbura & Modugno (2010).
      Abbreviated as BM2010
    """
    # Initialize preliminary values

    # Store series/model values
    n, T = y.shape
    nM = n - nQ  # Number of monthly series
    pC = R_mat.shape[1]
    ppC = max(p, pC)
    num_blocks = blocks.shape[1]  # Number of blocks

    # ESTIMATION STEP: Compute the (expected) sufficient statistics for a single
    # Kalman filter sequence

    # Running the Kalman filter and smoother with current parameters
    # Note that log-liklihood is NOT re-estimated after the runKF step: This
    # effectively gives the previous iteration's log-likelihood
    # For more information on output, see runKF
    Zsmooth, Vsmooth, VVsmooth, loglik = runKF(y, A, C, Q, R, Z_0, V_0)
    # MAXIMIZATION STEP (TRANSITION EQUATION)
    # See (Banbura & Modugno, 2010) for details.

    # Initialize output
    A_new = A.copy()
    Q_new = Q.copy()
    V_0_new = V_0.copy()

    # 2A. UPDATE FACTOR PARAMETERS INDIVIDUALLY ----------------------------

    for i in range(num_blocks):  # Loop for each block: factors are uncorrelated
        # SETUP INDEXING
        r_i = r[i]  # r_i = 1 if block is loaded
        rp = r_i * p
        rp1 = sum(r[:i]) * ppC
        b_subset = slice(
            rp1, rp1 + rp
        )  # Subset blocks: Helps for subsetting Zsmooth, Vsmooth
        t_start = rp1 + 1  # Transition matrix factor idx start
        t_end = rp1 + r_i * ppC  # Transition matrix factor idx end

        # ESTIMATE FACTOR PORTION OF Q, A
        # Note: EZZ, EZZ_BB, EZZ_FB are parts of equations 6 and 8 in BM 2010

        # E[f_t*f_t' | Omega_T]
        EZZ = Zsmooth[b_subset, 1:] @ Zsmooth[b_subset, 1:].T + np.nansum(
            Vsmooth[b_subset, b_subset, 1:], axis=2
        )

        # E[f_{t-1}*f_{t-1}' | Omega_T]
        EZZ_BB = Zsmooth[b_subset, :-1] @ Zsmooth[b_subset, :-1].T + np.nansum(
            Vsmooth[b_subset, b_subset, :-1], axis=2
        )

        # E[f_t*f_{t-1}' | Omega_T]
        EZZ_FB = Zsmooth[b_subset, 1:] @ Zsmooth[b_subset, :-1].T + np.nansum(
            VVsmooth[b_subset, b_subset, :], axis=2
        )

        # Select transition matrix/covariance matrix for block i
        A_i = A[t_start:t_end, t_start:t_end]
        Q_i = Q[t_start:t_end, t_start:t_end]

        # Equation 6: Estimate VAR(p) for factor
        A_i[:r_i, :rp] = EZZ_FB[:r_i, :rp] @ safe_inv(EZZ_BB[:rp, :rp])

        # Equation 8: Covariance matrix of residuals of VAR
        Q_i[:r_i, :r_i] = (EZZ[:r_i, :r_i] - A_i[:r_i, :rp] @ EZZ_FB[:r_i, :rp].T) / T

        # Place updated results in output matrix
        A_new[t_start:t_end, t_start:t_end] = A_i
        Q_new[t_start:t_end, t_start:t_end] = Q_i
        V_0_new[t_start:t_end, t_start:t_end] = Vsmooth[t_start:t_end, t_start:t_end, 0]

    # 2B. UPDATING PARAMETERS FOR IDIOSYNCRATIC COMPONENT ------------------
    rp1 = sum(r) * ppC  # Col size of factor portion
    niM = np.sum(i_idio[:nM])  # Number of monthly values
    t_start = rp1  # Start of idiosyncratic component index
    i_subset = slice(
        t_start, rp1 + niM
    )  # Gives indices for monthly idiosyncratic component values

    # Below 3 estimate the idiosyncratic component (for eqns 6, 8 BM 2010)
    # E[f_t*f_t' | \Omega_T]
    EZZ = np.diag(np.diag(Zsmooth[t_start:, 1:] @ Zsmooth[t_start:, 1:].T)) + np.diag(
        np.diag(np.sum(Vsmooth[t_start:, t_start:, 1:], axis=2))
    )

    # E[f_{t-1}*f_{t-1}' | \Omega_T]
    EZZ_BB = np.diag(
        np.diag(Zsmooth[t_start:, :-1] @ Zsmooth[t_start:, :-1].T)
    ) + np.diag(np.diag(np.sum(Vsmooth[t_start:, t_start:, :-1], axis=2)))

    # E[f_t*f_{t-1}' | \Omega_T]
    EZZ_FB = np.diag(
        np.diag(Zsmooth[t_start:, 1:] @ Zsmooth[t_start:, :-1].T)
    ) + np.diag(np.diag(np.sum(VVsmooth[t_start:, t_start:, :], axis=2)))

    A_i = EZZ_FB @ np.diag(1.0 / np.diag(EZZ_BB))  # Equation 6
    Q_i = (EZZ - A_i @ EZZ_FB.T) / T  # Equation 8

    # Place updated results in output matrix
    A_new[i_subset, i_subset] = A_i[:niM, :niM]
    Q_new[i_subset, i_subset] = Q_i[:niM, :niM]
    V_0_new[i_subset, i_subset] = np.diag(np.diag(Vsmooth[i_subset, i_subset, 0]))

    #  3 MAXIMIZATION STEP (observation equation)

    # INITIALIZATION AND SETUP ----------------------------------------------
    Z_0 = Zsmooth[:, 0]  # zeros(size(Zsmooth,1),1);

    # Set missing data series values to 0
    nanY = np.isnan(y)
    y[nanY] = 0

    # LOADINGS
    C_new = C.copy()

    # Blocks
    bl = np.unique(blocks, axis=0)  # Gives unique loadings
    n_bl = bl.shape[0]  # Number of unique loadings

    # Initialize indices: These later help with subsetting
    bl_idxM = np.empty((n_bl, 0), dtype=bool)  # Indicator for monthly factor loadings
    bl_idxQ = np.empty((n_bl, 0), dtype=bool)  # Indicator for quarterly factor loadings
    R_con = None  # Block diagonal matrix giving monthly-quarterly aggreg scheme
    q_con = np.empty((0, 1))  # Empty column vector

    # Loop through each block
    for i in range(num_blocks):
        # Create the repeated block matrices
        repeated_bl_Q = np.tile(bl[:, i : i + 1].reshape(-1, 1), (1, r[i] * ppC))
        bl_idxQ = np.hstack([bl_idxQ, repeated_bl_Q]) if bl_idxQ.size else repeated_bl_Q

        matrices_to_concat = [
            bl_idxM,
            np.tile(bl[:, i : i + 1].reshape(-1, 1), (1, r[i])),
            np.zeros((n_bl, r[i] * (ppC - 1))),
        ]  # Monthly
        matrices_to_concat = [matrix for matrix in matrices_to_concat if matrix.size]

        # Update bl_idxQ and bl_idxM by concatenating horizontally
        bl_idxM = np.hstack(matrices_to_concat)

        # Construct block diagonal matrix R_con
        block_kron = kron(R_mat, np.eye(r[i]))
        R_con = block_diag(R_con, block_kron) if R_con is not None else block_kron

        # Append zeros to q_con
        zeros_to_append = np.zeros((r[i] * R_mat.shape[0], 1))
        q_con = np.vstack([q_con, zeros_to_append])

    # Indicator for monthly/quarterly blocks in observation matrix
    bl_idxM = np.array(bl_idxM, dtype=bool)
    bl_idxQ = np.array(bl_idxQ, dtype=bool)

    i_idio_M = i_idio[:nM]  # Gives 1 for monthly series
    n_idio_M = np.count_nonzero(i_idio_M)  # Number of monthly series
    c_i_idio = np.cumsum(i_idio)  # Cumulative number of monthly series

    for i in range(n_bl):  # Loop through unique loadings (e.g. [1 0 0 0], [1 1 0 0])
        bl_i = bl[i, :]
        rs = sum(r[bl_i.astype(bool)])  # Total num of blocks loaded

        idx_i = np.where((blocks == bl_i).all(axis=1))[0]  # Indices for bl_i
        idx_iM = idx_i[idx_i < nM]  # Only monthly
        n_i = len(idx_iM)  # Number of monthly series

        # Initialize sums in equation 13 of BGR 2010
        denom = np.zeros((n_i * rs, n_i * rs))
        nom = np.zeros((n_i, rs))

        # Stores monthly indicies. These are done for input robustness
        i_idio_i = i_idio_M[idx_iM]
        i_idio_ii = c_i_idio[idx_iM]
        i_idio_ii = i_idio_ii[i_idio_i.flatten()]

        # UPDATE MONTHLY VARIABLES: Loop through each period ----------------
        for t in range(T):
            Wt = np.diag(
                ~nanY[idx_iM, t]
            )  # Gives selection matrix (1 for nonmissing values)

            denom = denom + kron(  # E[f_t*t_t' | Omega_T]
                (
                    Zsmooth[np.where(bl_idxM[i, :]), t + 1].reshape(-1, 1)
                    @ Zsmooth[np.where(bl_idxM[i, :]), t + 1].reshape(-1, 1).T
                    + Vsmooth[
                        np.where(bl_idxM[i, :]), np.where(bl_idxM[i, :]), t + 1
                    ].reshape(-1, 1)
                ),
                Wt,
            )

            # denom = denom +...
            #         kron(Zsmooth(bl_idxM(i, :), t+1) * Zsmooth(bl_idxM(i, :), t+1)' + ...
            #         Vsmooth(bl_idxM(i, :), bl_idxM(i, :), t+1), Wt);
            nom = (
                nom
                + y[idx_iM, t][:, np.newaxis]  # E[y_t*f_t' | \Omega_T]
                @ Zsmooth[np.where(bl_idxM[i, :]), t + 1].reshape(-1, 1).T
                - Wt[:, np.where(i_idio_i)[0]]
                @ (
                    Zsmooth[rp1 + i_idio_ii, t + 1].reshape(-1, 1)
                    @ Zsmooth[np.where(bl_idxM[i, :]), t + 1].reshape(-1, 1).T
                    + Vsmooth[rp1 + i_idio_ii, :, :][
                        :, np.where(bl_idxM[i, :])[0], t + 1
                    ]
                )
            )

            # nom = nom + ...  E[y_t*f_t' | \Omega_T]
            #       y(idx_iM, t) * Zsmooth(bl_idxM(i, :), t+1)' - ...
            #       Wt(:, i_idio_i) * (Zsmooth(rp1 + i_idio_ii, t+1) * ...
            #       Zsmooth(bl_idxM(i, :), t+1)' + ...
            #       Vsmooth(rp1 + i_idio_ii, bl_idxM(i, :), t+1));

        vec_C = safe_inv(denom) @ nom.flatten().reshape(-1, 1)  # Eqn 13 BGR 2010

        # Place updated monthly results in output matrix
        C_new[idx_iM[:, None], np.where(bl_idxM[i, :])] = vec_C.reshape(n_i, rs)

        # UPDATE QUARTERLY VARIABLES -----------------------------------------

        idx_iQ = idx_i[idx_i >= nM]  # Index for quarterly series
        rps = rs * ppC

        # Monthly-quarterly aggregation scheme
        R_con_i = R_con[:, bl_idxQ[i, :]]
        q_con_i = q_con

        no_c = ~np.any(R_con_i, axis=1)
        R_con_i = R_con_i[~no_c, :]  # R_con_i(no_c,:) = [];
        q_con_i = q_con_i[~no_c, :]  # q_con_i(no_c,:) = [];

        # Loop through quarterly series in loading. This parallels monthly code
        for j in idx_iQ:
            # Initialization
            denom = np.zeros((rps, rps))
            nom = np.zeros((1, rps))

            idx_jQ = j - nM  # Ordinal position of quarterly variable
            # Loc of factor structure corresponding to quarterly var residuals
            # i_idio_jQ = (rp1 + n_idio_M + 5*(idx_jQ-1)+1:rp1+ n_idio_M + 5*idx_jQ);
            i_idio_jQ = np.arange(
                rp1 + n_idio_M + 5 * (idx_jQ - 1), rp1 + n_idio_M + 5 * idx_jQ
            )

            # Place quarterly values in output matrix
            V_0_new[i_idio_jQ[:, None], i_idio_jQ] = Vsmooth[
                i_idio_jQ[:, None], i_idio_jQ, 0
            ]
            A_new[i_idio_jQ[0], i_idio_jQ[0]] = A_i[
                i_idio_jQ[0] - rp1, i_idio_jQ[0] - rp1
            ]
            Q_new[i_idio_jQ[0], i_idio_jQ[0]] = Q_i[
                i_idio_jQ[0] - rp1, i_idio_jQ[0] - rp1
            ]
            for t in range(T):
                Wt = np.diag([~nanY[j, t]])  # Selection matrix for quarterly values

                # Intermediate steps in BGR equation 13
                denom = denom + kron(
                    Zsmooth[np.where(bl_idxQ[i, :]), t + 1].reshape(-1, 1)
                    @ Zsmooth[np.where(bl_idxQ[i, :]), t + 1].reshape(-1, 1).T
                    + Vsmooth[
                        np.where(bl_idxQ[i, :]), np.where(bl_idxQ[i, :]), t + 1
                    ].reshape(-1, 1),
                    Wt,
                )
                # denom = denom + ...
                #         kron(Zsmooth(bl_idxQ(i,:), t+1) * Zsmooth(bl_idxQ(i,:), t+1)'...
                #         + Vsmooth(bl_idxQ(i,:), bl_idxQ(i,:), t+1), Wt);

                nom = (
                    nom
                    + y[j, t] * Zsmooth[np.where(bl_idxQ[i, :]), t + 1].reshape(-1, 1).T
                )
                # nom = nom + y(j, t)*Zsmooth(bl_idxQ(i,:), t+1)';

                nom = nom - Wt * (
                    [1, 2, 3, 2, 1]
                    @ Zsmooth[i_idio_jQ, t + 1].reshape(-1, 1)
                    @ Zsmooth[np.where(bl_idxQ[i, :]), t + 1].reshape(-1, 1).T
                    + np.array([1, 2, 3, 2, 1])
                    @ Vsmooth[i_idio_jQ[:, None], np.where(bl_idxQ[i, :]), t + 1]
                )
                # nom = nom -...
                #         Wt * ([1 2 3 2 1] * Zsmooth(i_idio_jQ,t+1) * ...
                #         Zsmooth(bl_idxQ(i,:),t+1)'+...
                #         [1 2 3 2 1]*Vsmooth(i_idio_jQ,bl_idxQ(i,:),t+1));

            C_i = safe_inv(denom) @ nom.T

            C_i_constr = C_i - safe_inv(  # BGR equation 13
                denom
            ) @ R_con_i.T @ safe_inv(R_con_i @ safe_inv(denom) @ R_con_i.T) @ (
                R_con_i @ C_i - q_con_i
            )

            # Place updated values in output structure
            C_new[j, np.where(bl_idxQ[i, :])] = C_i_constr.reshape(1, -1)
            # C_new(j,bl_idxQ(i,:)) = C_i_constr;

    # 3B. UPDATE COVARIANCE OF RESIDUALS FOR OBSERVATION EQUATION -----------
    # Initialize covariance of residuals of observation equation
    R_new = np.zeros((n, n))
    for t in range(T):
        Wt = np.diag(~nanY[:, t])  # Selection matrix

        R_new = (
            R_new
            + (y[:, t][:, np.newaxis] - Wt @ C_new @ Zsmooth[:, t + 1][:, np.newaxis])
            @ (y[:, t][:, np.newaxis] - Wt @ C_new @ Zsmooth[:, t + 1][:, np.newaxis]).T
            + Wt @ C_new @ Vsmooth[:, :, t + 1] @ C_new.T @ Wt
            + (np.eye(n) - Wt) @ R @ (np.eye(n) - Wt)
        )

        # R_new = R_new + (y(:,t) - ...  % BGR equation 15
        #         Wt * C_new * Zsmooth(:, t+1)) * (y(:,t) - Wt*C_new*Zsmooth(:,t+1))'...
        #       + Wt*C_new*Vsmooth(:,:,t+1)*C_new'*Wt + (eye(n)-Wt)*R*(eye(n)-Wt);

    R_new = R_new / T
    RR = np.diag(R_new).copy()  # RR(RR<1e-2) = 1e-2;
    RR[
        np.where(i_idio_M.flatten())
    ] = 1e-04  # Ensure non-zero measurement error. See Doz, Giannone, Reichlin (2012) for reference.
    RR[nM:] = 1e-04
    R_new = np.diag(RR)

    return C_new, R_new, A_new, Q_new, Z_0, V_0_new, loglik


# def em_converged(loglik, previous_loglik, threshold=1e-4, check_decreased=True):
#     """
#     Checks whether EM has converged. Convergence occurs if
#     the slope of the log-likelihood function falls below 'threshold'(i.e.
#     f(t) - f(t-1)| / avg < threshold) where avg = (|f(t)| + |f(t-1)|)/2
#     and f(t) is log lik at iteration t. 'threshold' defaults to 1e-4.

#     This stopping criterion is from Numerical Recipes in C (pg. 423).
#     With MAP estimation (using priors), the likelihood can decrease
#     even if the mode of the posterior increases.

#     Input arguments:
#       loglik: Log-likelihood from current EM iteration
#       previous_loglik: Log-likelihood from previous EM iteration
#       threshold: Convergence threshhold. The default is 1e-4.
#       check_decreased: Returns text output if log-likelihood decreases.

#     Output:
#       converged (numeric): Returns 1 if convergence criteria satisfied, and 0 otherwise.
#       decrease (numeric): Returns 1 if loglikelihood decreased.


#     """

#     # Initialize output
#     converged = 0
#     decrease = 0

#     # Check if log-likelihood decreases (optional)
#     if check_decreased:
#         if loglik - previous_loglik < -1e-3:  # allow for a little imprecision
#             print(f'******likelihood decreased from {previous_loglik:.4f} to {loglik:.4f}!')
#             decrease = 1

#     # Check convergence criteria

#     delta_loglik = abs(loglik - previous_loglik)  # Difference in loglik
#     avg_loglik = (abs(loglik) + abs(previous_loglik) + np.finfo(float).eps) / 2  # Avoid division by zero

#     if (delta_loglik / avg_loglik) < threshold:
#         converged = 1  # Check convergence

#     return converged, decrease


def em_converged(loglik, previous_loglik, threshold=1e-4, check_decreased=True):
    """
    Checks whether EM has converged. Convergence occurs if
    the slope of the log-likelihood function falls below 'threshold'(i.e.
    f(t) - f(t-1)| / avg < threshold) where avg = (|f(t)| + |f(t-1)|)/2
    and f(t) is log lik at iteration t. 'threshold' defaults to 1e-4.

    This stopping criterion is from Numerical Recipes in C (pg. 423).
    With MAP estimation (using priors), the likelihood can decrease
    even if the mode of the posterior increases.

    Input arguments:
      loglik: Log-likelihood from current EM iteration
      previous_loglik: Log-likelihood from previous EM iteration
      threshold: Convergence threshold. The default is 1e-4.
      check_decreased: Returns text output if log-likelihood decreases.

    Output:
      converged (numeric): Returns 1 if convergence criteria satisfied, and 0 otherwise.
      decrease (numeric): Returns 1 if loglikelihood decreased.
    """

    # Initialize output
    converged = 0
    decrease = 0

    # Check if log-likelihood decreases (optional)
    if check_decreased:
        if loglik - previous_loglik < -1e-3:  # allow for a little imprecision
            print(
                f"******likelihood decreased from {previous_loglik:.4f} to {loglik:.4f}!"
            )
            decrease = 1

    # Calculate difference in log-likelihood and average log-likelihood
    delta_loglik = abs(loglik - previous_loglik)
    avg_loglik = (
        abs(loglik) + abs(previous_loglik) + np.finfo(float).eps
    ) / 2  # Add epsilon to avoid division by zero

    # Check for small avg_loglik to prevent invalid division
    if avg_loglik < np.finfo(float).eps or np.isinf(avg_loglik):
        # print('******avg_loglik is too small, potential division by zero!')
        return converged, decrease

    # Check convergence criteria
    if (delta_loglik / avg_loglik) < threshold:
        converged = 1

    return converged, decrease


def InitCond(x, r, p, blocks, optNaN, Rcon, q, nQ, i_idio):
    """
    Calculates initial conditions for parameter estimation.

    Given standardized data and model information, this function creates
    initial parameter estimates. These are the initial inputs for the EM
    algorithm, which re-estimates these parameters using Kalman filtering
    techniques.

    Args:
        x (ndarray): Standardized data.
        r (int): Number of common factors for each block.
        p (int): Number of lags in the transition equation.
        blocks (ndarray): Series loadings.
        optNaN (bool): Option for handling missing values in spline.
                       See `remNaNs_spline()` for details.
        Rcon (ndarray): Matrix incorporating estimation for quarterly series
                        (i.e., "tent structure").
        q (ndarray): Constraints on loadings for quarterly variables.
        NQ (int): Number of quarterly variables.
        i_idio (ndarray): Logical array indicating indices for monthly (1)
                          and quarterly (0) variables.

    Returns:
        A (ndarray): Transition matrix.
        C (ndarray): Observation matrix.
        Q (ndarray): Covariance matrix for transition equation residuals.
        R (ndarray): Covariance matrix for observation equation residuals.
        Z_0 (ndarray): Initial value of the state.
        V_0 (ndarray): Initial value of the covariance matrix.
    """

    pC = Rcon.shape[1]  # Gives 'tent' structure size (quarterly to monthly)
    ppC = max(p, pC)
    n_b = blocks.shape[1]  # Number of blocks

    xBal, indNaN = remNaNs_spline(x, optNaN)
    # Spline without NaNs

    T, N = xBal.shape  # Time T series number N
    nM = N - nQ  # Number of monthly series

    xNaN = xBal.copy()
    xNaN[indNaN] = np.nan  # Set missing values equal to NaNs
    res = xBal.copy()  # Spline output equal to res: Later this is used for residuals
    resNaN = xNaN.copy()  # Later used for residuals

    # Initialize model coefficient output
    C = np.empty((0, 0), dtype=float)
    A = np.empty((0, 0), dtype=float)
    Q = np.empty((0, 0), dtype=float)
    V_0 = np.empty((0, 0), dtype=float)

    # Set the first observations as NaNs: For quarterly-monthly aggreg. scheme
    indNaN[: pC - 1, :] = True

    for i in range(n_b):  # Loop for each block
        r_i = r[i]  # r_i = 1 when block is loaded

        # Observation equation ------------------------------------------------
        C_i = np.zeros(
            (N, r_i * ppC), dtype=float
        )  # Initialize state variable matrix helper
        idx_i = np.where(blocks[:, i])[0]  # Returns series index loading block i
        idx_iM = idx_i[idx_i < nM + 1]  # Monthly series indices for loaded blocks
        idx_iQ = idx_i[idx_i >= nM]  # Quarterly series indices for loaded blocks

        values, vectors = eigsh(
            np.cov(res[:, idx_iM], rowvar=False, dtype=float), k=r_i, which="LM"
        )  # issue related with complex numbers: https://stackoverflow.com/questions/8765310/scipy-linalg-eig-return-complex-eigenvalues-for-covariance-matrix?rq=4
        v = vectors[:, 0].reshape(-1, 1)
        # Flip sign for cleaner output. Gives equivalent results without this section
        if np.sum(v) < 0:
            v = -v

        # For monthly series with loaded blocks (rows), replace with eigenvector
        # This gives the loading
        C_i[idx_iM, :r_i] = v
        f = res[:, idx_iM] @ v  # Data projection for eigenvector direction
        F = []

        # Lag matrix using loading. This is later used for quarterly series
        # max_lags = max(p + 1, pC) - 1
        for kk in range(max(p + 1, pC)):
            F.append(f[pC - kk - 1 : f.shape[0] - kk, :])
        F = np.hstack(F)

        Rcon_i = np.kron(Rcon, np.eye(r_i))
        q_i = np.kron(q, np.zeros((r_i, 1)))

        ff = F[:, : r_i * pC]
        for j in idx_iQ:  # Loop for quarterly variables
            # For series j, values are dropped to accommodate lag structure
            xx_j = resNaN[pC - 1 :, j]

            if np.sum(~np.isnan(xx_j)) < ff.shape[1] + 2:
                xx_j = res[pC - 1 :, j]  # Replaces xx_j with spline if too many NaNs

            ff_j = ff[~np.isnan(xx_j), :]
            xx_j = xx_j[~np.isnan(xx_j)]

            iff_j = safe_inv(ff_j.T @ ff_j)
            Cc = iff_j @ ff_j.T @ xx_j.reshape(-1, 1)  # Least squares

            # Spline data monthly to quarterly conversion
            Cc -= (
                iff_j
                @ Rcon_i.T
                @ safe_inv(Rcon_i @ iff_j @ Rcon_i.T)
                @ (Rcon_i @ Cc - q_i)
            )

            C_i[j, : pC * r_i] = Cc.T  # Place in output matrix

        ff = np.vstack((np.zeros((pC - 1, pC * r_i)), ff))

        # Residual calculations
        res = res - ff @ C_i.T
        resNaN = res.copy()
        resNaN[indNaN] = np.nan

        C = np.hstack((C, C_i)) if C.size else C_i  # Combine past loadings together
        # Transition equation ------------------------------------------------

        z = F[:, :r_i]  # Projected data (no lag)
        Z = F[:, r_i : r_i * (p + 1)]  # Data with lag 1

        A_i = np.zeros((r_i * ppC, r_i * ppC))  # Initialize transition matrix

        A_temp = (
            safe_inv(Z.T @ Z) @ Z.T @ z
        )  # OLS: gives coefficient value AR(p) process
        A_i[:r_i, : r_i * p] = A_temp.T
        A_i[r_i:, : r_i * (ppC - 1)] = np.eye(r_i * (ppC - 1))

        ##################################

        Q_i = np.zeros((ppC * r_i, ppC * r_i), dtype=float)
        e = z - Z @ A_temp  # VAR residuals
        Q_i[:r_i, :r_i] = np.cov(e, rowvar=False)  # VAR covariance matrix

        initV_i = np.linalg.solve(
            np.eye((r_i * ppC) ** 2) - np.kron(A_i, A_i), Q_i.ravel()
        ).reshape(r_i * ppC, r_i * ppC)

        # Gives top left block for the transition matrix
        A = block_diag(A, A_i)
        Q = block_diag(Q, Q_i)
        V_0 = block_diag(V_0, initV_i)

    eyeN = np.eye(N)  # Used inside observation matrix
    eyeN = eyeN[:, np.hstack(i_idio)]

    C = np.hstack((C, eyeN))
    C = np.hstack(
        (C, np.vstack((np.zeros((nM, 5 * nQ)), np.kron(np.eye(nQ), [1, 2, 3, 2, 1]))))
    )  # Monthly-quarterly agreggation scheme
    R = np.diag(
        np.nanvar(resNaN, axis=0, ddof=1)
    )  # Initialize covariance matrix for transition matrix

    ii_idio = np.where(i_idio)[0]  # Indicies for monthly variables
    n_idio = len(ii_idio)  # Number of monthly variables
    BM = np.zeros(
        (n_idio, n_idio), dtype=float
    )  # Initialize monthly transition matrix values
    SM = np.zeros(
        (n_idio, n_idio), dtype=float
    )  # Initialize monthly residual covariance matrix values

    for i in range(n_idio):  # Loop for monthly variables
        # Set observation equation residual covariance matrix diagonal
        R[ii_idio[i], ii_idio[i]] = 1e-4

        # Subsetting series residuals for series i
        res_i = resNaN[:, ii_idio[i]]

        # Returns number of leading/ending zeros
        leadZero = np.argmax(np.cumsum(np.isnan(res_i)) == np.arange(1, T + 1))
        endZero = np.argmax(np.cumsum(np.isnan(res_i[::-1])) == np.arange(1, T + 1))

        # Truncate leading and ending zeros
        res_i = res[:, ii_idio[i]]
        res_i = res_i[leadZero : (T - endZero)]
        res_i = res_i.reshape(-1, 1)

        # Linear regression: AR 1 process for monthly series residuals
        BM[i, i] = (
            safe_inv(res_i[:-1].T @ res_i[:-1]) @ res_i[:-1].T @ res_i[1:]
        ).item()
        SM[i, i] = np.cov(
            (res_i[1:] - res_i[:-1] * BM[i, i]).flatten()
        ).item()  # Residual covariance matrix

    Rdiag = np.copy(np.diag(R))
    sig_e = Rdiag[nM:N] / 19
    Rdiag[nM:N] = 1e-4
    R = np.diag(Rdiag)  # Covariance for obs matrix residuals

    # For BQ, SQ
    rho0 = 0.1
    temp = np.zeros((5, 5))
    temp[0, 0] = 1

    # Blocks for covariance matrices
    SQ = np.kron(np.diag((1 - rho0**2) * sig_e), temp)
    BQ = np.kron(
        np.eye(nQ),
        np.vstack(([rho0, 0, 0, 0, 0], np.hstack((np.eye(4), np.zeros((4, 1)))))),
    )

    # initViQ = reshape(inv(eye((5*nQ)^2)-kron(BQ,BQ))*SQ(:),5*nQ,5*nQ);
    # initViM = diag(1./diag(eye(size(BM,1))-BM.^2)).*SM;
    initViQ = np.linalg.solve(
        np.eye((5 * nQ) ** 2) - np.kron(BQ, BQ), SQ.ravel()
    ).reshape(5 * nQ, 5 * nQ)
    initViM = np.diag(1.0 / np.diag(np.eye(BM.shape[0]) - BM**2)) * SM

    # Output
    A = block_diag(A, BM, BQ)  # Observation matrix
    Q = block_diag(Q, SM, SQ)  # Residual covariance matrix (transition)
    Z_0 = np.zeros((A.shape[0], 1))  # States
    V_0 = block_diag(V_0, initViM, initViQ)  # Covariance of states

    return A, C, Q, R, Z_0, V_0


def runKF(Y, A, C, Q, R, Z_0, V_0):
    """ "
    Applies Kalman filter and fixed-interval smoother

     Description:
       runKF() applies a Kalman filter and fixed-interval smoother. The
       script uses the following model:
              Y_t = C_t Z_t + e_t for e_t ~ N(0, R)
              Z_t = A Z_{t-1} + mu_t for mu_t ~ N(0, Q)

     Throughout this file:
       'm' denotes the number of elements in the state vector Z_t.
       'k' denotes the number of elements (observed variables) in Y_t.
       'nobs' denotes the number of time periods for which data are observed.

     Input parameters:
       Y: k-by-nobs matrix of input data
       A: m-by-m transition matrix
       C: k-by-m observation matrix
       Q: m-by-m covariance matrix for transition equation residuals (mu_t)
       R: k-by-k covariance for observation matrix residuals (e_t)
       Z_0: 1-by-m vector, initial value of state
       V_0: m-by-m matrix, initial value of state covariance matrix

     Output parameters:
       zsmooth: k-by-(nobs+1) matrix, smoothed factor estimates
                (i.e. zsmooth(:,t+1) = Z_t|T)
       Vsmooth: k-by-k-by-(nobs+1) array, smoothed factor covariance matrices
                (i.e. Vsmooth(:,:,t+1) = Cov(Z_t|T))
       VVsmooth: k-by-k-by-nobs array, lag 1 factor covariance matrices
                 (i.e. Cov(Z_t,Z_t-1|T))
       loglik: scalar, log-likelihood

      References:
     - QuantEcon's "A First Look at the Kalman Filter"
     - Adapted from replication files for:
       "Nowcasting", 2010, (by Marta Banbura, Domenico Giannone and Lucrezia
       Reichlin), in Michael P. Clements and David F. Hendry, editors, Oxford
       Handbook on Economic Forecasting.

    The software can be freely used in applications.
    Users are kindly requested to add acknowledgements to published work and
    to cite the above reference in any resulting publications
    """

    S = SKF(Y, A, C, Q, R, Z_0, V_0)  # Kalman filter
    S = FIS(A, S)  # Fixed interval smoother

    k = Y.shape[0]

    # Organize output
    zsmooth = S["ZmT"]
    Vsmooth = S["VmT"]
    VVsmooth = S["VmT_1"]
    loglik = S["loglik"].item() if isinstance(S["loglik"], np.ndarray) else S["loglik"]

    return zsmooth, Vsmooth, VVsmooth, loglik


# def SKF(Y, A, C, Q, R, Z_0, V_0):
#     """
#     Applies Kalman filter

#       Syntax:
#         S = SKF(Y, A, C, Q, R, Z_0, V_0)

#       Description:
#         SKF() applies the Kalman filter

#       Input parameters:
#         Y: k-by-nobs matrix of input data
#         A: m-by-m transition matrix
#         C: k-by-m observation matrix
#         Q: m-by-m covariance matrix for transition equation residuals (mu_t)
#         R: k-by-k covariance for observation matrix residuals (e_t)
#         Z_0: 1-by-m vector, initial value of state
#         V_0: m-by-m matrix, initial value of state covariance matrix

#       Output parameters:
#         S.Zm: m-by-nobs matrix, prior/predicted factor state vector
#               (S.Zm(:,t) = Z_t|t-1)
#         S.ZmU: m-by-(nobs+1) matrix, posterior/updated state vector
#                (S.Zm(t+1) = Z_t|t)
#         S.Vm: m-by-m-by-nobs array, prior/predicted covariance of factor
#               state vector (S.Vm(:,:,t) = V_t|t-1)
#         S.VmU: m-by-m-by-(nobs+1) array, posterior/updated covariance of
#                factor state vector (S.VmU(:,:,t+1) = V_t|t)
#         S.loglik: scalar, value of likelihood function
#         S.k_t: k-by-m Kalman gain
#     """
#     # INITIALIZE OUTPUT VALUES ---------------------------------------------
#     # Output structure & dimensions of state space matrix
#     _, m = C.shape
#     # Outputs time for data matrix. "number of observations"
#     nobs  = Y.shape[1]

#     # Instantiate output
#     S = {
#         'Zm': np.full((m, nobs), np.nan, dtype=float),       # Z_t | t-1 (prior)
#         'Vm': np.full((m, m, nobs), np.nan, dtype=float),    # V_t | t-1 (prior)
#         'ZmU': np.full((m, nobs + 1), np.nan, dtype=float),  # Z_t | t (posterior/updated)
#         'VmU': np.full((m, m, nobs + 1), np.nan, dtype=float), # V_t | t (posterior/updated)
#         'loglik': 0,
#         'k_t': np.zeros((Y.shape[0], m))
#     }

#     # SET INITIAL VALUES ----------------------------------------------------
#     Zu = Z_0  # Z_0|0 (In below loop, Zu gives Z_t | t)
#     Vu = V_0  # V_0|0 (In below loop, Vu guvse V_t | t)

#     #   % Store initial values
#     S['ZmU'][:, 0] = Zu.flatten()
#     S['VmU'][:, :, 0] = Vu

#     epsilon = 1e-10  # Small regularization term

#     # KALMAN FILTER PROCEDURE ----------------------------------------------
#     for t in range(nobs):
#         # CALCULATING PRIOR DISTIBUTION----------------------------------

#         # Use transition eqn to create prior estimate for factor
#         # i.e. Z = Z_t|t-1
#         Z = A @ Zu
#         # Prior covariance matrix of Z (i.e. V = V_t|t-1)
#         #   Var(Z) = Var(A*Z + u_t) = Var(A*Z) + Var(\epsilon) =
#         #   A*Vu*A' + Q
#         V = A @ Vu @ A.T + Q

#         V = 0.5 * (V + V.T)  # Trick to make symmetric

#         # CALCULATING POSTERIOR DISTRIBUTION ----------------------------

#         # Removes missing series: These are removed from Y, C, and R
#         Y_t, C_t, R_t, _ = MissData(Y[:, t], C, R)
#         Y_t = Y_t.reshape(-1, 1)

#         # Check if y_t contains no data. If so, replace Zu and Vu with prior.
#         if Y_t.size == 0:
#             # If Y_t contains no data, replace Zu and Vu with prior
#             Zu = Z
#             Vu = V
#         else:
#             # Steps for variance and population regression coefficients:
#             # Var(c_t*Z_t + e_t) = c_t Var(A) c_t' + Var(u) = c_t*V *c_t' + R

#             VC = V @ C_t.T

#             iF = safe_inv(C_t @ VC + R_t)

#             # Matrix of population regression coefficients (QuantEcon eqn #4)
#             VCF = VC @ iF

#             # Gives difference between actual and predicted observation
#             # matrix values
#             innov = Y_t - (C_t @ Z).reshape(-1, 1)
#             # Update estimate of factor values (posterior)
#             Zu = Z.reshape(-1,1) + VCF @ innov

#             # Update covariance matrix (posterior) for time t
#             Vu = V - VCF @ VC.T
#             Vu = 0.5 * (Vu + Vu.T)  # Approximation trick to make symmetric

#             # Update log likelihood
#             det = np.linalg.det(iF)  # A short-hand for nth root of x: exp(log(x)/n) where x - number, n - degree

#             logdet = -np.inf if det == 0 else cmath.log(det)

#             S["loglik"] = S["loglik"] + 0.5 * (logdet - innov.T @ iF @ innov)

#         # STORE OUTPUT----------------------------------------------------

#         # Store covariance and observation values for t-1 (priors)
#         S['Zm'][:, t] = Z.flatten()
#         S['Vm'][:, :, t] = V

#         # Store covariance and state values for t (posteriors)
#         # i.e. Zu = Z_t|t   & Vu = V_t|t
#         S['ZmU'][:, t + 1] = Zu.flatten()
#         S['VmU'][:, :, t + 1] = Vu

#     # Store Kalman gain k_t
#     if Y_t.size == 0:
#         S['k_t'] = np.zeros((m, m))

#     S['k_t'] = VCF @ C_t

#     return S


def SKF(Y, A, C, Q, R, Z_0, V_0):
    """
    Applies Kalman filter

      Syntax:
        S = SKF(Y, A, C, Q, R, Z_0, V_0)

      Description:
        SKF() applies the Kalman filter

      Input parameters:
        Y: k-by-nobs matrix of input data
        A: m-by-m transition matrix
        C: k-by-m observation matrix
        Q: m-by-m covariance matrix for transition equation residuals (mu_t)
        R: k-by-k covariance for observation matrix residuals (e_t)
        Z_0: 1-by-m vector, initial value of state
        V_0: m-by-m matrix, initial value of state covariance matrix

      Output parameters:
        S.Zm: m-by-nobs matrix, prior/predicted factor state vector
              (S.Zm(:,t) = Z_t|t-1)
        S.ZmU: m-by-(nobs+1) matrix, posterior/updated state vector
               (S.Zm(t+1) = Z_t|t)
        S.Vm: m-by-m-by-nobs array, prior/predicted covariance of factor
              state vector (S.Vm(:,:,t) = V_t|t-1)
        S.VmU: m-by-m-by-(nobs+1) array, posterior/updated covariance of
               factor state vector (S.VmU(:,:,t+1) = V_t|t)
        S.loglik: scalar, value of likelihood function
        S.k_t: k-by-m Kalman gain
    """
    # INITIALIZE OUTPUT VALUES ---------------------------------------------
    # Output structure & dimensions of state space matrix
    _, m = C.shape
    # Outputs time for data matrix. "number of observations"
    nobs = Y.shape[1]

    # Instantiate output
    S = {
        "Zm": np.full((m, nobs), np.nan, dtype=float),  # Z_t | t-1 (prior)
        "Vm": np.full((m, m, nobs), np.nan, dtype=float),  # V_t | t-1 (prior)
        "ZmU": np.full(
            (m, nobs + 1), np.nan, dtype=float
        ),  # Z_t | t (posterior/updated)
        "VmU": np.full(
            (m, m, nobs + 1), np.nan, dtype=float
        ),  # V_t | t (posterior/updated)
        "loglik": 0,
        "k_t": np.zeros((Y.shape[0], m)),
    }

    # SET INITIAL VALUES ----------------------------------------------------
    Zu = Z_0  # Z_0|0 (In below loop, Zu gives Z_t | t)
    Vu = V_0  # V_0|0 (In below loop, Vu guvse V_t | t)

    #   % Store initial values
    S["ZmU"][:, 0] = Zu.flatten()
    S["VmU"][:, :, 0] = Vu

    # KALMAN FILTER PROCEDURE ----------------------------------------------
    for t in range(nobs):
        # CALCULATING PRIOR DISTIBUTION----------------------------------

        # Use transition eqn to create prior estimate for factor
        # i.e. Z = Z_t|t-1
        Z = A @ Zu
        # Prior covariance matrix of Z (i.e. V = V_t|t-1)
        #   Var(Z) = Var(A*Z + u_t) = Var(A*Z) + Var(\epsilon) =
        #   A*Vu*A' + Q
        V = A @ Vu @ A.T + Q
        V = 0.5 * (V + V.T)  # Trick to make symmetric

        # CALCULATING POSTERIOR DISTRIBUTION ----------------------------

        # Removes missing series: These are removed from Y, C, and R
        Y_t, C_t, R_t, _ = MissData(Y[:, t], C, R)
        Y_t = Y_t.reshape(-1, 1)

        # Check if y_t contains no data. If so, replace Zu and Vu with prior.
        if Y_t.size == 0:
            # If Y_t contains no data, replace Zu and Vu with prior
            Zu = Z
            Vu = V
        else:
            # Steps for variance and population regression coefficients:
            # Var(c_t*Z_t + e_t) = c_t Var(A) c_t' + Var(u) = c_t*V *c_t' + R

            VC = V @ C_t.T

            iF = safe_inv(C_t @ VC + R_t)

            # Matrix of population regression coefficients (QuantEcon eqn #4)
            VCF = VC @ iF

            # Gives difference between actual and predicted observation
            # matrix values
            innov = Y_t - (C_t @ Z).reshape(-1, 1)
            # Update estimate of factor values (posterior)
            Zu = Z.reshape(-1, 1) + VCF @ innov

            # Update covariance matrix (posterior) for time t
            Vu = V - VCF @ VC.T
            Vu = 0.5 * (Vu + Vu.T)  # Approximation trick to make symmetric

            # Update log likelihood
            det = np.linalg.det(
                iF
            )  # A short-hand for nth root of x: exp(log(x)/n) where x - number, n - degree

            if det == 0:
                # If there are precision related issues, replace Zu and Vu with prior
                Zu = Z
                Vu = V
            else:
                S["loglik"] = S["loglik"] + 0.5 * (
                    cmath.log(det) - innov.T @ iF @ innov
                )

        # STORE OUTPUT----------------------------------------------------

        # Store covariance and observation values for t-1 (priors)
        S["Zm"][:, t] = Z.flatten()
        S["Vm"][:, :, t] = V

        # Store covariance and state values for t (posteriors)
        # i.e. Zu = Z_t|t   & Vu = V_t|t
        S["ZmU"][:, t + 1] = Zu.flatten()
        S["VmU"][:, :, t + 1] = Vu

    # Store Kalman gain k_t
    if Y_t.size == 0:
        S["k_t"] = np.zeros((m, m))

    S["k_t"] = VCF @ C_t

    return S


# def FIS(A, S):
#     """
#     Applies fixed-interval smoother

#     Syntax:
#       S = FIS(A, S)

#     Description:
#       SKF() applies a fixed-interval smoother, and is used in conjunction
#       with SKF(). See  page 154 of 'Forecasting, structural time series models
#       and the Kalman filter' for more details (Harvey, 1990).

#     Input parameters:
#       A: m-by-m transition matrix
#       S: structure returned by SKF()

#     Output parameters:
#       S: FIS() adds the following smoothed estimates to the S structure:
#         - S.ZmT: m-by-(nobs+1) matrix, smoothed states
#             (S.ZmT(:,t+1) = Z_t|T)
#         - S.VmT: m-by-m-by-(nobs+1) array, smoothed factor covariance
#             matrices (S.VmT(:,:,t+1) = V_t|T = Cov(Z_t|T))
#         - S.VmT_1: m-by-m-by-nobs array, smoothed lag 1 factor covariance
#                 matrices (S.VmT_1(:,:,t) = Cov(Z_t Z_t-1|T))

#     Model:
#       Y_t = C_t Z_t + e_t for e_t ~ N(0, R)
#       Z_t = A Z_{t-1} + mu_t for mu_t ~ N(0, Q)
#     """

#     # ORGANIZE INPUT ---------------------------------------------------------

#     # Initialize output matrices
#     m, nobs = S['Zm'].shape
#     S['ZmT'] = np.zeros((m, nobs + 1), dtype=float)
#     S['VmT'] = np.zeros((m, m, nobs + 1), dtype=float)
#     S['VmT_1'] = np.empty((m, m, nobs), dtype=float)
#     S['VmT_1'][:] = np.nan

#     # Fill the final period of ZmT, VmT with SKF() posterior values
#     S['ZmT'][:, nobs] = S['ZmU'][:, nobs]
#     S['VmT'][:, :, nobs] = S['VmU'][:, :, nobs]

#     # Initialize VmT_1 lag 1 covariance matrix for final period
#     S['VmT_1'][:, :, nobs - 1] = (np.eye(m) - S['k_t']) @ A @ S['VmU'][:, :, nobs - 1]

#     # Used for recursion process. See companion file for details
#     J_2 = S['VmU'][:, :, nobs - 1] @ A.T @ safe_inv(S['Vm'][:, :, nobs - 1])

#     # RUN SMOOTHING ALGORITHM ----------------------------------------------

#     # Loop through time reverse-chronologically (starting at final period nobs)
#     for t in range(nobs - 1, -1, -1):
#         # Store posterior and prior factor covariance values
#         VmU = S['VmU'][:, :, t]
#         Vm1 = S['Vm'][:, :, t]

#         # Store previous period smoothed factor covariance and lag-1 covariance
#         V_T = S['VmT'][:, :, t + 1]
#         V_T1 = S['VmT_1'][:, :, t]

#         J_1 = J_2

#         # Update smoothed factor estimate
#         S['ZmT'][:, t] = S['ZmU'][:, t] + J_1 @ (S['ZmT'][:, t + 1] - A @ S['ZmU'][:, t])

#         # Update smoothed factor covariance matrix
#         S['VmT'][:, :, t] = VmU + J_1 @ (V_T - Vm1) @ J_1.T

#         if t > 0:
#             # Update weight
#             J_2 = S['VmU'][:, :, t - 1] @ A.T @ safe_inv(S['Vm'][:, :, t - 1])

#             # Update lag 1 factor covariance matrix
#             S['VmT_1'][:, :, t - 1] = VmU @ J_2.T + J_1 @ (V_T1 - A @ VmU) @ J_2.T
#     return S


def FIS(A, S):
    """
    Applies fixed-interval smoother

    Syntax:
      S = FIS(A, S)

    Description:
      SKF() applies a fixed-interval smoother, and is used in conjunction
      with SKF(). See  page 154 of 'Forecasting, structural time series models
      and the Kalman filter' for more details (Harvey, 1990).

    Input parameters:
      A: m-by-m transition matrix
      S: structure returned by SKF()

    Output parameters:
      S: FIS() adds the following smoothed estimates to the S structure:
        - S.ZmT: m-by-(nobs+1) matrix, smoothed states
            (S.ZmT(:,t+1) = Z_t|T)
        - S.VmT: m-by-m-by-(nobs+1) array, smoothed factor covariance
            matrices (S.VmT(:,:,t+1) = V_t|T = Cov(Z_t|T))
        - S.VmT_1: m-by-m-by-nobs array, smoothed lag 1 factor covariance
                matrices (S.VmT_1(:,:,t) = Cov(Z_t Z_t-1|T))

    Model:
      Y_t = C_t Z_t + e_t for e_t ~ N(0, R)
      Z_t = A Z_{t-1} + mu_t for mu_t ~ N(0, Q)
    """

    # ORGANIZE INPUT ---------------------------------------------------------

    # Initialize output matrices
    m, nobs = S["Zm"].shape
    S["ZmT"] = np.zeros((m, nobs + 1), dtype=float)
    S["VmT"] = np.zeros((m, m, nobs + 1), dtype=float)
    S["VmT_1"] = np.empty((m, m, nobs), dtype=float)
    S["VmT_1"][:] = np.nan

    # Fill the final period of ZmT, VmT with SKF() posterior values
    S["ZmT"][:, nobs] = S["ZmU"][:, nobs]
    S["VmT"][:, :, nobs] = S["VmU"][:, :, nobs]

    # Initialize VmT_1 lag 1 covariance matrix for final period
    S["VmT_1"][:, :, nobs - 1] = (np.eye(m) - S["k_t"]) @ A @ S["VmU"][:, :, nobs - 1]

    # Used for recursion process. See companion file for details
    J_2 = S["VmU"][:, :, nobs - 1] @ A.T @ safe_inv(S["Vm"][:, :, nobs - 1])

    # RUN SMOOTHING ALGORITHM ----------------------------------------------

    # Loop through time reverse-chronologically (starting at final period nobs)
    for t in range(nobs - 1, -1, -1):
        # Store posterior and prior factor covariance values
        VmU = S["VmU"][:, :, t]
        Vm1 = S["Vm"][:, :, t]

        # Store previous period smoothed factor covariance and lag-1 covariance
        V_T = S["VmT"][:, :, t + 1]
        V_T1 = S["VmT_1"][:, :, t]

        J_1 = J_2

        # Update smoothed factor estimate
        S["ZmT"][:, t] = S["ZmU"][:, t] + J_1 @ (
            S["ZmT"][:, t + 1] - A @ S["ZmU"][:, t]
        )

        # Update smoothed factor covariance matrix
        S["VmT"][:, :, t] = VmU + J_1 @ (V_T - Vm1) @ J_1.T

        if t > 0:
            # Update weight
            J_2 = S["VmU"][:, :, t - 1] @ A.T @ safe_inv(S["Vm"][:, :, t - 1])

            # Update lag 1 factor covariance matrix
            S["VmT_1"][:, :, t - 1] = VmU @ J_2.T + J_1 @ (V_T1 - A @ VmU) @ J_2.T

    return S


def MissData(y, C, R):
    """
    Description:
      Eliminates the rows in y & matrices C, R that correspond to missing
      data (NaN) in y

    Input:
      y: Vector of observations at time t
      C: Observation matrix
      R: Covariance for observation matrix residuals

    Output:
      y: Vector of observations at time t (reduced)
      C: Observation matrix (reduced)
      R: Covariance for observation matrix residuals
      L: Used to restore standard dimensions(n x #) where # is the nr of
         available data in y
    """
    # Returns 1 for nonmissing series
    ix = ~np.isnan(y)

    # Index for columns with nonmissing variables
    e = np.eye(y.shape[0])
    L = e[:, ix]

    # Removes missing series
    y = y[ix]

    # Removes missing series from observation matrix
    C = C[ix, :]

    # Removes missing series from transition matrix
    R = R[ix][:, ix]

    return y, C, R, L
