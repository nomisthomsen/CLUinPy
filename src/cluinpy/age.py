import numpy as np
from numba import njit, prange

def calc_age(old_cov: np.ndarray, new_cov: np.ndarray, old_age: np.ndarray) -> np.ndarray:
    """
    Calculates the age based on changes in coverage.

    :param old_cov: The old coverage as a NumPy array.
    :param new_cov: The new coverage as a NumPy array.
    :param old_age: The old age as a NumPy array.
    :return: The new age as a NumPy array.
    """
    same = old_cov == new_cov
    new_age = np.where(same, old_age + 1, 1)
    return new_age



@njit(parallel=True)
def autonomous_change(
        new_land_array: np.ndarray,
        old_land_array: np.ndarray,
        age_array: np.ndarray,
        allow: np.ndarray,
        no_data_value: float = -9999
) -> np.ndarray:
    """
    Apply autonomous (bottom-up) land cover transitions based on age thresholds.

    Parameters:
        new_land_array (np.ndarray):
            The land cover map after demand-driven change allocation (from the main model step).

        old_land_array (np.ndarray):
            The land cover map before allocation (previous state at time t-1).

        age_array (np.ndarray):
            The age of each pixel's current land cover (how long the pixel has remained unchanged).

        allow (np.ndarray):
            A 2D matrix of shape (n_lcov, n_lcov), where:
            - allow[i, j] = 0: transition from i to j is disallowed
            - allow[i, j] = 1: transition is allowed unconditionally
            - allow[i, j] > 1000: represents an autonomous (successional) change allowed
              only when pixel age ≥ (allow[i, j] - 1000)
              → only **one such value per row i** is expected

        no_data_value (float, optional):
            Special value used to indicate no-data pixels. These will be preserved in output.
            Default is -9999.

    Returns:
        np.ndarray:
            Updated land cover map where autonomous changes have been applied
            to unchanged pixels that satisfy age-based succession rules.
    """

    nrow, ncol = new_land_array.shape
    new_lc = np.zeros(new_land_array.shape, dtype=np.int32)

    for i in prange(nrow):
        for j in range(ncol):
            lc = new_land_array[i, j]

            # Preserve no-data values
            if lc == no_data_value:
                new_lc[i, j] = no_data_value

            # Check for pixels that were not changed in the previous step
            elif new_land_array[i, j] == old_land_array[i, j]:
                # Get the highest transition rule (assumes only one autonomous change >1000 per row)
                autonomous_value = np.max(allow[lc])

                # If an autonomous transition is defined
                if autonomous_value > 1000:
                    allow_age = autonomous_value - 1000

                    # Apply the autonomous change if age condition is satisfied
                    if age_array[i, j] > allow_age:
                        new_lc[i, j] = np.argmax(allow[lc])  # transition to the defined target class
                    else:
                        new_lc[i, j] = lc  # not old enough → retain original class
                else:
                    new_lc[i, j] = lc  # no autonomous rule defined → retain original class

            # If pixel has already changed from old → new in demand-driven step
            else:
                new_lc[i, j] = lc  # preserve already changed value

    return new_lc