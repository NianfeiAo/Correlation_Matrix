import numpy as np
from scipy.signal import welch, butter, filtfilt
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit
from filter_band import filter_band


def design_butterworth_filter(sr, low_freq=8.0, high_freq=12.0, order=4):
    """
    Design a Butterworth bandpass filter.
    
    Args:
        sr (int): Sampling rate in Hz
        low_freq (float): Lower cutoff frequency
        high_freq (float): Upper cutoff frequency
        order (int): Filter order
        
    Returns:
        tuple: (b, a) filter coefficients
    """
    nyquist = 0.5 * sr
    low = low_freq / nyquist
    high = high_freq / nyquist
    return butter(N=order, Wn=[low, high], btype='band')


def apply_bandpass_filter(data, sr, low_freq=8.0, high_freq=12.0, order=4):
    """
    Apply bandpass filter to data.
    
    Args:
        data (np.ndarray): Input data
        sr (int): Sampling rate
        low_freq (float): Lower cutoff frequency
        high_freq (float): Upper cutoff frequency
        order (int): Filter order
        
    Returns:
        np.ndarray: Filtered data
    """
    b, a = design_butterworth_filter(sr, low_freq, high_freq, order)
    
    if data.ndim == 1:
        return filtfilt(b, a, data)
    elif data.ndim == 2:
        n_channels = data.shape[0]
        filtered_data = np.zeros_like(data)
        for i in range(n_channels):
            filtered_data[i, :] = filtfilt(b, a, data[i, :])
        return filtered_data
    elif data.ndim == 3:
        n_trials, n_channels, _ = data.shape
        filtered_data = np.zeros_like(data)
        for i in range(n_trials):
            for j in range(n_channels):
                filtered_data[i, j, :] = filtfilt(b, a, data[i, j, :])
        return filtered_data
    else:
        raise ValueError("Input data must be a 1D, 2D, or 3D numpy array.")


def calculate_power_spectral_density(signal, sr, nperseg=None):
    """
    Calculate power spectral density using Welch's method.
    
    Args:
        signal (np.ndarray): Input signal
        sr (int): Sampling rate
        nperseg (int): Length of each segment
        
    Returns:
        tuple: (frequencies, psd)
    """
    if nperseg is None:
        nperseg = sr  # Default to 1 second
    return welch(signal, fs=sr, nperseg=nperseg)


def extract_band_power(freqs, psd, band_range):
    """
    Extract power in a specific frequency band.
    
    Args:
        freqs (np.ndarray): Frequency array
        psd (np.ndarray): Power spectral density
        band_range (tuple): (low_freq, high_freq)
        
    Returns:
        float: Average power in the band
    """
    idx_band = np.where((freqs >= band_range[0]) & (freqs <= band_range[1]))[0]
    if len(idx_band) > 0:
        return np.mean(psd[idx_band])
    return 0.0


def fit_exponential_decay(distances, correlations):
    """
    Fit exponential decay function to distance-correlation data.
    
    Args:
        distances (list): Distance values
        correlations (list): Correlation values
        
    Returns:
        tuple: (params, covariance, success)
    """
    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c
    
    try:
        initial_guesses = [1.0, 0.1, 0.0]
        params, covariance = curve_fit(exp_decay, distances, np.abs(correlations), p0=initial_guesses)
        return params, covariance, True
    except RuntimeError:
        return None, None, False


def create_electrode_positions(electrode_layout, channel_name_order, row_spacing=1.0, col_spacing=1.0):
    """
    Create electrode position mapping.
    
    Args:
        electrode_layout (list): 2D grid of electrode names
        channel_name_order (list): Order of channels in data
        row_spacing (float): Distance between rows
        col_spacing (float): Distance between columns
        
    Returns:
        tuple: (positions, distance_matrix)
    """
    pos_dict = {
        name: (r * row_spacing, c * col_spacing) 
        for r, row_list in enumerate(electrode_layout) 
        for c, name in enumerate(row_list)
    }
    
    coords = [pos_dict[name] for name in channel_name_order]
    positions = np.array(coords)
    distance_matrix = squareform(pdist(positions, 'euclidean'))
    
    return positions, distance_matrix


def get_correlation_distance_pairs(corr_matrix, distance_matrix):
    """
    Get pairs of correlation values and distances.
    
    Args:
        corr_matrix (np.ndarray): Correlation matrix
        distance_matrix (np.ndarray): Distance matrix
        
    Returns:
        list: List of (distance, correlation) tuples
    """
    if distance_matrix is None:
        raise ValueError("Distance matrix must be provided.")
    
    distances = []
    correlations = []
    num_channels = corr_matrix.shape[0]
    
    for i, j in zip(*np.triu_indices(num_channels, k=1)):
        distances.append(distance_matrix[i, j])
        correlations.append(corr_matrix[i, j])
        
    return list(zip(distances, correlations))


def ensure_3d_array(data):
    """
    Normalize input signal data to a 3D numpy array with shape
    (n_windows, n_channels, n_samples).

    Accepted inputs:
    - 1D: (n_samples,) → (1, 1, n_samples)
    - 2D: (n_channels, n_samples) → (1, n_channels, n_samples)
    - 3D: (n_windows, n_channels, n_samples) → unchanged
    - list of 1D arrays: [ (n_samples,), ... ] → (n_windows, 1, n_samples)
    - list of 2D arrays: [ (n_channels, n_samples), ... ] → (n_windows, n_channels, n_samples)

    Args:
        data: array-like or list of arrays

    Returns:
        np.ndarray: 3D array standardized to (n_windows, n_channels, n_samples)
    """
    # Handle list/tuple inputs explicitly to avoid ragged object arrays
    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            raise ValueError("Empty list/tuple provided; cannot infer data shape.")
        elements = [np.asarray(elem) for elem in data]
        # All 1D elements → stack into (n_windows, n_samples)
        if all(elem.ndim == 1 for elem in elements):
            stacked = np.stack(elements, axis=0)
            return stacked[:, np.newaxis, :]
        # All 2D elements → stack into (n_windows, n_channels, n_samples)
        if all(elem.ndim == 2 for elem in elements):
            # Validate consistent (channels, samples)
            first_shape = elements[0].shape
            if not all(elem.shape == first_shape for elem in elements):
                raise ValueError("All 2D window elements must have identical shapes (channels, samples).")
            return np.stack(elements, axis=0)
        # Fallback: try to convert whole structure to ndarray
        array_like = np.asarray(data)
    else:
        array_like = np.asarray(data)

    if array_like.ndim == 1:
        return array_like[np.newaxis, np.newaxis, :]
    if array_like.ndim == 2:
        return array_like[np.newaxis, :, :]
    if array_like.ndim == 3:
        return array_like

    raise ValueError("Input data must be 1D, 2D, or 3D (or list of 1D/2D arrays) to standardize to (windows, channels, samples).")


def standardize_subject_data(data, subject_prefix: str = "sub"):
    """
    Standardize arbitrary input into a dictionary mapping subject IDs to
    3D numpy arrays of shape (n_windows, n_channels, n_samples).

    Accepted inputs:
    - dict: {subject_id: array-like} where values are 1D/2D/3D or list of 1D/2D
    - ndarray 3D: (n_windows, n_channels, n_samples)
    - ndarray 2D: (n_channels, n_samples)
    - ndarray 1D: (n_samples,)
    - ndarray 4D: (n_subjects, n_windows, n_channels, n_samples)
    - list: treated as windows list → a single subject

    Returns:
    - dict[str, np.ndarray]
    """
    standardized = {}

    # Case 1: Already a mapping of subjects
    if isinstance(data, dict):
        for key, value in data.items():
            subject_key = str(key)
            standardized[subject_key] = ensure_3d_array(value)
        return standardized

    # Convert other array-likes
    if isinstance(data, (list, tuple)):
        standardized[f"{subject_prefix}-000"] = ensure_3d_array(data)
        return standardized

    array_like = np.asarray(data)

    if array_like.ndim == 4:
        n_subjects = array_like.shape[0]
        for i in range(n_subjects):
            standardized[f"{subject_prefix}-{i:03d}"] = ensure_3d_array(array_like[i])
        return standardized

    if array_like.ndim in (1, 2, 3):
        standardized[f"{subject_prefix}-000"] = ensure_3d_array(array_like)
        return standardized

    raise ValueError("Unsupported data shape. Expected dict, list, or ndarray with ndim in {1,2,3,4}.")
