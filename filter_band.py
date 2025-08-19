import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


def filter_band(data, sr, low_freq=8.0, high_freq=12.0, order=4):
    """
    Applies a bandpass filter to isolate a specific frequency band in a signal.

    This function is designed to be flexible and can handle data with different
    shapes:
    - 1D array (single channel, single trial)
    - 2D array (multiple channels, single trial) of shape (n_channels, n_timesteps)
    - 3D array (multiple channels, multiple trials) of shape (n_trials, n_channels, n_timesteps)

    Args:
        data (np.ndarray): The input signal data.
        sr (int): The sampling rate of the signal in Hz.
        low_freq (float, optional): The lower cutoff frequency of the band. Defaults to 8.0.
        high_freq (float, optional): The upper cutoff frequency of the band. Defaults to 12.0.
        order (int, optional): The order of the Butterworth filter. Defaults to 4.

    Returns:
        np.ndarray: The filtered signal data, with the same shape as the input.
    """
    # 1. Design the Butterworth bandpass filter
    nyquist = 0.5 * sr
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # The b and a are the numerator and denominator polynomials of the IIR filter
    b, a = butter(N=order, Wn=[low, high], btype='band')

    # 2. Apply the filter based on the input data's dimensions
    if data.ndim == 1:
        # Handle 1D array (e.g., a single channel's data)
        return filtfilt(b, a, data)
    
    elif data.ndim == 2:
        # Handle 2D array (e.g., multiple channels for one window/trial)
        # Assumes shape is (n_channels, n_timesteps)
        n_channels = data.shape[0]
        filtered_data = np.zeros_like(data)
        for i in range(n_channels):
            filtered_data[i, :] = filtfilt(b, a, data[i, :])
        return filtered_data
        
    elif data.ndim == 3:
        # Handle 3D array (e.g., multiple windows/trials of multi-channel data)
        # Assumes shape is (n_trials, n_channels, n_timesteps)
        n_trials, n_channels, _ = data.shape
        filtered_data = np.zeros_like(data)
        for i in range(n_trials):
            for j in range(n_channels):
                filtered_data[i, j, :] = filtfilt(b, a, data[i, j, :])
        return filtered_data
        
    else:
        raise ValueError("Input data must be a 1D, 2D, or 3D numpy array.")


def visualize_signal(data_to_plot, title="Signal Visualization", xlabel="Time (s)", 
                     ylabel="Amplitude", samplerate=None, separate_plots=False, channel_labels=None):
    """
    A flexible function to visualize 1D or 2D signal data.

    - If data is 1D, it plots a single line.
    - If data is 2D, it plots each row on the same graph by default.
    - If data is 2D and `separate_plots` is True, it plots each row on a separate subplot.

    Args:
        data_to_plot (np.ndarray): The 1D or 2D data to plot.
        title (str): The title for the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        sr (int, optional): The sampling rate. If provided, the x-axis will be in seconds.
        separate_plots (bool): If True and data is 2D, plot channels on separate subplots.
        channel_labels (list of str, optional): Custom labels for each channel's y-axis 
                                                when using separate_plots.
    """
    if not isinstance(data_to_plot, np.ndarray) or data_to_plot.ndim > 2:
        raise ValueError("This function can only plot 1D or 2D numpy arrays.")

    # Determine the x-axis based on sampling rate
    if data_to_plot.ndim == 1:
        num_points = len(data_to_plot)
    else:  # 2D
        num_points = data_to_plot.shape[1]

    if samplerate:
        x_axis = np.linspace(0, num_points / samplerate, num_points)
    else:
        x_axis = np.arange(num_points)
        xlabel = "Samples"

    # --- Plotting Logic ---
    if data_to_plot.ndim == 1 or not separate_plots:
        # --- Plot on a single graph (default behavior) ---
        plt.figure(figsize=(12, 6))
        if data_to_plot.ndim == 1:
            plt.plot(x_axis, data_to_plot)
        else:  # 2D on same plot
            num_channels = data_to_plot.shape[0]
            # Use provided labels or create default ones
            labels = channel_labels if channel_labels and len(channel_labels) == num_channels else [f'Signal {i+1}' for i in range(num_channels)]
            for i in range(num_channels):
                plt.plot(x_axis, data_to_plot[i, :], label=labels[i])
            plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    else:
        # --- Plot on separate subplots ---
        num_channels = data_to_plot.shape[0]
        fig, axes = plt.subplots(num_channels, 1, figsize=(12, 2 * num_channels), sharex=True)
        if num_channels == 1: # If only one channel, axes is not an array
            axes = [axes]
            
        fig.suptitle(title, fontsize=16, y=0.92) # Adjust title position for suptitle
        title = "" # Clear title to avoid duplication

        for i in range(num_channels):
            axes[i].plot(x_axis, data_to_plot[i, :])
            # Set individual y-labels for each subplot
            if channel_labels and i < len(channel_labels):
                axes[i].set_ylabel(channel_labels[i])
            else:
                axes[i].set_ylabel(f'Ch {i+1}')
            axes[i].grid(True, linestyle='--', alpha=0.6)
        
        # Set shared x-label only on the last plot
        plt.xlabel(xlabel)
        fig.tight_layout(rect=[0, 0, 1, 0.9]) # Adjust layout for suptitle
        plt.show()
        return # Exit function to avoid showing a second empty plot
