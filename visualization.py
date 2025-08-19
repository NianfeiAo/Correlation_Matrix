import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from core_functions import fit_exponential_decay
from scipy.signal import welch, butter, filtfilt
from filter_band import filter_band


def plot_correlation_vs_distance(distance_corr_pairs, title="Correlation vs. Distance", save_path=None):
    """
    Creates a scatter plot of correlation vs. distance with a best-fit line.

    Args:
        distance_corr_pairs (list of tuples): Data from get_correlation_distance_pairs.
        title (str): The title for the plot.
        save_path (str): Path to save the plot. If None, plot is not saved.
    """
    # Unzip the pairs into two separate lists
    distances, correlations = zip(*distance_corr_pairs)
    
    # Fit the exponential decay curve
    params, covariance, fit_successful = fit_exponential_decay(distances, correlations)

    # Plotting
    plt.figure(figsize=(10, 7))
    # Plot the original data points
    plt.scatter(distances, correlations, alpha=0.6, edgecolors='k', label='Data Points')
    
    # If the fit was successful, plot the line
    if fit_successful:
        # Generate x-values for a smooth line
        x_fit = np.linspace(min(distances), max(distances), 200)
        # Calculate the y-values using the fitted function and parameters
        def exp_decay(x, a, b, c):
            return a * np.exp(-b * x) + c
        y_fit = exp_decay(x_fit, *params)
        # Plot the best-fit line
        plt.plot(x_fit, y_fit, color='black', linewidth=2, label='Exponential Fit')

    plt.title(title, fontsize=16)
    plt.xlabel("Inter-Electrode Distance (Arbitrary Units)", fontsize=12)
    plt.ylabel("Correlation", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(-1.1, 1.1)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    # plt.show()


def plot_power_spectral_density(window_data, sr, subject_id, window_index, nperseg=None, save_path=None):
    """
    Plot Power Spectral Density (PSD) for all channels in a window.
    
    Args:
        window_data (np.ndarray): Window data of shape (n_channels, samples)
        sr (int): Sampling rate
        subject_id (str): Subject identifier
        window_index (int): Window index
        save_path (str): Path to save the plot
    """
    n_channels = window_data.shape[0]
    alpha_band = (8.0, 12.0)
    
    # Create a grid of subplots for all channels
    rows = (n_channels + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(15, 3 * rows), sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()
    fig.suptitle(f'Power Spectral Density (PSD) for Subject: {subject_id} | Window: {window_index}', fontsize=16)

    if nperseg is None:
        nperseg = 1 * sr
    
    for i in range(n_channels):
        channel_signal = window_data[i, :]
        # freqs, psd = plt.psd(channel_signal, Fs=sr, NFFT=min(256, len(channel_signal)))
        freqs, psd = welch(channel_signal, fs=sr, nperseg=nperseg)
        
        
        # Find the indices corresponding to the alpha band
        idx_alpha = np.where((freqs >= alpha_band[0]) & (freqs <= alpha_band[1]))[0]
        

        ax = axes[i]
        ax.plot(freqs, psd, color='blue', linewidth=1)
        # Shade the alpha band area
        ax.fill_between(freqs[idx_alpha], psd[idx_alpha], color='red', alpha=0.5)
        ax.set_title(f'Channel {i+1}')
        ax.set_xlim(0, 20)  # Limit x-axis to a reasonable frequency range
        ax.grid(True, linestyle='--', alpha=0.5)

    # Turn off any unused subplots
    for i in range(n_channels, len(axes)):
        axes[i].set_visible(False)

    # Add shared labels
    fig.text(0.5, 0.04, 'Frequency (Hz)', ha='center', va='center', fontsize=14)
    fig.text(0.06, 0.5, 'Power Spectral Density (uV^2/Hz)', ha='center', va='center', rotation='vertical', fontsize=14)
    plt.tight_layout(rect=[0.07, 0.05, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved PSD plot to '{save_path}'")
    # plt.show()


def plot_correlation_heatmap(corr_matrix, title="Correlation Matrix", custom_labels=None, save_path=None):
    """
    Plot correlation matrix as a heatmap.
    
    Args:
        corr_matrix (np.ndarray): Correlation matrix
        title (str): Plot title
        custom_labels (list): Custom labels for axes
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix, 
        cmap='coolwarm', 
        annot=False, 
        vmin=-1, 
        vmax=1,
        xticklabels=custom_labels,
        yticklabels=custom_labels
    )
    
    # Rotate labels for better readability
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    plt.title(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    # plt.show()


def plot_dual_correlation_heatmaps(raw_corr, alpha_corr, window_index, custom_labels=None, save_path=None):
    """
    Plot two correlation heatmaps side by side.
    
    Args:
        raw_corr (np.ndarray): Raw correlation matrix
        alpha_corr (np.ndarray): Alpha-filtered correlation matrix
        window_index (int): Window index for title
        custom_labels (list): Custom labels for axes
        save_path (str): Path to save the plot
    """
    # Create a figure with two subplots side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    # Plot raw correlation
    sns.heatmap(
        raw_corr, 
        ax=axes[0], 
        cmap='coolwarm', 
        annot=False,
        vmin=-1, 
        vmax=1,
        xticklabels=custom_labels,
        yticklabels=custom_labels
    )
    axes[0].set_title(f'Spatial Correlation (Raw): Window {window_index}', fontsize=16)
    axes[0].set_xlabel('Channels', fontsize=12)
    axes[0].set_ylabel('Channels', fontsize=12)
    plt.setp(axes[0].get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor")
    plt.setp(axes[0].get_yticklabels(), rotation=0)

    # Plot alpha-filtered correlation
    sns.heatmap(
        alpha_corr, 
        ax=axes[1], 
        cmap='coolwarm',
        annot=False,
        vmin=-1, 
        vmax=1,
        xticklabels=custom_labels,
        yticklabels=custom_labels
    )
    axes[1].set_title(f'Spatial Correlation (Alpha): Window {window_index}', fontsize=16)
    axes[1].set_xlabel('Channels', fontsize=12)
    axes[1].set_ylabel('Channels', fontsize=12)
    plt.setp(axes[1].get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor")
    plt.setp(axes[1].get_yticklabels(), rotation=0)

    # Add a main title for the entire figure
    fig.suptitle(f'Correlation Analysis for Window {window_index}', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path)
    # plt.show()


def plot_cross_window_correlation(cross_corr_matrix, window_index, custom_labels=None, save_path=None):
    """
    Plot cross-window correlation matrix.
    
    Args:
        cross_corr_matrix (np.ndarray): Cross-window correlation matrix
        window_index (int): Window index for title
        custom_labels (list): Custom labels for axes
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cross_corr_matrix, 
        cmap='coolwarm', 
        annot=False, 
        vmin=-1, 
        vmax=1,
        xticklabels=custom_labels,
        yticklabels=custom_labels
    )
    
    plt.title(f'Cross-Window Correlation: Window {window_index}', fontsize=16)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    # plt.show()


def plot_signal_comparison(raw_signal, filtered_signal, title="Signal Comparison", 
                          xlabel="Time (s)", ylabel="Amplitude", sr=None, save_path=None):
    """
    Plot raw and filtered signals for comparison.
    
    Args:
        raw_signal (np.ndarray): Raw signal data
        filtered_signal (np.ndarray): Filtered signal data
        title (str): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        sr (int): Sampling rate
        save_path (str): Path to save the plot
    """
    if sr:
        time_axis = np.linspace(0, len(raw_signal) / sr, len(raw_signal))
    else:
        time_axis = np.arange(len(raw_signal))
        xlabel = "Samples"
    
    plt.figure(figsize=(12, 8))
    
    # Plot raw signal
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, raw_signal, label='Raw Signal', color='blue')
    plt.title(f'{title} - Raw Signal')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot filtered signal
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, filtered_signal, label='Filtered Signal', color='red')
    plt.title(f'{title} - Filtered Signal')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    # plt.show()


def plot_raw_vs_alpha(data, sr, title="Raw vs Alpha-Filtered", channel_index=0, window_index=0,
                      low_freq=8.0, high_freq=12.0, order=4, save_path=None):
    """
    Visualize a signal before and after alpha-band filtering.

    Accepts 1D (samples), 2D (channels, samples), or 3D (windows, channels, samples) inputs.

    Args:
        data (np.ndarray): Input data.
        sr (int): Sampling rate in Hz.
        title (str): Plot title prefix.
        channel_index (int): Channel index to visualize for 2D/3D inputs.
        window_index (int): Window index for 3D inputs.
        low_freq (float): Lower cutoff for alpha band.
        high_freq (float): Upper cutoff for alpha band.
        order (int): Butterworth filter order.
        save_path (str): Optional path to save the plot.
    """
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    if data.ndim == 1:
        raw_signal = data
    elif data.ndim == 2:
        if not (0 <= channel_index < data.shape[0]):
            raise IndexError("channel_index out of bounds for 2D data")
        raw_signal = data[channel_index, :]
    elif data.ndim == 3:
        if not (0 <= window_index < data.shape[0]):
            raise IndexError("window_index out of bounds for 3D data")
        if not (0 <= channel_index < data.shape[1]):
            raise IndexError("channel_index out of bounds for 3D data")
        raw_signal = data[window_index, channel_index, :]
    else:
        raise ValueError("Input data must be a 1D, 2D, or 3D numpy array.")

    # Filter just the selected 1D signal for efficiency
    filtered_signal = filter_band(raw_signal, sr, low_freq=low_freq, high_freq=high_freq, order=order)

    plot_signal_comparison(
        raw_signal,
        filtered_signal,
        title=title,
        sr=sr,
        save_path=save_path
    )
