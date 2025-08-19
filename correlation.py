import numpy as np
import os
import pandas as pd
from scipy.signal import welch
from filter_band import filter_band
from core_functions import (
    calculate_power_spectral_density, 
    extract_band_power, 
    create_electrode_positions,
    get_correlation_distance_pairs,
    standardize_subject_data,
)
from visualization import plot_correlation_vs_distance

class CorrelationEvaluator:
    """
    A class to evaluate synthetic signals by calculating various correlation matrices.
    This class is designed to work with multi-subject, multi-channel time-series data,
    where the data is structured as a dictionary of {subject_id: (n_windows, n_channels, samples)}.
    """

    def __init__(self, real_data=None, synthetic_data=None, sample_rate=512, electrode_layout=None, 
                 channel_name_order=None, row_spacing=1.0, col_spacing=1.0):
        """
        Initializes the evaluator with custom electrode spacing.

        Args:
            real_data (dict): Data dictionary.
            synthetic_data (dict): Data dictionary.
            sample_rate (int): Sampling rate in Hz.
            electrode_layout (list of list of str): 2D grid of electrode names.
            channel_name_order (list of str): Order of channels in the data array.
            row_spacing (float): The distance between adjacent rows.
            col_spacing (float): The distance between adjacent columns.
        """
        # Standardize input datasets to {subject_id: (n_windows, n_channels, n_samples)}
        self.real_data = standardize_subject_data(real_data) if real_data is not None else None
        self.synthetic_data = standardize_subject_data(synthetic_data) if synthetic_data is not None else None
        self.sr = sample_rate
        self.electrode_positions = None
        self.distance_matrix = None

        self.real_subj_names = list(self.real_data.keys()) if self.real_data else []
        self.synthe_subj_names = list(self.synthetic_data.keys()) if self.synthetic_data else []

        if electrode_layout and channel_name_order:
            self.electrode_positions, self.distance_matrix = create_electrode_positions(
                electrode_layout, channel_name_order, row_spacing, col_spacing
            )
    
    def get_data_info(self):
        """
        Returns the list of subject IDs and the data shape for the first subject.

        Returns:
            tuple: A tuple containing (list_of_subject_ids, shape_of_first_subject_data).
        """
        # Prefer synthetic data for info; fallback to real
        if self.synthe_subj_names:
            first_subject_key = self.synthe_subj_names[0]
            shape_info = self.synthetic_data[first_subject_key].shape
            return self.synthe_subj_names, shape_info
        if self.real_subj_names:
            first_subject_key = self.real_subj_names[0]
            shape_info = self.real_data[first_subject_key].shape
            return self.real_subj_names, shape_info
        return [], None

    def get_correlation_distance_pairs(self, corr_matrix):
        """
        Get correlation-distance pairs for analysis.
        
        Args:
            corr_matrix (np.ndarray): Correlation matrix
            
        Returns:
            list: List of (distance, correlation) tuples
        """
        if self.distance_matrix is None:
            raise ValueError("Electrode layout must be provided during initialization to calculate distances.")
        
        return get_correlation_distance_pairs(corr_matrix, self.distance_matrix)

    def extract_alpha_band_power(self, subject_index, window_index, nperseg=None):
        """
        Extracts alpha band power for a specific window.

        Args:
            subject_index (int): The integer index of the subject.
            window_index (int): The index of the window to process.
            nperseg (int, optional): Length of each segment for Welch's method. Defaults to 3 second.

        Returns:
            np.ndarray: Alpha band power for each channel.
        """
        if not 0 <= subject_index < len(self.synthe_subj_names):
            raise IndexError(f"Subject index {subject_index} is out of bounds.")
        
        subject_id = self.synthe_subj_names[subject_index]
        subject_data = self.synthetic_data[subject_id]
        
        if not 0 <= window_index < subject_data.shape[0]:
            raise IndexError(f"Window index {window_index} is out of bounds.")
        
        if nperseg is None:
            nperseg = 1 * self.sr

        window_data = subject_data[window_index]
        n_channels = window_data.shape[0]
        alpha_band = (8.0, 12.0)
        alpha_band_power = np.zeros(n_channels)
        
        for i in range(n_channels):
            channel_signal = window_data[i, :]
            freqs, psd = calculate_power_spectral_density(channel_signal, self.sr, nperseg)
            alpha_band_power[i] = extract_band_power(freqs, psd, alpha_band)

        return alpha_band_power

    def calculate_alpha_power_correlation(self, subject_index, **kwargs):
        """
        Calculates the correlation matrix of alpha band power across channels for a specific subject.

        Args:
            subject_index (int): The index of the subject to process.
            **kwargs: Keyword arguments to be passed to the 
                      `extract_alpha_band_power` method (e.g., nperseg).

        Returns:
            np.ndarray: A 2D array of shape (n_channels, n_channels) representing
                        the correlation matrix for the specified subject.
        """
        if not 0 <= subject_index < len(self.synthe_subj_names):
            raise IndexError(f"Subject index {subject_index} is out of bounds.")
            
        subject_id = self.synthe_subj_names[subject_index]
        subject_data = self.synthetic_data[subject_id]

        n_windows, n_channels, _ = subject_data.shape
        all_window_alpha_powers = np.zeros((n_windows, n_channels))
        
        for i in range(n_windows):
            all_window_alpha_powers[i, :] = self.extract_alpha_band_power(subject_index, i, **kwargs)
        
        # Calculate the correlation matrix
        windows_corr_matrix = np.corrcoef(all_window_alpha_powers, rowvar=False)
        return windows_corr_matrix

    def calculate_spatial_correlation_for_window(self, subject_index, window_index, start_time=None, end_time=None):
        """
        Calculates spatial correlation for a window, with optional time slicing.
        
        Args:
            subject_index (int): Subject index
            window_index (int): Window index
            start_time (float, optional): Start time in seconds
            end_time (float, optional): End time in seconds
            
        Returns:
            tuple: (raw_correlation_matrix, alpha_filtered_correlation_matrix)
        """
        if not 0 <= subject_index < len(self.synthe_subj_names):
            raise IndexError(f"Subject index {subject_index} is out of bounds.")
        
        subject_name = self.synthe_subj_names[subject_index]
        subject_data = self.synthetic_data[subject_name]
        
        if not 0 <= window_index < subject_data.shape[0]:
            raise IndexError(f"Window index {window_index} is out of bounds.")
        
        window_data = subject_data[window_index]
        
        # Apply time slicing if specified
        if start_time is not None and end_time is not None:
            start_sample = int(start_time * self.sr)
            end_sample = int(end_time * self.sr)
            if end_sample > window_data.shape[1]: 
                end_sample = window_data.shape[1]
            if start_sample < 0: 
                start_sample = 0
            window_data = window_data[:, start_sample:end_sample]
        
        # Filter for alpha band
        alpha_filtered_window_data = filter_band(window_data, self.sr)
        
        # Calculate correlation matrices
        channel_corr_matrix = np.corrcoef(window_data)
        channel_alpha_corr_matrix = np.corrcoef(alpha_filtered_window_data)
        
        return channel_corr_matrix, channel_alpha_corr_matrix


    def calculate_cross_window_correlation(self, real_subj_idx, synthe_subj_idx, window_index, use_alpha_filter=False):
        """
        Calculates the spatial correlation matrix between two different windows.

        Args:
            real_subj_idx (int): The index of the first subject.
            synthe_subj_idx (int): The index of the second subject.
            window_index (int): The index of the window to process.
            use_alpha_filter (bool): If True, performs the correlation on the
                                     alpha-band-filtered signals.

        Returns:
            np.ndarray: A 2D array of shape (n_channels, n_channels) where the
                        element at [i, j] is the correlation between channel i
                        from window 1 and channel j from window 2.
        """
        if not (0 <= real_subj_idx < len(self.real_subj_names) and 
                0 <= synthe_subj_idx < len(self.synthe_subj_names)):
            raise IndexError(f"Subject index {real_subj_idx} (real) or {synthe_subj_idx} (synthetic) is out of bounds.")
        
        real_subj_id, synthe_subj_id = self.real_subj_names[real_subj_idx], self.synthe_subj_names[synthe_subj_idx]
        real_data, synthe_data = self.real_data[real_subj_id], self.synthetic_data[synthe_subj_id]

        # Validate window indices
        if not (0 <= window_index < real_data.shape[0] and 0 <= window_index < synthe_data.shape[0]):
            raise IndexError("One or both window indices are out of bounds.")

        # Get the data for both windows
        data1 = real_data[window_index]
        data2 = synthe_data[window_index]

        # Optionally filter both signals for the alpha band
        if use_alpha_filter:
            data1 = filter_band(data1, self.sr)
            data2 = filter_band(data2, self.sr)

        # To calculate the cross-correlation, we stack the two sets of channels
        # and compute the full correlation matrix.
        combined_data = np.vstack((data1, data2))
        full_corr_matrix = np.corrcoef(combined_data)

        # The full matrix contains four blocks. We want the top-right block,
        # which represents the correlation between data1 and data2.
        n_channels = data1.shape[0]
        cross_corr_matrix = full_corr_matrix[:n_channels, n_channels:]
        
        return cross_corr_matrix


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import seaborn as sns

    data_path = "/Users/groupies/Documents/NOVAProjects/Multichannel_ECoG_GAN-main/Version2/global_norm_3s_512hz.pkl"
    data = pd.read_pickle(data_path)

    # evaluator = CorrelationEvaluator(data, sample_rate=512)

    # window_corr_matrix = evaluator.calculate_alpha_power_correlation(subject_index=1)

    electrode_grid = [
        ["Gr16", "Gr11", "Gr6", "Gr1"],
        ["Gr17", "Gr12", "Gr7", "Gr2"],
        ["Gr18", "Gr13", "Gr8", "Gr3"],
        ["Gr19", "Gr14", "Gr9", "Gr4"],
        ["Gr20", "Gr15", "Gr10", "Gr5"]
    ]

    channel_name_order = [f'Gr{i}' for i in range(1, 21)]


    evaluator = CorrelationEvaluator(
        synthetic_data=data,
        sample_rate=512,
        electrode_layout=electrode_grid,
        channel_name_order=channel_name_order,
        col_spacing=2,
        row_spacing=1
        )
    
    custom_labels = [f'C{i}{j}' for i in range(1, 6) for j in range(1, 5)]
    

    # for i in range(20):
    # Calculate spatial correlation for a window
    raw_corr, alpha_corr = evaluator.calculate_spatial_correlation_for_window(
        subject_index=0, 
        window_index=1
    )

    # Get the distance-correlation pairs
    # We'll use the alpha-filtered correlation matrix for this analysis
    dist_corr_data = evaluator.get_correlation_distance_pairs(alpha_corr)

    # Plot the results
    plot_correlation_vs_distance(
        dist_corr_data,
        title=f"C-D Alpha_decay Window:{1}"
    )


    plt.figure(figsize=(12, 10))
    sns.heatmap(
        raw_corr, 
        cmap='coolwarm', 
        annot=False, 
        vmin=-1, 
        vmax=1,
        xticklabels=custom_labels, # Use custom labels for x-axis
        yticklabels=custom_labels  # Use custom labels for y-axis
    )
    
    # Rotate labels for better readability
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join('/Users/groupies/Documents/NOVAProjects/Multichannel_ECoG_GAN-main/Spatial_Matrix', f"Corr_heat_alpha Window:{1}"))
    plt.show()



    # VISUALIZE Channel Correlation matraix between different Windows(Signals)

    cross_proj_corr_matrix = evaluator.calculate_cross_window_correlation(synthe_subj_idx=1, real_subj_idx=2, window_index=1, use_alpha_filter=True)


    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cross_proj_corr_matrix, 
        cmap='coolwarm', 
        annot=False, 
        vmin=-1, 
        vmax=1,
        xticklabels=custom_labels, # Use custom labels for x-axis
        yticklabels=custom_labels  # Use custom labels for y-axis
    )
    
    # Rotate labels for better readability
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.show()


    print(f"Generating heatmaps for window {1}...")
    # Create a figure with two subplots side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    sns.heatmap(
        raw_corr, 
        ax=axes[0], 
        cmap='coolwarm', 
        annot=False,  # Set annot to False for cleaner look with many channels
        vmin=-1, 
        vmax=1,
        xticklabels=custom_labels, # Use custom labels
        yticklabels=custom_labels  # Use custom labels
    )
    axes[0].set_title('Spatial Correlation (Raw): Window {1}', fontsize=16)
    axes[0].set_xlabel('Channels', fontsize=12)
    axes[0].set_ylabel('Channels', fontsize=12)
    # Rotate labels for better readability
    plt.setp(axes[0].get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor")
    plt.setp(axes[0].get_yticklabels(), rotation=0)

    sns.heatmap(
        alpha_corr, 
        ax=axes[1], 
        cmap='coolwarm', # viridis
        annot=False, # Set annot to False for cleaner look with many channels
        vmin=-1, 
        vmax=1,
        xticklabels=custom_labels, # Use custom labels
        yticklabels=custom_labels  # Use custom labels
    )
    axes[1].set_title(f'Spatial Correlation (Alpha): Window {1}', fontsize=16)
    axes[1].set_xlabel('Channels', fontsize=12)
    axes[1].set_ylabel('Channels', fontsize=12)
    # Rotate labels for better readability
    plt.setp(axes[1].get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor")
    plt.setp(axes[1].get_yticklabels(), rotation=0)


    # Add a main title for the entire figure and save it
    fig.suptitle(f'Correlation Analysis for Subject 1', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


