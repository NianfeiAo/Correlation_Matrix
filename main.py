import os
import pandas as pd
import numpy as np
from correlation import CorrelationEvaluator
from visualization import (
    plot_correlation_vs_distance,
    plot_power_spectral_density,
    plot_correlation_heatmap,
    plot_dual_correlation_heatmaps,
    plot_cross_window_correlation,
    plot_signal_comparison,
    plot_raw_vs_alpha
)
from filter_band import filter_band


def main():
    """
    Main function to run the correlation analysis pipeline.
    """
    # Configuration
    data_path = "/Users/groupies/Documents/NOVAProjects/Multichannel_ECoG_GAN-main/Version2/global_norm_3s_512hz.pkl"
    output_dir = "/Users/groupies/Documents/NOVAProjects/Multichannel_ECoG_GAN-main/Spatial_Matrix_Optimized/results"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    data = pd.read_pickle(data_path)
    
    # Define electrode layout
    electrode_grid = [
        ["Gr16", "Gr11", "Gr6", "Gr1"],
        ["Gr17", "Gr12", "Gr7", "Gr2"],
        ["Gr18", "Gr13", "Gr8", "Gr3"],
        ["Gr19", "Gr14", "Gr9", "Gr4"],
        ["Gr20", "Gr15", "Gr10", "Gr5"]
    ]
    
    channel_name_order = [f'Gr{i}' for i in range(1, 21)]
    custom_labels = [f'C{i}{j}' for i in range(1, 6) for j in range(1, 5)]
    
    # Initialize evaluator
    print("Initializing correlation evaluator...")
    evaluator = CorrelationEvaluator(
        synthetic_data=data,
        real_data=data, 
        sample_rate=512,
        electrode_layout=electrode_grid,
        channel_name_order=channel_name_order,
        col_spacing=2,
        row_spacing=1
    )
    
    # Get data info
    subjects, shape_info = evaluator.get_data_info()
    print(f"Loaded data for {len(subjects)} subjects")
    print(f"Data shape: {shape_info}")
    
    # Analysis parameters
    num_windows = min(1, shape_info[0]) if shape_info else 20
    subject_indices = [0, 1, 2]  # Analyze first 3 subjects
    SHOW_PLOTS = True   # Toggle showing plots interactively
    
    print(f"Analyzing {num_windows} windows for subjects {subject_indices}")
    
    # Main analysis loop
    for window_idx in range(num_windows):
        print(f"\nProcessing window {window_idx}...")
        
        # 1. Calculate spatial correlations for each subject
        for subject_idx in subject_indices:
            if subject_idx >= len(subjects):
                continue
                
            try:
                # Calculate spatial correlations
                raw_corr, alpha_corr = evaluator.calculate_spatial_correlation_for_window(
                    subject_index=subject_idx, 
                    window_index=window_idx
                )
                
                # Get distance-correlation pairs for alpha-filtered data
                dist_corr_data = evaluator.get_correlation_distance_pairs(alpha_corr)
                
                # Plot correlation vs distance
                plot_correlation_vs_distance(
                    dist_corr_data,
                    title=f"Correlation vs Distance - Subject {subject_idx}, Window {window_idx}",
                    save_path=os.path.join(output_dir, f"corr_vs_dist_subj_{subject_idx}_window_{window_idx}.png"),
                    show=SHOW_PLOTS
                )
                
                # Plot dual correlation heatmaps
                plot_dual_correlation_heatmaps(
                    raw_corr, 
                    alpha_corr, 
                    window_idx,
                    custom_labels=custom_labels,
                    save_path=os.path.join(output_dir, f"dual_corr_heatmaps_subj_{subject_idx}_window_{window_idx}.png"),
                    show=SHOW_PLOTS
                )
                
                # Plot individual correlation heatmaps
                plot_correlation_heatmap(
                    raw_corr,
                    title=f"Raw Correlation - Subject {subject_idx}, Window {window_idx}",
                    custom_labels=custom_labels,
                    save_path=os.path.join(output_dir, f"raw_corr_subj_{subject_idx}_window_{window_idx}.png"),
                    show=SHOW_PLOTS
                )
                
                plot_correlation_heatmap(
                    alpha_corr,
                    title=f"Alpha-Filtered Correlation - Subject {subject_idx}, Window {window_idx}",
                    custom_labels=custom_labels,
                    save_path=os.path.join(output_dir, f"alpha_corr_subj_{subject_idx}_window_{window_idx}.png"),
                    show=SHOW_PLOTS
                )

                # Use the same subject id already resolved above to avoid hardcoded keys
                subj_id = subjects[subject_idx]
                plot_raw_vs_alpha(
                    evaluator.synthetic_data[subj_id][window_idx],
                    sr=512,
                    title=f"Raw vs Alpha-Filtered - Subject {subject_idx}, Window {window_idx}",
                    channel_index=0,
                    window_index=window_idx,
                    save_path=os.path.join(output_dir, f"raw_vs_alpha_subj_{subject_idx}_window_{window_idx}.png"),
                    show=SHOW_PLOTS
                )

            except Exception as e:
                print(f"Error processing subject {subject_idx}, window {window_idx}: {e}")
                continue
        
        # 2. Calculate cross-window correlations between subjects
        if len(subject_indices) >= 2:
            try:
                cross_corr_matrix = evaluator.calculate_cross_window_correlation(
                    real_subj_idx=subject_indices[0],
                    synthe_subj_idx=subject_indices[1],
                    window_index=window_idx,
                    use_alpha_filter=True
                )
                
                # Plot cross-window correlation
                plot_cross_window_correlation(
                    cross_corr_matrix,
                    window_idx,
                    custom_labels=custom_labels,
                    save_path=os.path.join(output_dir, f"cross_window_corr_window_{window_idx}.png"),
                    show=SHOW_PLOTS
                )
                
            except Exception as e:
                print(f"Error calculating cross-window correlation for window {window_idx}: {e}")
                continue
    
    # 3. Calculate alpha power correlations across windows
    print("\nCalculating alpha power correlations across windows...")
    for subject_idx in subject_indices:
        if subject_idx >= len(subjects):
            continue
            
        try:
            alpha_power_corr = evaluator.calculate_alpha_power_correlation(subject_index=subject_idx)

            # Plot alpha power correlation matrix
            plot_correlation_heatmap(
                alpha_power_corr,
                title=f"Alpha Power Correlation - Subject {subject_idx}",
                custom_labels=custom_labels,
                save_path=os.path.join(output_dir, f"alpha_power_corr_subj_{subject_idx}.png"),
                show=SHOW_PLOTS
            )
            
        except Exception as e:
            print(f"Error calculating alpha power correlation for subject {subject_idx}: {e}")
            continue
    
    # 4. Generate PSD plots for sample windows
    print("\nGenerating PSD plots...")
    for subject_idx in subject_indices:  # Just for first subject
        if subject_idx >= len(subjects):
            continue
            
        for window_idx in [0, 5, 10]:  # Sample windows
            try:
                subject_data = evaluator.synthetic_data[subjects[subject_idx]]
                window_data = subject_data[window_idx]
                
                plot_power_spectral_density(
                    window_data,
                    sr=512,
                    subject_id=subjects[subject_idx],
                    window_index=window_idx,
                    save_path=os.path.join(output_dir, f"psd_subj_{subject_idx}_window_{window_idx}.png"),
                    show=SHOW_PLOTS
                )
                
            except Exception as e:
                print(f"Error generating PSD for subject {subject_idx}, window {window_idx}: {e}")
                continue
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")


def demo_single_window_analysis():
    """
    Demo function for analyzing a single window with detailed visualization.
    """
    print("Running single window analysis demo...")
    
    # Load data
    data_path = "/Users/groupies/Documents/NOVAProjects/Multichannel_ECoG_GAN-main/Version2/global_norm_3s_512hz.pkl"
    data = pd.read_pickle(data_path)
    
    # Initialize evaluator
    electrode_grid = [
        ["Gr16", "Gr11", "Gr6", "Gr1"],
        ["Gr17", "Gr12", "Gr7", "Gr2"],
        ["Gr18", "Gr13", "Gr8", "Gr3"],
        ["Gr19", "Gr14", "Gr9", "Gr4"],
        ["Gr20", "Gr15", "Gr10", "Gr5"]
    ]
    
    channel_name_order = [f'Gr{i}' for i in range(1, 21)]
    custom_labels = [f'C{i}{j}' for i in range(1, 6) for j in range(1, 5)]
    
    evaluator = CorrelationEvaluator(
        synthetic_data=data,
        real_data=data,
        sample_rate=512,
        electrode_layout=electrode_grid,
        channel_name_order=channel_name_order,
        col_spacing=2,
        row_spacing=1
    )
    

    # Analyze single window
    subject_idx = 0
    window_idx = 0
    
    # Get window data
    subject_data = data[list(data.keys())[subject_idx]]
    window_data = subject_data[window_idx]
    
    # Filter for alpha band
    alpha_filtered_data = filter_band(window_data, 512)

    # Plot signal comparison for first channel
    plot_signal_comparison(
        window_data[0], 
        alpha_filtered_data[0],
        title=f"Channel 1 - Subject {subject_idx}, Window {window_idx}",
        sr=512
    )
    
    # Calculate and plot correlations
    raw_corr, alpha_corr = evaluator.calculate_spatial_correlation_for_window(
        subject_index=subject_idx, 
        window_index=window_idx
    )
    
    # Plot correlation heatmaps
    plot_dual_correlation_heatmaps(
        raw_corr, 
        alpha_corr, 
        window_idx,
        custom_labels=custom_labels
    )
    
    # Plot correlation vs distance
    dist_corr_data = evaluator.get_correlation_distance_pairs(alpha_corr)
    plot_correlation_vs_distance(
        dist_corr_data,
        title=f"Correlation vs Distance - Demo Window {window_idx}"
    )


if __name__ == "__main__":
    # Run the main analysis
    main()
    
    # # Uncomment to run demo
    # demo_single_window_analysis()
