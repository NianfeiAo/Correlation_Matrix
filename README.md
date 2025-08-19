# Multichannel ECoG GAN Spatial Matrix Analysis

This project is used to analyze spatial correlation matrices of multichannel ECoG data, including correlation analysis of raw signals and alpha-band filtered signals.

## Project Structure

The refactored code has been clearly separated according to functionality:

```
Spatial_Matrix/
├── core_functions.py      # Core functional functions
├── visualization.py       # Visualization functions
├── correlation.py         # Correlation evaluator class
├── filter_band.py        # Signal filtering functions
├── main.py               # Main program entry
├── config.py             # Configuration file
└── README.md             # Documentation
```

## File Descriptions

### 1. core_functions.py
Contains all core signal processing and mathematical calculation functions:
- `design_butterworth_filter()`: Design Butterworth filter
- `apply_bandpass_filter()`: Apply bandpass filter
- `calculate_power_spectral_density()`: Calculate power spectral density
- `extract_band_power()`: Extract power in specific frequency band
- `calculate_correlation_matrix()`: Calculate correlation matrix
- `fit_exponential_decay()`: Fit exponential decay function
- `create_electrode_positions()`: Create electrode position mapping
- `get_correlation_distance_pairs()`: Get correlation-distance pairs

### 2. visualization.py
Contains all plotting and visualization functions:
- `plot_correlation_vs_distance()`: Plot correlation vs distance
- `plot_power_spectral_density()`: Plot power spectral density
- `plot_correlation_heatmap()`: Plot correlation heatmap
- `plot_dual_correlation_heatmaps()`: Plot dual correlation heatmaps
- `plot_cross_window_correlation()`: Plot cross-window correlation
- `plot_signal_comparison()`: Plot signal comparison

### 3. correlation.py
Contains the main `CorrelationEvaluator` class for:
- Calculating spatial correlation matrices
- Extracting alpha band power
- Computing cross-window correlations
- Managing electrode layout and distance matrices

### 4. filter_band.py
Signal filtering related functions:
- `filter_band()`: Apply frequency band filter
- `visualize_signal()`: Signal visualization

### 5. main.py
Main program entry, containing:
- `main()`: Complete analysis pipeline
- `demo_single_window_analysis()`: Single window analysis demonstration

### 6. config.py
Configuration file managing all parameters:
- Data paths
- Signal processing parameters
- Electrode configuration
- Visualization parameters

## Usage

### Run Complete Analysis
```bash
python main.py
```

### Run Demo Analysis
```bash
python main.py
# Uncomment demo_single_window_analysis() call
```

### Custom Analysis
```python
from correlation import CorrelationEvaluator
from visualization import plot_correlation_heatmap
from config import *

# Initialize evaluator
evaluator = CorrelationEvaluator(
    synthetic_data=data,
    sample_rate=SAMPLE_RATE,
    electrode_layout=ELECTRODE_GRID,
    channel_name_order=CHANNEL_NAME_ORDER,
    col_spacing=COL_SPACING,
    row_spacing=ROW_SPACING
)

# Calculate correlations
raw_corr, alpha_corr = evaluator.calculate_spatial_correlation_for_window(
    subject_index=0, 
    window_index=0
)

# Plot heatmap
plot_correlation_heatmap(
    raw_corr,
    title="Raw Correlation Matrix",
    custom_labels=CUSTOM_LABELS
)
```

## Main Features

1. **Spatial Correlation Analysis**: Calculate correlation matrices between multichannel signals
2. **Alpha Band Filtering**: Extract 8-12Hz frequency band signals for analysis
3. **Distance-Correlation Analysis**: Analyze the relationship between electrode distance and correlation
4. **Cross-Window Analysis**: Compare correlations between different time windows
5. **Power Spectral Analysis**: Calculate and visualize signal power spectral density

## Output Files

Analysis results will be saved in the configured output directory, including:
- Correlation vs distance plots
- Correlation heatmaps (raw and alpha-filtered)
- Power spectral density plots
- Cross-window correlation plots

## Dependencies

- numpy
- pandas
- matplotlib
- seaborn
- scipy

## Installation

```bash
pip install -r requirements.txt
```

## Notes

1. Ensure data paths are correctly configured
2. Electrode layout and channel order must match the data
3. Sampling rate parameters must be consistent with actual data
4. Output directory must have write permissions
# Correlation_Matrix_Optimized
# Correlation_Matrix_Optimized
# Correlation_Matrix_Optimized
# Correlation_Matrix_Optimized
# Correlation_Matrix_Optimized
# Correlation_Matrix_Optimized
