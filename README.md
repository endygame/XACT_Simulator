# XACT Simulation with XCAT Thorax Phantom

## Overview

This project provides an X-ray-induced Acoustic Computed Tomography (XACT) simulation using the XCAT thorax phantom with accurate tissue properties.

### Features

- 3D Thorax Phantom based on XCAT phantom with 10 tissue types
- Accurate physics implementation for acoustic and X-ray properties
- Interactive GUI for simulation control
- Comprehensive visualization of 3D anatomy, sensor arrays, and results
- Audio export of acoustic sensor signals as WAV files
- Generic sensor interface supporting multiple sensor types
- Results organization with timestamped folders
- Cross-platform setup script

### Installation

#### Quick Setup (Recommended)

Run the cross-platform setup script:

```bash
python setup.py
```

This will:
- Check Python version compatibility
- Install all required dependencies
- Set up project directories
- Verify installation

#### Manual Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/endygame/XACT_Simulator/tree/main
   cd Acoustic
   ```

2. **Install minimal dependencies**:
   ```bash
   pip install -r requirements_minimal.txt
   ```

3. **Install optional dependencies**:
   ```bash
   pip install nibabel soundfile opencv-python plotly numba vtk pyvista
   ```

### Usage

#### 1. Generate XCAT Phantom
```bash
python xcat_thorax_loader.py
```

#### 2. Run XACT Simulation
```bash
python xact_demo.py
```

#### 3. Launch GUI
```bash
python gui.py
```

#### 4. Test Sensor Interface
```bash
python sensor_interface.py
```

### File Structure

```
├── xact_demo.py                       # Main simulation demo
├── gui.py                            # Tkinter GUI interface
├── xact_physics.py                   # XACT physics implementation
├── anatomy_loader.py                 # Anatomical data loader
├── xcat_thorax_loader.py             # XCAT phantom generator
├── sensor_interface.py               # Generic sensor interface
├── results_manager.py                # Results organization system
├── setup.py                          # Cross-platform setup script
├── requirements.txt                  # Full dependency list
├── requirements_minimal.txt          # Minimal dependencies
├── anatomical_data/                  # Generated phantom files
│   ├── xcat_thorax_phantom.nii.gz
│   ├── xcat_thorax_speed.nii.gz
│   ├── xcat_thorax_density.nii.gz
│   └── xcat_thorax_absorption.nii.gz
├── results/                          # Organized simulation results
│   └── YYYYMMDD_HHMMSS_run_name/
│       ├── images/
│       ├── audio/
│       ├── data/
│       └── logs/
└── XCAT_Thorax_Phantom_Documentation.md
```

### Results Organization

All simulation results are automatically organized into timestamped directories:

- **Images**: 3D visualizations and simulation results (PNG)
- **Audio**: Acoustic sensor signals (WAV files)
- **Data**: Simulation parameters and datasets (JSON, NPZ)
- **Logs**: Execution logs and metadata

Example structure:
```
results/
├── 20241201_143022_thorax_scan/
│   ├── images/
│   │   ├── anatomy/
│   │   └── results/
│   ├── audio/
│   │   └── sensors/
│   ├── data/
│   │   └── parameters.json
│   └── run_metadata.json
└── 20241201_150315_test_run/
    └── ...
```

### Key Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **Phantom Size** | 256×256×200 | - | Voxels |
| **Voxel Resolution** | 1.5 mm | - | Spatial resolution |
| **Sensors** | 128 | - | Circular array |
| **X-ray Energy** | 120 keV | 50-300 | Beam energy |
| **Center Frequency** | 1.5 MHz | 1-10 | Acoustic frequency |
| **Scan Duration** | 100 ns | 0.1-1000 | Signal duration |

### Sensor Interface

The generic sensor interface supports multiple sensor types:

```python
from sensor_interface import create_default_ultrasound_array

# Create ultrasound sensor array
array = create_default_ultrasound_array(
    center=np.array([0, 0, 0]),
    radius=0.1,  # 10 cm
    num_sensors=128,
    center_frequency=1.5e6,  # 1.5 MHz
    bandwidth=3.0e6  # 3 MHz
)

# Detect signals
signals = array.detect_all_signals(pressure_field, time_step)
```

### Tissue Types

The phantom includes 10 tissue types with accurate acoustic properties:

1. **Air** (343 m/s) - Lung air spaces
2. **Soft Tissue** (1540 m/s) - General organs
3. **Lung** (600 m/s) - Lung parenchyma
4. **Bone** (3500 m/s) - Ribs, spine
5. **Blood** (1576 m/s) - Major vessels
6. **Heart** (1576 m/s) - Cardiac muscle
7. **Liver** (1570 m/s) - Hepatic tissue
8. **Fat** (1450 m/s) - Adipose tissue
9. **Muscle** (1580 m/s) - Skeletal muscle
10. **Cartilage** (1660 m/s) - Costal cartilage

### Platform Support

The setup script automatically detects and supports:

- **macOS**: Including Apple Silicon with MPS acceleration
- **Linux**: Ubuntu, CentOS, Arch Linux distributions
- **Windows**: Windows 10/11 with standard Python installation

### Documentation

For complete technical details, mathematical models, and tissue properties, see:
**[XCAT_Thorax_Phantom_Documentation.md](XCAT_Thorax_Phantom_Documentation.md)**

### Troubleshooting

**Common Issues:**

1. **Import errors**: Run `python setup.py` to install dependencies
2. **Memory issues**: Reduce phantom size or use synthetic data
3. **GUI crashes**: Use `python gui.py` instead of running from IDE
4. **File not found**: Run `python xcat_thorax_loader.py` first
5. **Results not saved**: Check that `results/` directory exists


### License

This project is for educational and research purposes. The XCAT phantom methodology is based on published literature (see documentation for references).


```

### Contact

For questions or issues, please open a GitHub issue or contact the maintainers. 