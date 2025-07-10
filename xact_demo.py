"""
XACT Demo

High-performance XACT simulation using realistic anatomical models
with immediate 3D visualization and no artificial delays.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend to avoid threading issues
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import warnings
from scipy import signal
from scipy.fft import fft, fftfreq
import sys

from anatomy_loader import AnatomyLoader
from xact_engine import TISSUE_DATABASE
from results_manager import ResultsManager

# Audio processing
try:
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# GPU acceleration
try:
    import torch
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        print(" Apple Silicon GPU ")
    else:
        DEVICE = torch.device("cpu")
        print(" Using CPU")
    GPU_AVAILABLE = True
except ImportError:
    DEVICE = None
    GPU_AVAILABLE = False

class XACTDemo:
    """High-performance XACT demo with realistic anatomical models"""
    
    def __init__(self):
        self.anatomy_loader = AnatomyLoader()
        self.results_manager = ResultsManager()
        self.current_dataset = None
        self.current_session = None
        self.sensors = []
        self.time_signals = []
        self.acoustic_signals = []
        self.time_axis = None
        self.scan_params = {
            'scan_duration': 1e-6,  # 1 microsecond
            'sample_rate': 50e6     # 50 MHz
        }
    
    def setup_acoustic_sensors(self, num_sensors: int = 128):
        """Setup acoustic sensor array"""
        radius = 0.1  # 10cm radius
        angles = np.linspace(0, 2*np.pi, num_sensors, endpoint=False)
        
        self.sensors = []
        for angle in angles:
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = 0  # Ring array in XY plane
            
            sensor = {
                'position': np.array([x, y, z]),
                'normal': np.array([-x, -y, 0]) / radius,  # Point towards center
                'frequency': 2.5e6,  # 2.5 MHz center frequency
                'bandwidth': 0.8     # 80% bandwidth
            }
            self.sensors.append(sensor)
            
    def run_full_demo(self):
        """Run complete XACT demo with anatomy"""

        # Initialize results session
        self.current_session = self.results_manager.create_run_directory("xact_full_demo")
        print(f"\n Results will be saved to: {self.current_session}")
        
        print("\n Loading anatomical models...")

        # First try to load XCAT thorax phantom
        print("\n Loading XCAT thorax phantom...")
        xcat_data = self.anatomy_loader.load_xcat_thorax(generate_new=False)

        if xcat_data:
            print(f"\n Loaded XCAT thorax phantom")
            # Convert to expected format
            dataset = {
                'data': xcat_data['speed_volume'], # Use speed of sound for visualization
                'voxel_size': xcat_data['spacing'],
                'metadata': {
                    'description': 'XCAT Thorax Phantom',
                    'modality': 'CT',
                    'source': 'XCAT Phantom'
                },
                'anatomy_data': xcat_data,
                'filepath': 'anatomical_data/xcat_thorax_phantom.nii.gz'
            }
            datasets_to_process = [dataset]
        else:
            print("\n No XCAT data found, creating synthetic datasets...")
            # Create synthetic datasets as fallback
            synthetic_xcat = self.anatomy_loader.create_synthetic_data(shape=(128, 128, 128), mode='thorax')
            if synthetic_xcat:
                dataset = {
                    'data': synthetic_xcat['speed_volume'],
                    'voxel_size': synthetic_xcat['spacing'],
                    'metadata': {
                        'description': 'Synthetic XCAT Thorax',
                        'modality': 'CT',
                        'source': 'Synthetic'
                    },
                    'anatomy_data': synthetic_xcat,
                    'filepath': 'synthetic'
                }
                datasets_to_process = [dataset]
            else:
                datasets_to_process = []

        # Process each dataset
        for i, dataset in enumerate(datasets_to_process):
            print(f"\n{'='*60}")
            print(f" XACT Analysis {i+1}: {dataset['metadata']['description']}")
            print(f" Source: {dataset['metadata'].get('source', 'Synthetic')}")
            print(f" Modality: {dataset['metadata']['modality']}")
            print(f" File: {dataset.get('filepath', 'Generated')}")
            print('='*60)

            # Set current dataset
            self.current_dataset = dataset

            # Setup acoustic sensors first
            self.setup_acoustic_sensors(num_sensors=128)

            # Show 3D anatomy
            self.show_3d_anatomy()

            # Run XACT simulation
            self.run_xact_simulation()

            # Show results
            self.show_xact_results()

        print("\n XACT Demo Complete")
        
        # Finalize results session
        summary = f"Processed {len(datasets_to_process)} anatomical datasets with full XACT simulation"
        self.results_manager.finalize_run(summary)
        
        print(f"Results saved to: {self.current_session}")
        print("Generated files:")
        print(" - 3D visualization images")
        print(" - XACT simulation results")
        print(" - Acoustic waveform analysis")
        print(" - Audio files (WAV format)")
        print(f" - Processed {len(datasets_to_process)} anatomical datasets")

    def run_with_existing_data(self):
        """Run XACT demo specifically using existing anatomical data"""

        # Initialize results session
        self.current_session = self.results_manager.create_run_directory("xact_existing_data")
        print(f"\n Results will be saved to: {self.current_session}")
        
        print("\n XACT Demo with Existing Anatomical Data")
        print("=" * 50)

        # Load existing data
        existing_datasets = self.anatomy_loader.load_existing_anatomical_data()

        if not existing_datasets:
            print(" No existing anatomical data found in anatomical_data/ folder")
            print(" Please ensure you have .nii.gz or .npz files in the anatomical_data/ directory")
            return

        print(f"\n Found {len(existing_datasets)} existing datasets:")
        for i, dataset in enumerate(existing_datasets):
            print(f" {i+1}. {dataset['metadata']['description']}")
            print(f" Source: {dataset['metadata']['source']}")
            print(f" Modality: {dataset['metadata']['modality']}")
            print(f" Shape: {dataset['data'].shape}")
            print(f" File: {dataset['filepath']}")

        # Process each existing dataset
        for i, dataset in enumerate(existing_datasets):
            print(f"\n{'='*60}")
            print(f" XACT Analysis {i+1}: {dataset['metadata']['description']}")
            print(f" Source: {dataset['metadata']['source']}")
            print(f" Modality: {dataset['metadata']['modality']}")
            print(f" File: {dataset['filepath']}")
            print('='*60)

            # Set current dataset
            self.current_dataset = dataset

            # Setup acoustic sensors first
            self.setup_acoustic_sensors(num_sensors=128)

            # Show 3D anatomy
            self.show_3d_anatomy()

            # Run XACT simulation
            self.run_xact_simulation()

            # Show results
            self.show_xact_results()

        print("\n XACT Demo with Existing Data Complete")
        
        # Finalize results session
        summary = f"Processed {len(existing_datasets)} existing anatomical datasets with XACT analysis"
        self.results_manager.finalize_run(summary)
        
        print(f"Results saved to: {self.current_session}")
        print(f" Processed {len(existing_datasets)} existing anatomical datasets")
        print(" Generated visualization and analysis files")

    def show_3d_anatomy(self):
        """Display 3D anatomical model with sensor array and X-ray setup"""

        # Safety check for current_dataset
        if self.current_dataset is None:
            print(" No current dataset set. Cannot display 3D anatomy.")
            return

        data = self.current_dataset['data']
        metadata = self.current_dataset['metadata']

        # Check if data is valid
        if data is None or data.size == 0:
            print(" Dataset data is empty or None.")
            return

        # Additional safety checks for data integrity
        try:
            # Test if we can compute min/max safely
            if np.all(np.isnan(data)) or np.all(np.isinf(data)):
                print(" Dataset contains only NaN or infinite values.")
                return

            # Filter out NaN and infinite values for statistics
            valid_data = data[np.isfinite(data)]
            if valid_data.size == 0:
                print(" No finite values in dataset.")
                return

            data_min = np.min(valid_data)
            data_max = np.max(valid_data)

        except Exception as e:
            print(f" Error computing data statistics: {e}")
            return

        print(f" Displaying 3D pre-scan setup: {metadata['description']}")
        print(f" Data shape: {data.shape}")
        print(f" Data range: {data_min:.1f} to {data_max:.1f}")

        # Create 3D pre-scan visualization
        fig = plt.figure(figsize=(20, 12))

        # Main 3D view with anatomy, sensors, and X-ray setup
        ax_3d = fig.add_subplot(1, 2, 1, projection='3d')

        # Create 3D isosurface visualization of anatomy
        # Check if we have any non-zero data
        if np.all(data == 0):
            print(" All data values are zero. Cannot create meaningful visualization.")
            plt.close(fig)
            return

        # Use a more robust threshold calculation
        non_zero_data = data[data > 0]
        if len(non_zero_data) == 0:
            print(" No non-zero data found. Cannot create meaningful visualization.")
            plt.close(fig)
            return

        try:
            threshold = np.percentile(non_zero_data, 75)
        except Exception as e:
            print(f" Error computing threshold: {e}")
            threshold = np.min(non_zero_data)

        print(f" Using threshold: {threshold:.1f}")

        # Downsample for performance
        step = max(1, max(data.shape) // 30)
        x, y, z = np.meshgrid(
            np.arange(0, data.shape[0], step),
            np.arange(0, data.shape[1], step),
            np.arange(0, data.shape[2], step),
            indexing='ij'
        )

        data_sample = data[::step, ::step, ::step]
        mask = data_sample > threshold

        # Check if we have any data to plot
        if not np.any(mask):
            print(f" No data above threshold {threshold:.1f}. Trying lower threshold...")
            # Try a lower threshold
            try:
                threshold = np.percentile(non_zero_data, 25)
                mask = data_sample > threshold
                if not np.any(mask):
                    print(f" Still no data above threshold {threshold:.1f}. Using minimum non-zero value.")
                    threshold = np.min(non_zero_data)
                    mask = data_sample > threshold
            except Exception as e:
                print(f" Error adjusting threshold: {e}")
            plt.close(fig)
            return

        if not np.any(mask):
            print(" No data to visualize even with minimum threshold.")
            plt.close(fig)
            return

        print(f" Plotting {np.sum(mask)} data points")

        try:
            # Plot anatomy as semi-transparent scatter
            scatter_anatomy = ax_3d.scatter(x[mask], y[mask], z[mask], 
                c=data_sample[mask], cmap='viridis', 
                alpha=0.4, s=3, label='Anatomy')

            # Add sensor array if available
            if hasattr(self, 'sensors') and len(self.sensors) > 0:
                sensor_positions = np.array([s['position'] for s in self.sensors])
                scatter_sensors = ax_3d.scatter(sensor_positions[:, 0], 
                    sensor_positions[:, 1], 
                    sensor_positions[:, 2], 
                    c='red', s=50, alpha=0.8, 
                    label=f'Sensors ({len(self.sensors)})')

                # Draw sensor array circle
                angles = np.linspace(0, 2*np.pi, 100)
                radius = 150 # mm
                circle_x = radius * np.cos(angles)
                circle_y = radius * np.sin(angles)
                circle_z = np.zeros_like(angles)
                ax_3d.plot(circle_x, circle_y, circle_z, 'r--', alpha=0.6, linewidth=2)

                # Add X-ray source and cone
                xray_source = np.array([0, 0, -200]) # X-ray source position
                ax_3d.scatter([xray_source[0]], [xray_source[1]], [xray_source[2]], 
                    c='yellow', s=200, alpha=0.9, label='X-ray Source')

                # Draw X-ray cone
                cone_angles = np.linspace(0, 2*np.pi, 20)
                cone_radius = 100 # mm at anatomy level
                cone_height = 200 # mm from source to anatomy

                # Cone base at anatomy level
                cone_x = cone_radius * np.cos(cone_angles)
                cone_y = cone_radius * np.sin(cone_angles)
                cone_z = np.zeros_like(cone_angles)

                # Draw cone lines from source to base
                for i in range(len(cone_angles)):
                    ax_3d.plot([xray_source[0], cone_x[i]], 
                        [xray_source[1], cone_y[i]], 
                        [xray_source[2], cone_z[i]], 
                        'y-', alpha=0.6, linewidth=1)

                # Draw cone base circle
                ax_3d.plot(cone_x, cone_y, cone_z, 'y-', alpha=0.8, linewidth=2, label='X-ray Cone')

                # Add coordinate system
                origin = np.array([0, 0, 0])
                axis_length = 50
                ax_3d.quiver(origin[0], origin[1], origin[2], axis_length, 0, 0, color='red', alpha=0.7, label='X')
                ax_3d.quiver(origin[0], origin[1], origin[2], 0, axis_length, 0, color='green', alpha=0.7, label='Y')
                ax_3d.quiver(origin[0], origin[1], origin[2], 0, 0, axis_length, color='blue', alpha=0.7, label='Z')

                # Set view and labels
                ax_3d.set_title(f'XACT Pre-Scan Setup\n{metadata["description"]}', fontsize=14)
                ax_3d.set_xlabel('X (mm)')
                ax_3d.set_ylabel('Y (mm)')
                ax_3d.set_zlabel('Z (mm)')
                ax_3d.legend()

                # Set equal aspect ratio and view angle
                ax_3d.set_box_aspect([1, 1, 1])
                ax_3d.view_init(elev=20, azim=45)

                # Setup information panel
                ax_info = fig.add_subplot(1, 2, 2)
                ax_info.axis('off')

                # Create setup information text
                setup_info = f"""XACT Pre-Scan Setup Information

ANATOMICAL MODEL:
• Description: {metadata['description']}
• Modality: {metadata['modality']}
• Dimensions: {data.shape[0]} × {data.shape[1]} × {data.shape[2]} voxels
• Voxel Size: {self.current_dataset['voxel_size']} mm
• Volume: {data.shape[0] * data.shape[1] * data.shape[2]:,} voxels

SENSOR ARRAY:
• Configuration: Circular Array
• Number of Sensors: {len(self.sensors) if hasattr(self, 'sensors') else 'Not set'}
• Array Radius: 150 mm
• Angular Coverage: 360°
• Sensor Spacing: {360/len(self.sensors):.1f}° (if sensors set)

X-RAY SOURCE:
• Position: (0, 0, -200) mm
• Beam Type: Cone Beam
• Cone Angle: ~30°
• Beam Coverage: 100 mm radius at anatomy
• Energy: 120 keV (typical)

SCAN PARAMETERS:
• Projections: 36 (10° intervals)
• Scan Duration: {getattr(self, 'scan_params', {}).get('scan_duration', 0.001)*1e9:.1f} ns
• Sample Rate: {getattr(self, 'sample_rate', 44100)/1000:.1f} kHz

PHYSICS:
• Speed of Sound: 1540 m/s
• Acoustic Frequency: 2.5 MHz
• Tissue Types: 7 (Air, Fat, Soft, Muscle, Bone, Dense)
• Reconstruction: Filtered Back-Projection
 """

                ax_info.text(0.05, 0.95, setup_info, transform=ax_info.transAxes,
                    fontsize=11, family='monospace', verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9))

                plt.suptitle(f"XACT Pre-Scan Visualization: {metadata['description']}", fontsize=16)
                plt.tight_layout()

                # Save to results folder
                filename = f"3d_prescan_setup_{metadata['modality'].lower()}.png"
                filepath = self.current_session / "images" / "setup" / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close() # Close figure to free memory

                print(f" Saved pre-scan visualization: {filepath}")

        except Exception as e:
            print(f" Error creating 3D visualization: {e}")
            plt.close(fig)
            return

    def run_xact_simulation(self):
        """Run XACT simulation with acoustic sensor integration"""
        
        data = self.current_dataset['data']
        metadata = self.current_dataset['metadata']
        
        print(f" Running XACT simulation on {metadata['description']}...")
        
        # Convert to tissue map
        tissue_map = self._convert_to_tissue_map(data, metadata)
        
        # Setup simulation parameters
        num_projections = 36  # Faster with fewer projections
        angles = [(angle, 0.0) for angle in np.linspace(0, 2*np.pi, num_projections, endpoint=False)]
        
        print(f" {num_projections} X-ray projections")
        print(f" Simulating acoustic generation...")
        
        # Use XACT engine for simulation
        from xact_engine import simulate_xact_scan
        
        # Run simulation with enhanced physics
        simulation_results = simulate_xact_scan(
            anatomy_model=tissue_map,
            tissue_labels=self._get_tissue_labels(),
            beam_angles=angles,
            beam_energy=100,  # 100 keV
            beam_fluence=1e10,  # photons/cm²
            beam_width=0.02,  # 2cm beam width
            num_elements=len(self.sensors),
            detector_type='ring_array',
            frequency=2.5e6  # 2.5 MHz
        )
        
        # Store results
        self.time_signals = simulation_results['detected_signals']
        self.acoustic_signals = []
        
        # Process acoustic signals
        for i, signals in enumerate(self.time_signals):
            acoustic_info = {
                'sensor_id': i,
                'position': simulation_results['detector_positions'][i],
                'peak_amplitude': np.max(np.abs(signals)),
                'rms_amplitude': np.sqrt(np.mean(signals**2)),
                'signal_energy': np.sum(signals**2),
                'dominant_freq': self._find_dominant_frequency(signals)
            }
            self.acoustic_signals.append(acoustic_info)
            
        # Set time axis for plotting
        if len(self.time_signals) > 0:
            time_points = len(self.time_signals[0])
            self.time_axis = np.linspace(0, self.scan_params['scan_duration'], time_points)
            
        print(" XACT simulation completed successfully")
        
    def _convert_to_tissue_map(self, data: np.ndarray, metadata: Dict) -> np.ndarray:
        """Convert medical image to tissue property map"""
        # Simple thresholding for tissue types
        tissue_map = np.zeros_like(data, dtype=np.uint8)
        
        if metadata['modality'] == 'CT':
            # Convert HU to tissue types
            tissue_map[data <= -1000] = 1  # Air
            tissue_map[(data > -1000) & (data <= -100)] = 2  # Fat
            tissue_map[(data > -100) & (data <= 100)] = 3  # Soft tissue
            tissue_map[(data > 100) & (data <= 300)] = 4  # Muscle
            tissue_map[(data > 300) & (data <= 1000)] = 5  # Bone
            tissue_map[data > 1000] = 6  # Dense bone
        else:
            # Simple intensity-based segmentation
            max_val = np.max(data)
            tissue_map[data <= 0.1 * max_val] = 1
            tissue_map[(data > 0.1 * max_val) & (data <= 0.3 * max_val)] = 2
            tissue_map[(data > 0.3 * max_val) & (data <= 0.5 * max_val)] = 3
            tissue_map[(data > 0.5 * max_val) & (data <= 0.7 * max_val)] = 4
            tissue_map[(data > 0.7 * max_val) & (data <= 0.9 * max_val)] = 5
            tissue_map[data > 0.9 * max_val] = 6
            
        return tissue_map
    
    def _find_dominant_frequency(self, signal: np.ndarray) -> float:
        """Find dominant frequency in signal using FFT"""
        if len(signal) < 2:
            return 0.0
            
        try:
            # Compute FFT
            fft_vals = np.abs(fft(signal))
            freqs = fftfreq(len(signal), 1.0/self.scan_params['sample_rate'])
            
            # Find peak in positive frequencies
            pos_mask = freqs > 0
            peak_idx = np.argmax(fft_vals[pos_mask])
            
            return freqs[pos_mask][peak_idx]
        except Exception as e:
            print(f"Warning: Error computing dominant frequency: {e}")
            return 0.0
            
    def _get_tissue_labels(self) -> Dict[int, str]:
        """Map tissue IDs to tissue names for XACT engine"""
        return {
            0: 'air',      # Background
            1: 'air',      # Air
            2: 'fat',      # Fat
            3: 'muscle',   # Soft tissue
            4: 'muscle',   # Muscle
            5: 'bone_cortical',    # Bone
            6: 'bone_trabecular'   # Dense bone
        }

    def show_xact_results(self):
        """Display simplified XACT simulation results with key acoustic data"""

        metadata = self.current_dataset['metadata']
        xact_results = self.current_dataset['xact_results']

        print(f" Displaying XACT results for {metadata['description']}")

        tissue_map = xact_results['tissue_map']
        reconstruction = xact_results['reconstruction']

        # Create simplified results visualization
        fig = plt.figure(figsize=(16, 12))

        # Row 1: Anatomy and Reconstruction (3 panels)
        mid_z = tissue_map.shape[2] // 2
        mid_y = tissue_map.shape[1] // 2

        # Original anatomy - axial
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.imshow(tissue_map[:, :, mid_z].T, cmap='tab10', origin='lower')
        ax1.set_title('Original Anatomy\n(Axial View)')
        ax1.set_xticks([])
        ax1.set_yticks([])

        # XACT reconstruction - axial
        ax2 = fig.add_subplot(2, 3, 2)
        im_recon = ax2.imshow(reconstruction[:, :, mid_z].T, cmap='hot', origin='lower')
        ax2.set_title('XACT Reconstruction\n(Axial View)')
        ax2.set_xticks([])
        ax2.set_yticks([])
        cbar = plt.colorbar(im_recon, ax=ax2, shrink=0.8)
        cbar.set_label('Acoustic Source Strength (Pa)', rotation=270, labelpad=20)

        # Sensor array layout
        ax3 = fig.add_subplot(2, 3, 3)
        sensor_positions = np.array([s['position'] for s in self.sensors])
        scatter = ax3.scatter(sensor_positions[:, 0], sensor_positions[:, 1], 
            c=range(len(self.sensors)), cmap='viridis', s=30)
        ax3.set_title(f'Sensor Array Layout\n({len(self.sensors)} Sensors)')
        ax3.set_xlabel('X (mm)')
        ax3.set_ylabel('Y (mm)')
        ax3.set_aspect('equal')

        # Row 2: Key acoustic data (3 panels)

        # Time-domain signals (representative sensors)
        ax4 = fig.add_subplot(2, 3, 4)

        # Convert time to appropriate units with error handling
        if self.time_axis is not None and len(self.time_axis) > 0:
            if self.time_axis[-1] < 1e-6: # Less than 1 microsecond
                time_display = self.time_axis * 1e9 # Convert to nanoseconds
                time_unit = 'ns'
            elif self.time_axis[-1] < 1e-3: # Less than 1 millisecond
                time_display = self.time_axis * 1e6 # Convert to microseconds
                time_unit = 'μs'
            else:
                time_display = self.time_axis * 1000 # Convert to milliseconds
                time_unit = 'ms'
        else:
            # Create a default time axis if none exists
            time_points = len(self.time_signals[0]) if len(self.time_signals) > 0 else 1000
            time_display = np.linspace(0, 1000, time_points) # Default 1000 ns
            time_unit = 'ns'
            print("Warning: No time axis found, using default time scale")

        # Show representative sensors (every 32nd for clarity)
        for i in range(0, len(self.time_signals), 32):
            ax4.plot(time_display, self.time_signals[i], alpha=0.8, linewidth=1.5, 
                label=f'Sensor {i}' if i < 128 else '')

        ax4.set_title(f'Acoustic Time Signals\n(Representative Sensors)')
        ax4.set_xlabel(f'Time ({time_unit})')
        ax4.set_ylabel('Amplitude (Pa)')
        ax4.grid(True, alpha=0.3)
        if len(self.time_signals) <= 128:
            ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        # Peak amplitude distribution
        ax5 = fig.add_subplot(2, 3, 5)
        peak_amplitudes = [sig['peak_amplitude'] for sig in self.acoustic_signals]

        # Robust peak amplitude handling
        if len(peak_amplitudes) > 0:
            try:
                # Filter finite values
                finite_peaks = [p for p in peak_amplitudes if np.isfinite(p)]
                if len(finite_peaks) > 0:
                    ax5.plot(range(len(peak_amplitudes)), peak_amplitudes, 'b-', linewidth=2)
                    ax5.set_title('Peak Amplitude\nvs Sensor Position')
                    ax5.set_xlabel('Sensor Index')
                    ax5.set_ylabel('Peak Amplitude (Pa)')
                    ax5.grid(True, alpha=0.3)

                    # Add statistics text
                    try:
                        min_peak = np.min(finite_peaks)
                        max_peak = np.max(finite_peaks)
                        mean_peak = np.mean(finite_peaks)
                        stats_text = f'Min: {min_peak:.3f} Pa\nMax: {max_peak:.3f} Pa\nMean: {mean_peak:.3f} Pa'
                        ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, 
                            verticalalignment='top', fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                    except Exception as e:
                        print(f"Warning: Error computing peak amplitude statistics: {e}")
                else:
                    ax5.text(0.5, 0.5, 'No valid peak amplitudes', 
                        transform=ax5.transAxes, ha='center', va='center')
                    ax5.set_title('Peak Amplitude\nvs Sensor Position')
                    ax5.set_ylabel('Peak Amplitude (Pa)')
            except Exception as e:
                print(f"Warning: Error plotting peak amplitudes: {e}")
                ax5.text(0.5, 0.5, f'Error plotting: {e}', 
                    transform=ax5.transAxes, ha='center', va='center')
                ax5.set_title('Peak Amplitude\nvs Sensor Position')
                ax5.set_ylabel('Peak Amplitude (Pa)')
        else:
            ax5.text(0.5, 0.5, 'No acoustic signals available', 
                transform=ax5.transAxes, ha='center', va='center')
            ax5.set_title('Peak Amplitude\nvs Sensor Position')
            ax5.set_ylabel('Peak Amplitude (Pa)')

        # Polar plot of signal strength
        ax6 = fig.add_subplot(2, 3, 6, projection='polar')
        sensor_angles = [s['angle'] for s in self.sensors]
        ax6.plot(sensor_angles, peak_amplitudes, 'ro-', markersize=4, linewidth=2)
        ax6.set_title('Signal Strength\n(Polar Distribution)')
        ax6.set_ylabel('Amplitude (Pa)', labelpad=30) # Add padding for polar plot

        # Add summary statistics
        original_data = self.current_dataset['data']
        correlation = np.corrcoef(original_data.flatten(), reconstruction.flatten())[0, 1]

        # Robust statistics computation
        try:
            if len(time_display) > 0:
                signal_duration = time_display[-1]
            else:
                signal_duration = 0.0

            if len(peak_amplitudes) > 0:
                finite_peaks = [p for p in peak_amplitudes if np.isfinite(p)]
                if len(finite_peaks) > 0:
                    peak_range = f"{np.min(finite_peaks):.3f} - {np.max(finite_peaks):.3f}"
                else:
                    peak_range = "No valid data"
            else:
                peak_range = "No data"
        except Exception as e:
            print(f"Warning: Error computing statistics: {e}")
            signal_duration = 0.0
            peak_range = "Error"

        stats_text = f"""XACT Simulation Summary:

RECONSTRUCTION:
• Correlation with Original: {correlation:.3f}
• Reconstruction Quality: {'Excellent' if correlation > 0.8 else 'Good' if correlation > 0.6 else 'Fair'}

ACOUSTIC DATA:
• Sensors: {len(self.sensors)}
• Signal Duration: {signal_duration:.1f} {time_unit}
• Sample Rate: {self.sample_rate/1000:.1f} kHz
• Peak Amplitude Range: {peak_range}

PHYSICS:
• Speed of Sound: 1540 m/s
• Center Frequency: 2.5 MHz
• Tissue Types Detected: {len(np.unique(tissue_map))}
 """

        fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9))

        plt.suptitle(f'XACT Simulation Results: {metadata["description"]}', fontsize=16)
        plt.tight_layout()

        # Save results to results folder
        filename = f"xact_simulation_results_{metadata['modality'].lower()}.png"
        filepath = self.current_session / "images" / "results" / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close() # Close figure to free memory

        print(f" Saved XACT results: {filepath}")
        print(f" Reconstruction correlation: {correlation:.3f}")
        print(f" Acoustic sensors: {len(self.sensors)}")

        # Export audio files if possible
        if AUDIO_AVAILABLE and len(self.time_signals) > 0:
            self._export_audio_files()

    def _simulate_beamforming(self) -> np.ndarray:
        """Simulate delay-and-sum beamforming"""

        if len(self.time_signals) == 0:
            return np.zeros(100)

        sound_speed = 1540 # m/s
        focus_point = np.array([0, 0, 0]) # Focus at center

        # Calculate delays for each sensor
        delays = []
        for sensor in self.sensors:
            distance = np.linalg.norm(sensor['position'] - focus_point)
            delay = distance / (sound_speed * 1000) # Convert to seconds
            delays.append(delay)

        # Apply delays and sum
        beamformed_signal = np.zeros(len(self.time_axis))

        for i, (signal, delay) in enumerate(zip(self.time_signals, delays)):
            # Convert delay to samples
            delay_samples = int(delay * self.sample_rate)

            # Apply delay (simple shift)
            if delay_samples < len(signal):
                delayed_signal = np.zeros_like(signal)
                delayed_signal[delay_samples:] = signal[:-delay_samples]
                beamformed_signal += delayed_signal / len(self.sensors)

        return beamformed_signal

    def _export_audio_files(self):
        """Export acoustic sensor signals as audio files"""

        if not hasattr(self, 'time_signals') or len(self.time_signals) == 0:
            print(" No time signals available for export")
            return

        print(" Exporting acoustic sensor audio files...")

        # Export individual sensor files (sample of 8 sensors)
        sensor_indices = np.linspace(0, len(self.time_signals)-1, 8, dtype=int)

        for i, sensor_idx in enumerate(sensor_indices):
            signal = self.time_signals[sensor_idx]

            # Robust normalization with error handling
            try:
                if signal.size > 0 and np.any(np.isfinite(signal)):
                    finite_signal = signal[np.isfinite(signal)]
                    if finite_signal.size > 0:
                        max_amplitude = np.max(np.abs(finite_signal))
                        if max_amplitude > 0:
                            normalized_signal = signal / max_amplitude * 0.8
                        else:
                            normalized_signal = signal
                    else:
                        normalized_signal = np.zeros_like(signal)
                else:
                    normalized_signal = np.zeros_like(signal)
            except Exception as e:
                print(f"Warning: Error normalizing sensor {sensor_idx} signal: {e}")
                normalized_signal = np.zeros_like(signal)

            filename = f"sensor_{sensor_idx:03d}.wav"
            filepath = self.current_session / "audio" / "sensors" / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)

            try:
                # Use scipy.io.wavfile as fallback if soundfile not available
                if 'soundfile' in sys.modules:
                    import soundfile as sf
                    sf.write(filepath, normalized_signal, self.sample_rate)
                else:
                    from scipy.io import wavfile
                    # Ensure signal is in proper format for wavfile
                    if normalized_signal.dtype != np.int16:
                        # Convert to 16-bit integer
                        normalized_signal = (normalized_signal * 32767).astype(np.int16)
                    wavfile.write(filepath, self.sample_rate, normalized_signal)
            except Exception as e:
                print(f"Warning: Could not export {filename}: {e}")

        # Export mixed signal
        try:
            # Combine all signals with robust error handling
            mixed_signal = np.zeros(len(self.time_signals[0]))
            valid_signals = 0

            for signal in self.time_signals:
                if signal.size > 0 and np.any(np.isfinite(signal)):
                    mixed_signal += signal
                    valid_signals += 1

            if valid_signals > 0:
                mixed_signal /= valid_signals

                # Robust normalization
                if mixed_signal.size > 0 and np.any(np.isfinite(mixed_signal)):
                    finite_mixed = mixed_signal[np.isfinite(mixed_signal)]
                    if finite_mixed.size > 0:
                        max_amplitude = np.max(np.abs(finite_mixed))
                        if max_amplitude > 0:
                            normalized_mixed = mixed_signal / max_amplitude * 0.8
                        else:
                            normalized_mixed = mixed_signal
                    else:
                        normalized_mixed = np.zeros_like(mixed_signal)
                else:
                    normalized_mixed = np.zeros_like(mixed_signal)

                # Export mixed file to results folder
                mixed_filename = "all_sensors_mixed.wav"
                mixed_filepath = self.current_session / "audio" / "mixed" / mixed_filename
                mixed_filepath.parent.mkdir(parents=True, exist_ok=True)
                try:
                    if 'soundfile' in sys.modules:
                        import soundfile as sf
                        sf.write(mixed_filepath, normalized_mixed, self.sample_rate)
                    else:
                        from scipy.io import wavfile
                        if normalized_mixed.dtype != np.int16:
                            normalized_mixed = (normalized_mixed * 32767).astype(np.int16)
                        wavfile.write(mixed_filepath, self.sample_rate, normalized_mixed)
                except Exception as e:
                    print(f"Warning: Could not export mixed audio: {e}")

        except Exception as e:
            print(f"Warning: Error creating mixed signal: {e}")

        print(f" Exported {len(sensor_indices)} sensor audio files")
        print(" Exported all_sensors_mixed.wav")

    def _display_detailed_waveforms(self):
        """Display simplified acoustic waveform analysis"""

        print("\n Displaying acoustic waveform analysis...")

        # Create simplified waveform visualization
        fig = plt.figure(figsize=(16, 10))

        # Convert time to appropriate units
        if self.time_axis[-1] < 1e-6: # Less than 1 microsecond
            time_display = self.time_axis * 1e9 # Convert to nanoseconds
            time_unit = 'ns'
        elif self.time_axis[-1] < 1e-3: # Less than 1 millisecond
            time_display = self.time_axis * 1e6 # Convert to microseconds
            time_unit = 'μs'
        else:
            time_display = self.time_axis * 1000 # Convert to milliseconds
            time_unit = 'ms'

        # Top: Representative sensor signals
        ax1 = fig.add_subplot(2, 2, 1)
        for i in range(0, min(8, len(self.time_signals))):
            ax1.plot(time_display, self.time_signals[i], alpha=0.8, linewidth=1.5, 
                label=f'Sensor {i}')
        ax1.set_title('Representative Sensor Signals\n(First 8 Sensors)')
        ax1.set_xlabel(f'Time ({time_unit})')
        ax1.set_ylabel('Amplitude (Pa)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Mixed signal
        ax2 = fig.add_subplot(2, 2, 2)
        mixed_signal = np.mean(self.time_signals, axis=0)
        ax2.plot(time_display, mixed_signal, 'red', linewidth=2, label='Mixed Signal')
        ax2.set_title('Mixed Signal Waveform\n(All Sensors)')
        ax2.set_xlabel(f'Time ({time_unit})')
        ax2.set_ylabel('Amplitude (Pa)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Frequency spectrum
        ax3 = fig.add_subplot(2, 2, 3)
        mixed_fft = fft(mixed_signal)
        freqs = fftfreq(len(mixed_signal), 1/self.sample_rate)
        positive_freqs = freqs[:len(freqs)//2]
        magnitude = np.abs(mixed_fft[:len(mixed_fft)//2])

        ax3.plot(positive_freqs/1e6, magnitude)
        ax3.set_title('Frequency Spectrum\n(Mixed Signal)')
        ax3.set_xlabel('Frequency (MHz)')
        ax3.set_ylabel('Magnitude (Pa·Hz)')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 10)

        # Peak amplitude distribution
        ax4 = fig.add_subplot(2, 2, 4)
        peak_amplitudes = [sig['peak_amplitude'] for sig in self.acoustic_signals]
        ax4.hist(peak_amplitudes, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.set_title('Peak Amplitude Distribution')
        ax4.set_xlabel('Peak Amplitude (Pa)')
        ax4.set_ylabel('Count (Sensors)')
        ax4.grid(True, alpha=0.3)

        # Add summary info
        stats_text = f"""Acoustic Analysis Summary:

SIGNAL CHARACTERISTICS:
• Duration: {time_display[-1]:.1f} {time_unit}
• Sample Rate: {self.sample_rate/1000:.1f} kHz
• Sensors: {len(self.sensors)}

AMPLITUDE STATISTICS:
• Max Amplitude: {np.max(peak_amplitudes):.3f}
• Mean Amplitude: {np.mean(peak_amplitudes):.3f}
• Amplitude Range: {np.min(peak_amplitudes):.3f} - {np.max(peak_amplitudes):.3f}

AUDIO FILES:
• all_sensors_mixed.wav
• sensor_000-007_acoustic.wav
 """

        fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

        plt.suptitle('XACT Acoustic Waveform Analysis', fontsize=16)
        plt.tight_layout()

        # Save waveform analysis to results folder
        waveform_filename = 'acoustic_waveform_analysis.png'
        filepath = self.current_session / "images" / "analysis" / waveform_filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close() # Close figure to free memory

        print(f" Saved waveform analysis: {filepath}")

        # Also create a simplified audio waveform display
        self._display_audio_waveforms()

    def _display_audio_waveforms(self):
        """Display audio-style waveform visualization"""

        print("\n Displaying audio waveform view...")

        # Create audio-style visualization
        fig, axes = plt.subplots(3, 1, figsize=(16, 10))

        time_ms = self.time_axis * 1000

        # Top: Raw mixed signal
        mixed_signal = np.mean(self.time_signals, axis=0)
        axes[0].plot(time_ms, mixed_signal, 'blue', linewidth=1)
        axes[0].set_title('Mixed Audio Signal (All 128 Sensors)', fontsize=14)
        axes[0].set_ylabel('Amplitude (Pa)')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0, time_ms[-1])

        # Middle: Stereo-style display (left/right channel simulation)
        left_sensors = self.time_signals[:len(self.time_signals)//2]
        right_sensors = self.time_signals[len(self.time_signals)//2:]

        left_channel = np.mean(left_sensors, axis=0)
        right_channel = np.mean(right_sensors, axis=0)

        axes[1].plot(time_ms, left_channel, 'red', linewidth=1, label='Left Sensors (0-63)')
        axes[1].plot(time_ms, right_channel, 'green', linewidth=1, label='Right Sensors (64-127)')
        axes[1].set_title('Stereo Channel Simulation', fontsize=14)
        axes[1].set_ylabel('Amplitude (Pa)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(0, time_ms[-1])

        # Bottom: Normalized for audio playback
        normalized_mixed = mixed_signal / (np.max(np.abs(mixed_signal)) + 1e-12) * 0.8

        axes[2].plot(time_ms, normalized_mixed, 'purple', linewidth=1)
        axes[2].set_title('Normalized Audio Signal ( for Playback)', fontsize=14)
        axes[2].set_xlabel('Time (ms)')
        axes[2].set_ylabel('Normalized Amplitude (-1 to +1)')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlim(0, time_ms[-1])
        axes[2].set_ylim(-1, 1)

        # Add audio info
        fig.text(0.02, 0.02, 
            f'Audio Info: {self.sample_rate/1000:.1f} kHz, {len(mixed_signal)} samples, '
            f'{time_ms[-1]:.1f} ms duration\n'
            f'Files: all_sensors_mixed.wav, sensor_000-007_acoustic.wav',
            fontsize=10, family='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

        plt.tight_layout()

        # Save audio waveform to results folder
        audio_filename = 'audio_waveforms.png'
        filepath = self.current_session / "images" / "audio" / audio_filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close() # Close figure to free memory

        print(f" Saved audio waveforms: {filepath}")
        print("\n✨ Waveform analysis complete!")
        print("\n Audio files available:")
        print(" - all_sensors_mixed.wav (mixed signal from all 128 sensors)")

        for i in range(min(8, len(self.time_signals))):
            print(f" - sensor_{i:03d}_acoustic.wav (individual sensor)")

    def demonstrate_real_data_integration(self):
        """Demonstrate integration with anatomical data sources"""

        print("\n Anatomical Data Integration Demo")
        print("=" * 50)

        # List available datasets
        self.anatomy_loader.list_available_datasets()

        # Try to download and load datasets
        print("\n Attempting to download anatomical data...")

        real_datasets = []
        synthetic_datasets = []

        for dataset_name in self.anatomy_loader.datasets.keys():
            print(f"\n Processing {dataset_name}...")

            # Try to download
            if self.anatomy_loader.download_dataset(dataset_name):
                # Try to load
                dataset = self.anatomy_loader.load_dataset(dataset_name)
                if dataset:
                    real_datasets.append(dataset)
                    print(f" Loaded data: {dataset_name}")
                else:
                    print(f"⚠️ Download succeeded but load failed: {dataset_name}")
                    # Create synthetic fallback
                    if 'brain' in dataset_name or 'head' in dataset_name:
                        synthetic_data = self.anatomy_loader.create_realistic_brain_mri()
                    elif 'thorax' in dataset_name:
                        synthetic_data = self.anatomy_loader.create_realistic_thorax_ct()
                    else:
                        synthetic_data = self.anatomy_loader.create_realistic_brain_mri()
                    synthetic_datasets.append(synthetic_data)
                    print(f" Using synthetic fallback for {dataset_name}")
            else:
                print(f" Download failed: {dataset_name}")
                # Create synthetic fallback
                if 'brain' in dataset_name or 'head' in dataset_name:
                    synthetic_data = self.anatomy_loader.create_realistic_brain_mri()
                elif 'thorax' in dataset_name:
                    synthetic_data = self.anatomy_loader.create_realistic_thorax_ct()
                else:
                    synthetic_data = self.anatomy_loader.create_realistic_brain_mri()
                synthetic_datasets.append(synthetic_data)
                print(f" Using synthetic fallback for {dataset_name}")

        # Summary
        print(f"\n Data Integration Summary:")
        print(f" datasets loaded: {len(real_datasets)}")
        print(f" Synthetic datasets created: {len(synthetic_datasets)}")
        print(f" Total datasets available: {len(real_datasets) + len(synthetic_datasets)}")

        # Show sample of data if available
        if real_datasets:
            print(f"\n Sample Dataset:")
            sample_dataset = real_datasets[0]
            print(f" Description: {sample_dataset['metadata']['description']}")
            print(f" Source: {sample_dataset['metadata'].get('source', 'Unknown')}")
            print(f" Modality: {sample_dataset['metadata']['modality']}")
            print(f" Shape: {sample_dataset['data'].shape}")
            print(f" Voxel size: {sample_dataset['voxel_size']} mm")
            print(f" Data range: {np.min(sample_dataset['data']):.1f} to {np.max(sample_dataset['data']):.1f}")

        # Show sample of synthetic data
        if synthetic_datasets:
            print(f"\n Sample Synthetic Dataset:")
            sample_dataset = synthetic_datasets[0]
            print(f" Description: {sample_dataset['metadata']['description']}")
            print(f" Source: Synthetic (based on parameters)")
            print(f" Modality: {sample_dataset['metadata']['modality']}")
            print(f" Shape: {sample_dataset['data'].shape}")
            print(f" Voxel size: {sample_dataset['voxel_size']} mm")
            print(f" Data range: {np.min(sample_dataset['data']):.1f} to {np.max(sample_dataset['data']):.1f}")

        print(f"\n data integration demonstration complete!")
        print(f" The system can now work with both and synthetic anatomical data")
        print(f" data sources: MNI, Cancer Imaging Archive, NITRC")
        print(f" Fallback: High-quality synthetic data based on parameters")

        return real_datasets + synthetic_datasets

def main():
    """Run the XACT demo"""

    # Create demo
    demo = XACTDemo()

    # Check if we have existing anatomical data
    existing_data = demo.anatomy_loader.load_existing_anatomical_data()

    if existing_data:
        print(f"\n Found {len(existing_data)} existing anatomical datasets")
        print(" Using existing data for XACT simulation...")
        demo.run_with_existing_data()
    else:
        print(f"\n No existing data found, running full demo with synthetic data...")
        demo.run_full_demo()

if __name__ == "__main__":
    main() 