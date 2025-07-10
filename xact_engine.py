"""
X-ray-induced Acoustic Computed Tomography (XACT) Engine

Based on:
"X-ray-induced acoustic computed tomography and its applications in biomedicine"
by Yuchen Yan and Shawn (Liangzhong) Xiang

Key Equations:
1. Acoustic Wave Equation: (∇²-1/vs² ∂²/∂t²)p(r,t) = -β/Cp ∂H(r,t)/∂t
2. Initial Pressure: p0 = Γ × ηth × μ × F
3. Gruneisen Parameter: Γ = β × vs²/Cp
"""

import numpy as np
from scipy.signal import butter, filtfilt
from dataclasses import dataclass
from typing import Tuple, Dict, List
import warnings

@dataclass
class TissueProperties:
    """Physical properties of biological tissues for XACT simulation"""
    density: float  # kg/m³
    sound_speed: float  # m/s
    attenuation: float  # dB/cm/MHz
    thermal_expansion_coeff: float  # K⁻¹
    specific_heat: float  # J/(kg·K)
    x_ray_absorption_coeff: float  # cm⁻¹
    thermal_efficiency: float  # dimensionless
    gruneisen_parameter: float  # dimensionless

# Standard tissue properties database
TISSUE_DATABASE = {
    'air': TissueProperties(1.2, 343, 0.0, 3.43e-3, 1005, 0.0001, 0.1, 0.0),
    'lung': TissueProperties(300, 600, 0.6, 3.0e-3, 3600, 0.02, 0.8, 0.2),
    'fat': TissueProperties(950, 1450, 0.6, 7.0e-4, 2300, 0.18, 0.85, 0.9),
    'muscle': TissueProperties(1050, 1580, 0.5, 3.0e-4, 3600, 0.19, 0.9, 0.85),
    'liver': TissueProperties(1060, 1570, 0.5, 3.0e-4, 3600, 0.20, 0.9, 0.85),
    'kidney': TissueProperties(1050, 1560, 0.5, 3.0e-4, 3900, 0.20, 0.9, 0.85),
    'bone_cortical': TissueProperties(1900, 4080, 13.0, 3.0e-5, 1300, 2.3, 0.9, 0.4),
    'bone_trabecular': TissueProperties(1180, 1800, 9.0, 8.0e-5, 2200, 0.5, 0.9, 0.6),
    'brain_gray': TissueProperties(1040, 1540, 0.6, 3.2e-4, 3600, 0.18, 0.9, 0.83),
    'brain_white': TissueProperties(1040, 1540, 0.6, 3.2e-4, 3600, 0.18, 0.9, 0.83),
    'blood': TissueProperties(1060, 1560, 0.2, 3.0e-4, 3900, 0.19, 0.95, 0.85),
    'skin': TissueProperties(1100, 1540, 0.8, 3.0e-4, 3600, 0.19, 0.85, 0.85),
    'water': TissueProperties(1000, 1500, 0.0022, 2.14e-4, 4180, 0.02, 1.0, 0.11),
    'contrast_agent': TissueProperties(1200, 1600, 0.3, 5.0e-4, 3000, 5.0, 0.95, 1.2)
}

class XRayBeam:
    """X-ray beam simulation with realistic energy spectrum"""
    
    def __init__(self, energy_kV: float = 120, filtration: str = 'Al_2.5mm'):
        self.energy_kV = energy_kV
        self.filtration = filtration
        self.spectrum = self._generate_spectrum()
        
    def _generate_spectrum(self) -> np.ndarray:
        """Generate realistic X-ray energy spectrum"""
        energies = np.linspace(1, self.energy_kV, int(self.energy_kV))
        # Simplified Kramers' law for bremsstrahlung
        spectrum = energies * (self.energy_kV - energies)
        
        # Add characteristic K-edges for tungsten anode
        if self.energy_kV > 69.5:  # K-alpha edge
            spectrum += 0.3 * np.exp(-((energies - 59.3) / 2.0)**2)
            spectrum += 0.2 * np.exp(-((energies - 67.2) / 2.0)**2)
            
        # Apply filtration attenuation
        if 'Al' in self.filtration:
            thickness_mm = float(self.filtration.split('_')[1].replace('mm', ''))
            mu_al = 0.18 * (energies / 100)**(-3)  # Aluminum attenuation
            spectrum *= np.exp(-mu_al * thickness_mm / 10)
            
        return spectrum / np.sum(spectrum)

class XACTPhysicsEngine:
    """Core physics engine for XACT simulation"""
    
    def __init__(self, grid_size: Tuple[int, int, int], 
                 voxel_size: float = 0.5e-3,  # 0.5mm voxels
                 time_step: float = 1e-8):    # 10ns time steps
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.time_step = time_step
        self.c_max = 4500  # Maximum sound speed for stability
        
        # Initialize grids
        self._initialize_grids()
        
    def _initialize_grids(self):
        """Initialize computational grids"""
        self.tissue_map = np.zeros(self.grid_size, dtype=np.uint8)
        self.density = np.zeros(self.grid_size, dtype=np.float32)
        self.sound_speed = np.zeros(self.grid_size, dtype=np.float32)
        self.absorption_coeff = np.zeros(self.grid_size, dtype=np.float32)
        self.gruneisen = np.zeros(self.grid_size, dtype=np.float32)
        self.thermal_efficiency = np.zeros(self.grid_size, dtype=np.float32)
        
        # Pressure fields for wave simulation
        self.pressure_current = np.zeros(self.grid_size, dtype=np.float32)
        self.pressure_previous = np.zeros(self.grid_size, dtype=np.float32)
        self.pressure_next = np.zeros(self.grid_size, dtype=np.float32)
        
    def set_tissue_properties(self, tissue_map: np.ndarray, tissue_labels: Dict[int, str]):
        """Set tissue properties from segmented anatomical model"""
        self.tissue_map = tissue_map.astype(np.uint8)
        
        for label, tissue_name in tissue_labels.items():
            if tissue_name in TISSUE_DATABASE:
                props = TISSUE_DATABASE[tissue_name]
                mask = (tissue_map == label)
                
                self.density[mask] = props.density
                self.sound_speed[mask] = props.sound_speed
                self.absorption_coeff[mask] = props.x_ray_absorption_coeff
                self.gruneisen[mask] = props.gruneisen_parameter
                self.thermal_efficiency[mask] = props.thermal_efficiency
    
    def _calculate_xray_attenuation(self, source_pos: Tuple[float, float, float],
                                  beam_direction: Tuple[float, float, float],
                                  beam_width: float) -> np.ndarray:
        """Calculate X-ray energy deposition using Beer-Lambert law"""
        nx, ny, nz = self.grid_size
        energy_deposition = np.zeros((nx, ny, nz), dtype=np.float32)
        
        # Normalize beam direction
        dx, dy, dz = beam_direction
        norm = np.sqrt(dx*dx + dy*dy + dz*dz)
        dx, dy, dz = dx/norm, dy/norm, dz/norm
        
        # Calculate energy deposition for each voxel
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Voxel position
                    x = i * self.voxel_size
                    y = j * self.voxel_size
                    z = k * self.voxel_size
                    
                    # Distance from beam center
                    dist_from_beam = self._point_to_line_distance(
                        (x, y, z), source_pos, (dx, dy, dz))
                    
                    if dist_from_beam <= beam_width:
                        # Get tissue properties
                        mu = self.absorption_coeff[i, j, k]  # X-ray absorption coefficient
                        rho = self.density[i, j, k]  # Tissue density
                        
                        if mu > 0 and rho > 0:
                            # Calculate path length through tissue
                            path_length = self._calculate_path_length(
                                (x, y, z), source_pos, (dx, dy, dz))
                            
                            # Beer-Lambert attenuation
                            attenuation = np.exp(-mu * path_length)
                            
                            # Gaussian beam profile
                            beam_profile = np.exp(-2 * (dist_from_beam/beam_width)**2)
                            
                            # Energy deposition considering density
                            energy_deposition[i, j, k] = (
                                mu * rho * attenuation * beam_profile
                            )
        
        return energy_deposition
    
    def _calculate_path_length(self, point: Tuple[float, float, float],
                             source: Tuple[float, float, float],
                             direction: Tuple[float, float, float]) -> float:
        """Calculate path length from source to point along beam direction"""
        # Vector from source to point
        vx = point[0] - source[0]
        vy = point[1] - source[1]
        vz = point[2] - source[2]
        
        # Project onto beam direction
        return abs(vx*direction[0] + vy*direction[1] + vz*direction[2])
    
    @staticmethod
    def _point_to_line_distance(point: Tuple[float, float, float],
                               line_point: Tuple[float, float, float],
                               line_direction: Tuple[float, float, float]) -> float:
        """Calculate distance from point to line"""
        px, py, pz = point
        lx, ly, lz = line_point
        dx, dy, dz = line_direction
        
        # Vector from line point to target point
        vx, vy, vz = px - lx, py - ly, pz - lz
        
        # Cross product magnitude
        cross_x = vy * dz - vz * dy
        cross_y = vz * dx - vx * dz
        cross_z = vx * dy - vy * dx
        
        return np.sqrt(cross_x*cross_x + cross_y*cross_y + cross_z*cross_z)
    
    def generate_initial_pressure(self, xray_energy: np.ndarray, 
                                beam_fluence: float = 1e10) -> np.ndarray:
        """Generate initial acoustic pressure from X-ray absorption"""
        initial_pressure = (self.gruneisen * 
                          self.thermal_efficiency * 
                          self.absorption_coeff * 
                          xray_energy * 
                          beam_fluence)
        
        return initial_pressure.astype(np.float32)
    
    def propagate_acoustic_wave(self, initial_pressure: np.ndarray, 
                               time_steps: int = 1000) -> List[np.ndarray]:
        """Propagate acoustic waves using finite difference method"""
        # Initialize pressure fields
        self.pressure_current = initial_pressure.copy()
        self.pressure_previous = np.zeros_like(initial_pressure)
        
        pressure_history = []
        dt = self.time_step
        dx = self.voxel_size
        
        # CFL stability condition
        cfl_factor = 0.5
        max_speed = np.max(self.sound_speed[np.isfinite(self.sound_speed)])
        if max_speed == 0:
            max_speed = 1540.0  # Default speed of sound in tissue
        
        stable_dt = cfl_factor * dx / max_speed
        if dt > stable_dt:
            warnings.warn(f"Time step {dt} may be unstable. Consider using dt <= {stable_dt}")
        
        for step in range(time_steps):
            # Calculate Laplacian using finite differences
            laplacian = self._calculate_laplacian_3d(self.pressure_current)
            
            # Wave equation: ∂²p/∂t² = vs² * ∇²p
            vs_squared = self.sound_speed ** 2
            acceleration = vs_squared * laplacian
            
            # Finite difference time stepping
            self.pressure_next = (2 * self.pressure_current - 
                                self.pressure_previous + 
                                dt**2 * acceleration)
            
            # Apply absorbing boundary conditions
            self._apply_absorbing_boundaries()
            
            # Store pressure field
            if step % 10 == 0:  # Store every 10th step to save memory
                pressure_history.append(self.pressure_current.copy())
            
            # Update for next iteration
            self.pressure_previous = self.pressure_current.copy()
            self.pressure_current = self.pressure_next.copy()
        
        return pressure_history
    
    def _calculate_laplacian_3d(self, field: np.ndarray) -> np.ndarray:
        """Calculate 3D Laplacian using finite differences"""
        laplacian = np.zeros_like(field)
        dx2 = self.voxel_size ** 2
        
        # Interior points (avoid boundaries)
        laplacian[1:-1, 1:-1, 1:-1] = (
            (field[2:, 1:-1, 1:-1] - 2*field[1:-1, 1:-1, 1:-1] + field[:-2, 1:-1, 1:-1]) +
            (field[1:-1, 2:, 1:-1] - 2*field[1:-1, 1:-1, 1:-1] + field[1:-1, :-2, 1:-1]) +
            (field[1:-1, 1:-1, 2:] - 2*field[1:-1, 1:-1, 1:-1] + field[1:-1, 1:-1, :-2])
        ) / dx2
        
        return laplacian
    
    def _apply_absorbing_boundaries(self):
        """Apply absorbing boundary conditions to prevent reflections"""
        damping_factor = 0.95
        boundary_width = 5
        
        for i in range(boundary_width):
            damping = damping_factor ** (boundary_width - i)
            
            # Apply to all six faces
            self.pressure_next[i, :, :] *= damping
            self.pressure_next[-i-1, :, :] *= damping
            self.pressure_next[:, i, :] *= damping
            self.pressure_next[:, -i-1, :] *= damping
            self.pressure_next[:, :, i] *= damping
            self.pressure_next[:, :, -i-1] *= damping

class UltrasoundDetector:
    """Ultrasound transducer array simulation"""
    
    def __init__(self, detector_type: str = 'ring_array', 
                 num_elements: int = 128,
                 frequency: float = 7.5e6,  # 7.5 MHz
                 bandwidth: float = 0.8):
        self.detector_type = detector_type
        self.num_elements = num_elements
        self.frequency = frequency
        self.bandwidth = bandwidth
        self.element_positions = self._generate_detector_positions()
    
    def _generate_detector_positions(self) -> np.ndarray:
        """Generate detector element positions"""
        if self.detector_type == 'ring_array':
            angles = np.linspace(0, 2*np.pi, self.num_elements, endpoint=False)
            radius = 0.1  # 10cm radius
            positions = np.zeros((self.num_elements, 3))
            positions[:, 0] = radius * np.cos(angles)
            positions[:, 1] = radius * np.sin(angles)
            positions[:, 2] = 0  # Ring in XY plane
            
        elif self.detector_type == 'linear_array':
            positions = np.zeros((self.num_elements, 3))
            element_spacing = 0.3e-3  # 0.3mm spacing
            positions[:, 0] = np.arange(self.num_elements) * element_spacing
            positions[:, 1] = 0
            positions[:, 2] = 0.1  # 10cm from center
            
        return positions
    
    def detect_acoustic_signals(self, pressure_history: List[np.ndarray],
                               grid_size: Tuple[int, int, int],
                               voxel_size: float,
                               time_step: float) -> np.ndarray:
        """Detect acoustic signals at detector positions"""
        num_time_steps = len(pressure_history)
        detected_signals = np.zeros((self.num_elements, num_time_steps))
        
        for t_idx, pressure_field in enumerate(pressure_history):
            for elem_idx, pos in enumerate(self.element_positions):
                # Convert position to grid coordinates
                grid_pos = pos / voxel_size
                grid_pos = grid_pos.astype(int)
                
                # Check bounds
                if (0 <= grid_pos[0] < grid_size[0] and 
                    0 <= grid_pos[1] < grid_size[1] and 
                    0 <= grid_pos[2] < grid_size[2]):
                    
                    # Trilinear interpolation for sub-voxel accuracy
                    signal = self._trilinear_interpolate(
                        pressure_field, pos / voxel_size)
                    detected_signals[elem_idx, t_idx] = signal
        
        # Apply frequency response and noise
        detected_signals = self._apply_transducer_response(detected_signals, time_step)
        
        return detected_signals
    
    def _trilinear_interpolate(self, field: np.ndarray, 
                              position: np.ndarray) -> float:
        """Trilinear interpolation for sub-voxel sampling"""
        x, y, z = position
        
        # Get integer coordinates
        x0, y0, z0 = int(x), int(y), int(z)
        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
        
        # Check bounds
        if (x1 >= field.shape[0] or y1 >= field.shape[1] or 
            z1 >= field.shape[2] or x0 < 0 or y0 < 0 or z0 < 0):
            return 0.0
        
        # Fractional parts
        fx, fy, fz = x - x0, y - y0, z - z0
        
        # Trilinear interpolation
        c000 = field[x0, y0, z0] * (1-fx) * (1-fy) * (1-fz)
        c001 = field[x0, y0, z1] * (1-fx) * (1-fy) * fz
        c010 = field[x0, y1, z0] * (1-fx) * fy * (1-fz)
        c011 = field[x0, y1, z1] * (1-fx) * fy * fz
        c100 = field[x1, y0, z0] * fx * (1-fy) * (1-fz)
        c101 = field[x1, y0, z1] * fx * (1-fy) * fz
        c110 = field[x1, y1, z0] * fx * fy * (1-fz)
        c111 = field[x1, y1, z1] * fx * fy * fz
        
        return c000 + c001 + c010 + c011 + c100 + c101 + c110 + c111
    
    def _apply_transducer_response(self, signals: np.ndarray, 
                                  time_step: float) -> np.ndarray:
        """Apply frequency response and add realistic noise"""
        # Butterworth bandpass filter
        nyquist = 0.5 / time_step
        low_freq = self.frequency * (1 - self.bandwidth/2) / nyquist
        high_freq = self.frequency * (1 + self.bandwidth/2) / nyquist
        
        # Ensure frequencies are within valid range
        low_freq = max(0.01, min(low_freq, 0.99))
        high_freq = max(low_freq + 0.01, min(high_freq, 0.99))
        
        b, a = butter(4, [low_freq, high_freq], btype='band')
        
        filtered_signals = np.zeros_like(signals)
        for i in range(signals.shape[0]):
            if np.any(signals[i, :] != 0):
                filtered_signals[i, :] = filtfilt(b, a, signals[i, :])
        
        # Add thermal noise
        signal_max = np.max(np.abs(filtered_signals))
        noise_level = 1e-6 * signal_max if signal_max > 0 else 1e-12
        noise = np.random.normal(0, noise_level, filtered_signals.shape)
        
        return filtered_signals + noise

def simulate_xact_scan(anatomy_model: np.ndarray,
                      tissue_labels: Dict[int, str],
                      beam_angle: Tuple[float, float] = (0.0, 0.0),  # Single beam angle (azimuth, elevation)
                      **kwargs) -> Dict:
    """2D XACT scan simulation with single X-ray beam for reduced radiation"""
    # Simulation parameters
    grid_size = tuple(anatomy_model.shape[:3])  # Ensure 3D shape
    voxel_size = kwargs.get('voxel_size', 0.5e-3)
    time_steps = kwargs.get('time_steps', 1000)
    beam_energy = kwargs.get('beam_energy', 120)  # keV
    beam_fluence = kwargs.get('beam_fluence', 1e10)
    beam_width = kwargs.get('beam_width', 0.02)  # 2cm beam width
    
    # Initialize physics engine
    physics = XACTPhysicsEngine(grid_size, voxel_size)
    physics.set_tissue_properties(anatomy_model, tissue_labels)
    
    # Initialize detector
    detector = UltrasoundDetector(
        detector_type=kwargs.get('detector_type', 'linear_array'),  # Changed to linear array for 2D imaging
        num_elements=kwargs.get('num_elements', 128),
        frequency=kwargs.get('frequency', 7.5e6)
    )
    
    # Initialize X-ray beam
    xray_beam = XRayBeam(energy_kV=beam_energy)
    
    print(f"Simulating 2D XACT scan with single beam at angle: "
          f"Az={np.degrees(beam_angle[0]):.1f}°, El={np.degrees(beam_angle[1]):.1f}°")
    
    # Calculate beam direction
    azimuth, elevation = beam_angle
    beam_direction = (
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation)
    )
    
    # Source position (outside the object)
    source_distance = 0.2  # 20cm from center
    source_pos = (-source_distance * beam_direction[0],
                 -source_distance * beam_direction[1],
                 -source_distance * beam_direction[2])
    
    # Calculate X-ray energy deposition with enhanced physics
    energy_deposition = physics._calculate_xray_attenuation(
        source_pos, beam_direction, beam_width)
    
    # Apply X-ray spectrum effects
    spectrum = xray_beam.spectrum
    energy_deposition *= np.sum(spectrum)  # Total beam energy
    
    # Generate initial acoustic pressure
    initial_pressure = physics.generate_initial_pressure(
        energy_deposition, beam_fluence)
    
    # Propagate acoustic waves with improved wave physics
    pressure_history = physics.propagate_acoustic_wave(
        initial_pressure, time_steps)
    
    # Detect signals with realistic sensor response
    detected_signals = detector.detect_acoustic_signals(
        pressure_history, grid_size, voxel_size, physics.time_step)
    
    beam_metadata = {
        'azimuth': azimuth,
        'elevation': elevation,
        'source_position': source_pos,
        'beam_direction': beam_direction,
        'beam_energy': beam_energy,
        'beam_fluence': beam_fluence,
        'beam_width': beam_width
    }
    
    return {
        'detected_signals': detected_signals[np.newaxis, ...],  # Add batch dimension for consistency
        'beam_metadata': [beam_metadata],  # Keep list format for API compatibility
        'detector_positions': detector.element_positions,
        'simulation_params': {
            'grid_size': grid_size,
            'voxel_size': voxel_size,
            'time_step': physics.time_step,
            'time_steps': time_steps,
            'beam_energy': beam_energy,
            'beam_fluence': beam_fluence,
            'beam_width': beam_width
        },
        'tissue_properties': physics
    } 