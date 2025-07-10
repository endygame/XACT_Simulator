"""
Generic Sensor Interface for XACT Simulation

This module provides a generic sensor interface with properties and simulated
behavior for various sensor types used in XACT imaging.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from scipy.signal import butter, filtfilt
import warnings


@dataclass
class SensorProperties:
    """Physical and electrical properties of a sensor"""
    # Physical properties
    position: np.ndarray  # 3D position [x, y, z] in meters
    orientation: np.ndarray  # Unit vector for sensor direction
    active_area: float  # Active sensing area in m²
    
    # Frequency response
    center_frequency: float  # Hz
    bandwidth: float  # Hz
    frequency_range: Tuple[float, float]  # (min_freq, max_freq) in Hz
    
    # Sensitivity and performance
    sensitivity: float  # V/Pa or equivalent units
    noise_floor: float  # Noise level in same units as output
    dynamic_range: float  # dB
    
    # Physical constraints
    max_amplitude: float  # Maximum detectable amplitude
    saturation_threshold: float  # Saturation threshold
    
    # Additional properties (sensor-specific)
    custom_properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_properties is None:
            self.custom_properties = {}


class GenericSensor(ABC):
    """Abstract base class for all sensor types"""
    
    def __init__(self, sensor_id: str, properties: SensorProperties):
        self.sensor_id = sensor_id
        self.properties = properties
        self.is_active = True
        self.calibration_factor = 1.0
        self._signal_history = []
        
    @abstractmethod
    def detect_signal(self, input_field: np.ndarray, time_step: float, **kwargs) -> float:
        """
        Detect and convert physical field to electrical signal
        
        Args:
            input_field: Physical field at sensor location
            time_step: Time step for simulation
            **kwargs: Additional sensor-specific parameters
            
        Returns:
            Electrical signal value
        """
        pass
    
    @abstractmethod
    def apply_frequency_response(self, signal: np.ndarray, sample_rate: float) -> np.ndarray:
        """
        Apply sensor frequency response to signal
        
        Args:
            signal: Input signal array
            sample_rate: Sampling rate in Hz
            
        Returns:
            Filtered signal
        """
        pass
    
    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        """Add thermal and electronic noise to signal"""
        noise_std = self.properties.noise_floor
        noise = np.random.normal(0, noise_std, signal.shape)
        return signal + noise
    
    def apply_saturation(self, signal: np.ndarray) -> np.ndarray:
        """Apply saturation effects"""
        return np.clip(signal, -self.properties.saturation_threshold, 
                      self.properties.saturation_threshold)
    
    def apply_calibration(self, signal: np.ndarray) -> np.ndarray:
        """Apply calibration factor"""
        return signal * self.calibration_factor
    
    def get_position(self) -> np.ndarray:
        """Get sensor position"""
        return self.properties.position.copy()
    
    def set_position(self, position: np.ndarray):
        """Set sensor position"""
        self.properties.position = position.copy()
    
    def get_orientation(self) -> np.ndarray:
        """Get sensor orientation"""
        return self.properties.orientation.copy()
    
    def set_orientation(self, orientation: np.ndarray):
        """Set sensor orientation (will be normalized)"""
        self.properties.orientation = orientation / np.linalg.norm(orientation)
    
    def is_within_frequency_range(self, frequency: float) -> bool:
        """Check if frequency is within sensor range"""
        min_freq, max_freq = self.properties.frequency_range
        return min_freq <= frequency <= max_freq
    
    def get_sensitivity_at_frequency(self, frequency: float) -> float:
        """Get frequency-dependent sensitivity"""
        if not self.is_within_frequency_range(frequency):
            return 0.0
        
        # Simple Gaussian response around center frequency
        center_freq = self.properties.center_frequency
        bandwidth = self.properties.bandwidth
        
        # Normalized frequency difference
        freq_diff = abs(frequency - center_freq) / (bandwidth / 2)
        
        # Gaussian response
        response = np.exp(-0.5 * freq_diff**2)
        
        return self.properties.sensitivity * response
    
    def record_signal(self, signal_value: float, timestamp: float):
        """Record signal for history tracking"""
        self._signal_history.append((timestamp, signal_value))
    
    def get_signal_history(self) -> List[Tuple[float, float]]:
        """Get recorded signal history"""
        return self._signal_history.copy()
    
    def clear_history(self):
        """Clear signal history"""
        self._signal_history.clear()


class UltrasoundSensor(GenericSensor):
    """Ultrasound transducer sensor implementation"""
    
    def __init__(self, sensor_id: str, position: np.ndarray, 
                 center_frequency: float = 7.5e6, bandwidth: float = 3.0e6):
        """
        Initialize ultrasound sensor
        
        Args:
            sensor_id: Unique identifier
            position: 3D position [x, y, z] in meters
            center_frequency: Center frequency in Hz (default 7.5 MHz)
            bandwidth: Bandwidth in Hz (default 3 MHz)
        """
        # Default orientation pointing toward origin
        orientation = -position / np.linalg.norm(position) if np.linalg.norm(position) > 0 else np.array([0, 0, 1])
        
        # Define ultrasound sensor properties
        properties = SensorProperties(
            position=position.copy(),
            orientation=orientation,
            active_area=1e-6,  # 1 mm² active area
            center_frequency=center_frequency,
            bandwidth=bandwidth,
            frequency_range=(center_frequency - bandwidth/2, center_frequency + bandwidth/2),
            sensitivity=1e-3,  # 1 mV/Pa
            noise_floor=1e-6,  # 1 µV noise floor
            dynamic_range=80.0,  # 80 dB dynamic range
            max_amplitude=1.0,  # 1 V maximum
            saturation_threshold=0.8,  # 0.8 V saturation
            custom_properties={
                'piezo_element_type': 'PZT-5H',
                'impedance': 50.0,  # 50 Ohm impedance
                'beam_pattern': 'omnidirectional',
                'temperature_coefficient': -0.002  # -0.2% per °C
            }
        )
        
        super().__init__(sensor_id, properties)
        
        # Ultrasound-specific parameters
        self.acoustic_coupling = 1.0  # Perfect coupling by default
        self.temperature = 20.0  # °C
        
    def detect_signal(self, pressure_field: np.ndarray, time_step: float, 
                     field_positions: np.ndarray = None, **kwargs) -> float:
        """
        Detect acoustic pressure and convert to electrical signal
        
        Args:
            pressure_field: 3D pressure field array
            time_step: Time step for simulation
            field_positions: Grid positions corresponding to pressure field
            **kwargs: Additional parameters
            
        Returns:
            Electrical signal value in volts
        """
        if pressure_field.size == 0:
            return 0.0
        
        # Get pressure at sensor location
        if field_positions is not None:
            # Interpolate pressure at exact sensor position
            pressure = self._interpolate_field_value(pressure_field, field_positions, 
                                                   self.properties.position)
        else:
            # Use simple sampling if no position grid provided
            # Assume field is centered and use middle values
            center_idx = tuple(s//2 for s in pressure_field.shape)
            pressure = pressure_field[center_idx] if len(pressure_field.shape) == 3 else pressure_field.mean()
        
        # Apply directional sensitivity (simple cosine response)
        directional_factor = self._calculate_directional_response(pressure_field, field_positions)
        pressure *= directional_factor
        
        # Convert pressure to voltage using sensitivity
        voltage = pressure * self.properties.sensitivity * self.acoustic_coupling
        
        # Apply temperature compensation
        temp_factor = 1.0 + self.properties.custom_properties['temperature_coefficient'] * (self.temperature - 20.0)
        voltage *= temp_factor
        
        return voltage
    
    def _interpolate_field_value(self, field: np.ndarray, positions: np.ndarray, 
                               target_position: np.ndarray) -> float:
        """Interpolate field value at target position"""
        # Simple nearest neighbor for now
        # In practice, would use trilinear interpolation
        if positions.size == 0 or field.size == 0:
            return 0.0
        
        distances = np.linalg.norm(positions - target_position, axis=-1)
        nearest_idx = np.unravel_index(np.argmin(distances), distances.shape)
        
        return field[nearest_idx] if field.size > 0 else 0.0
    
    def _calculate_directional_response(self, field: np.ndarray, positions: np.ndarray) -> float:
        """Calculate directional response factor"""
        # Simple omnidirectional response for now
        # In practice, would calculate based on sensor orientation and beam pattern
        return 1.0
    
    def apply_frequency_response(self, signal: np.ndarray, sample_rate: float) -> np.ndarray:
        """Apply ultrasound transducer frequency response"""
        if signal.size < 2:
            return signal
        
        # Design bandpass filter based on sensor properties
        nyquist = sample_rate / 2
        low_freq = max(self.properties.frequency_range[0] / nyquist, 0.01)
        high_freq = min(self.properties.frequency_range[1] / nyquist, 0.99)
        
        # Ensure valid frequency range
        if low_freq >= high_freq:
            return signal
        
        try:
            # 4th order Butterworth bandpass filter
            b, a = butter(4, [low_freq, high_freq], btype='band')
            filtered_signal = filtfilt(b, a, signal)
            return filtered_signal
        except Exception as e:
            warnings.warn(f"Filter design failed: {e}")
            return signal
    
    def set_acoustic_coupling(self, coupling: float):
        """Set acoustic coupling factor (0.0 to 1.0)"""
        self.acoustic_coupling = max(0.0, min(1.0, coupling))
    
    def set_temperature(self, temperature: float):
        """Set operating temperature in Celsius"""
        self.temperature = temperature
    
    def get_beam_pattern(self, angles: np.ndarray) -> np.ndarray:
        """Get beam pattern response at different angles"""
        # Simple omnidirectional pattern
        # In practice, would depend on element size and frequency
        return np.ones_like(angles)


class SensorArray:
    """Collection of sensors with array-level operations"""
    
    def __init__(self, array_name: str):
        self.array_name = array_name
        self.sensors: List[GenericSensor] = []
        self._geometry_type = None
        
    def add_sensor(self, sensor: GenericSensor):
        """Add sensor to array"""
        self.sensors.append(sensor)
    
    def remove_sensor(self, sensor_id: str) -> bool:
        """Remove sensor by ID"""
        for i, sensor in enumerate(self.sensors):
            if sensor.sensor_id == sensor_id:
                self.sensors.pop(i)
                return True
        return False
    
    def get_sensor(self, sensor_id: str) -> Optional[GenericSensor]:
        """Get sensor by ID"""
        for sensor in self.sensors:
            if sensor.sensor_id == sensor_id:
                return sensor
        return None
    
    def get_all_positions(self) -> np.ndarray:
        """Get positions of all sensors"""
        return np.array([sensor.get_position() for sensor in self.sensors])
    
    def detect_all_signals(self, input_field: np.ndarray, time_step: float, 
                          field_positions: np.ndarray = None, **kwargs) -> np.ndarray:
        """Detect signals from all sensors"""
        signals = []
        for sensor in self.sensors:
            if sensor.is_active:
                signal = sensor.detect_signal(input_field, time_step, field_positions, **kwargs)
                signals.append(signal)
            else:
                signals.append(0.0)
        return np.array(signals)
    
    def apply_array_processing(self, signals: np.ndarray, sample_rate: float) -> np.ndarray:
        """Apply array-level signal processing"""
        processed_signals = np.zeros_like(signals)
        
        for i, sensor in enumerate(self.sensors):
            if sensor.is_active and i < len(signals):
                # Apply individual sensor processing
                signal = signals[i] if signals.ndim == 1 else signals[i, :]
                filtered = sensor.apply_frequency_response(signal.reshape(-1), sample_rate)
                noisy = sensor.add_noise(filtered)
                saturated = sensor.apply_saturation(noisy)
                calibrated = sensor.apply_calibration(saturated)
                
                if signals.ndim == 1:
                    processed_signals[i] = calibrated[0] if calibrated.size > 0 else 0.0
                else:
                    processed_signals[i, :] = calibrated[:signals.shape[1]]
        
        return processed_signals
    
    def create_circular_array(self, center: np.ndarray, radius: float, 
                            num_sensors: int, sensor_type: type = UltrasoundSensor,
                            **sensor_kwargs) -> 'SensorArray':
        """Create circular sensor array"""
        self._geometry_type = 'circular'
        
        angles = np.linspace(0, 2*np.pi, num_sensors, endpoint=False)
        
        for i, angle in enumerate(angles):
            # Position in circular array
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = center[2]
            position = np.array([x, y, z])
            
            # Create sensor
            sensor_id = f"sensor_{i:03d}"
            sensor = sensor_type(sensor_id, position, **sensor_kwargs)
            self.add_sensor(sensor)
        
        return self
    
    def create_linear_array(self, start_pos: np.ndarray, end_pos: np.ndarray,
                          num_sensors: int, sensor_type: type = UltrasoundSensor,
                          **sensor_kwargs) -> 'SensorArray':
        """Create linear sensor array"""
        self._geometry_type = 'linear'
        
        for i in range(num_sensors):
            # Linear interpolation between start and end
            t = i / (num_sensors - 1) if num_sensors > 1 else 0
            position = start_pos + t * (end_pos - start_pos)
            
            # Create sensor
            sensor_id = f"sensor_{i:03d}"
            sensor = sensor_type(sensor_id, position, **sensor_kwargs)
            self.add_sensor(sensor)
        
        return self
    
    def get_array_info(self) -> Dict[str, Any]:
        """Get array information"""
        return {
            'array_name': self.array_name,
            'num_sensors': len(self.sensors),
            'geometry_type': self._geometry_type,
            'active_sensors': sum(1 for s in self.sensors if s.is_active),
            'sensor_types': list(set(type(s).__name__ for s in self.sensors)),
            'center_position': np.mean(self.get_all_positions(), axis=0) if self.sensors else None
        }


def create_default_ultrasound_array(center: np.ndarray = np.array([0, 0, 0]), 
                                  radius: float = 0.1, num_sensors: int = 128,
                                  center_frequency: float = 7.5e6,
                                  bandwidth: float = 3.0e6) -> SensorArray:
    """
    Create default ultrasound sensor array for XACT
    
    Args:
        center: Array center position in meters
        radius: Array radius in meters (default 10 cm)
        num_sensors: Number of sensors (default 128)
        center_frequency: Center frequency in Hz (default 7.5 MHz)
        bandwidth: Bandwidth in Hz (default 3 MHz)
        
    Returns:
        Configured sensor array
    """
    # Use the improved UltrasoundSensor from sensors package if available
    try:
        from sensors import UltrasoundSensor as ImprovedUltrasoundSensor
        sensor_class = ImprovedUltrasoundSensor
    except ImportError:
        # Fallback to the basic implementation
        sensor_class = UltrasoundSensor
    
    array = SensorArray("default_ultrasound_array")
    array.create_circular_array(
        center=center,
        radius=radius,
        num_sensors=num_sensors,
        sensor_type=sensor_class,
        center_frequency=center_frequency,
        bandwidth=bandwidth
    )
    return array


# Example usage and testing
if __name__ == "__main__":
    # Create default ultrasound array
    array = create_default_ultrasound_array(
        center=np.array([0, 0, 0]),
        radius=0.1,  # 10 cm
        num_sensors=64
    )
    
    print(f"Created sensor array with {len(array.sensors)} sensors")
    print(f"Array info: {array.get_array_info()}")
    
    # Test signal detection with dummy data
    dummy_field = np.random.normal(0, 1e-3, (32, 32, 32))  # Random pressure field
    signals = array.detect_all_signals(dummy_field, 1e-8)  # 10 ns time step
    
    print(f"Detected signals: {signals.shape}")
    print(f"Signal range: {signals.min():.2e} to {signals.max():.2e}")
    
    # Apply array processing
    time_signals = np.random.normal(0, 1e-3, (len(array.sensors), 1000))
    processed = array.apply_array_processing(time_signals, 100e6)  # 100 MHz sample rate
    
    print(f"Processed signals: {processed.shape}")
    print(f"Processing complete") 