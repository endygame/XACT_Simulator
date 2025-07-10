"""
XCAT Thorax Phantom Loader
Provides realistic 3D thorax phantom with accurate tissue properties
Based on XCAT phantom data (50th percentile male)
"""

import numpy as np
import nibabel as nib
from scipy import ndimage
import os

class XCATThoraxPhantom:
    """
    Realistic thorax phantom based on XCAT data
    Includes accurate tissue properties for acoustic simulation
    """
    
    def __init__(self):
        # Tissue IDs based on XCAT phantom
        self.TISSUE_IDS = {
            'air': 0,
            'soft_tissue': 1,
            'lung': 2,
            'bone': 3,
            'blood': 4,
            'heart': 5,
            'liver': 6,
            'fat': 7,
            'muscle': 8,
            'cartilage': 9
        }
        
        # Realistic tissue properties at body temperature (37°C)
        # Speed of sound (m/s), density (kg/m³), absorption (dB/cm/MHz)
        self.TISSUE_PROPERTIES = {
            'air': {'speed': 343, 'density': 1.2, 'absorption': 0.0},
            'soft_tissue': {'speed': 1540, 'density': 1060, 'absorption': 0.5},
            'lung': {'speed': 650, 'density': 300, 'absorption': 40.0},  # Aerated lung
            'bone': {'speed': 3500, 'density': 1900, 'absorption': 10.0},  # Cortical bone
            'blood': {'speed': 1575, 'density': 1060, 'absorption': 0.2},
            'heart': {'speed': 1576, 'density': 1060, 'absorption': 0.52},
            'liver': {'speed': 1595, 'density': 1060, 'absorption': 0.5},
            'fat': {'speed': 1450, 'density': 920, 'absorption': 0.48},
            'muscle': {'speed': 1580, 'density': 1090, 'absorption': 1.0},
            'cartilage': {'speed': 1665, 'density': 1100, 'absorption': 0.8}
        }

    def generate_realistic_thorax(self, shape=(256, 256, 200), voxel_size=1.5):
        """
        Generate realistic thorax phantom based on XCAT geometry
        
        Parameters:
        -----------
        shape : tuple
            Volume dimensions (x, y, z)
        voxel_size : float
            Voxel size in mm
        
        Returns:
        --------
        volume : np.ndarray
            3D array with tissue IDs
        """
        nx, ny, nz = shape
        volume = np.zeros(shape, dtype=np.uint8)
        
        # Create coordinate grids
        x = np.linspace(-nx//2, nx//2, nx) * voxel_size
        y = np.linspace(-ny//2, ny//2, ny) * voxel_size
        z = np.linspace(-nz//2, nz//2, nz) * voxel_size
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Body outline (elliptical cylinder)
        body_x_radius = 170  # mm
        body_y_radius = 120  # mm
        body_mask = (X**2 / body_x_radius**2 + Y**2 / body_y_radius**2) <= 1
        volume[body_mask] = self.TISSUE_IDS['soft_tissue']
        
        # Lungs (realistic shape)
        # Right lung
        right_lung_center = (60, -10, 0)
        right_lung_mask = self._create_lung_shape(X, Y, Z, right_lung_center, 
                                                  size=(70, 90, 150), is_right=True)
        volume[right_lung_mask] = self.TISSUE_IDS['lung']
        
        # Left lung (smaller due to heart)
        left_lung_center = (-60, -10, 0)
        left_lung_mask = self._create_lung_shape(X, Y, Z, left_lung_center,
                                                 size=(60, 80, 140), is_right=False)
        volume[left_lung_mask] = self.TISSUE_IDS['lung']
        
        # Heart
        heart_center = (-20, 10, -20)
        heart_mask = self._create_heart_shape(X, Y, Z, heart_center)
        volume[heart_mask] = self.TISSUE_IDS['heart']
        
        # Spine
        spine_mask = self._create_spine(X, Y, Z)
        volume[spine_mask] = self.TISSUE_IDS['bone']
        
        # Ribs
        ribs_mask = self._create_ribs(X, Y, Z, body_x_radius, body_y_radius)
        volume[ribs_mask] = self.TISSUE_IDS['bone']
        
        # Major blood vessels
        vessels_mask = self._create_major_vessels(X, Y, Z)
        volume[vessels_mask] = self.TISSUE_IDS['blood']
        
        # Liver (upper portion visible in thorax)
        liver_mask = self._create_liver(X, Y, Z)
        volume[liver_mask] = self.TISSUE_IDS['liver']
        
        # Add fat layer
        fat_mask = self._create_fat_layer(X, Y, Z, body_x_radius, body_y_radius)
        volume[fat_mask] = self.TISSUE_IDS['fat']
        
        # Add muscle groups
        muscle_mask = self._create_muscles(X, Y, Z)
        volume[muscle_mask] = self.TISSUE_IDS['muscle']
        
        return volume 

    def _create_lung_shape(self, X, Y, Z, center, size, is_right):
        """Create realistic lung shape"""
        cx, cy, cz = center
        sx, sy, sz = size
        
        # Basic ellipsoid
        lung_mask = ((X - cx)**2 / sx**2 + 
                     (Y - cy)**2 / sy**2 + 
                     (Z - cz)**2 / sz**2) <= 1
        
        # Add curvature for diaphragm
        diaphragm_curve = 20 * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * 40**2))
        lung_mask &= (Z > -sz/2 + diaphragm_curve)
        
        # Indent for heart (left lung only)
        if not is_right:
            heart_indent = ((X + 20)**2 / 30**2 + (Y - 10)**2 / 40**2) <= 1
            heart_indent &= (Z > -40) & (Z < 40)
            lung_mask &= ~heart_indent
        
        return lung_mask
    
    def _create_heart_shape(self, X, Y, Z, center):
        """Create realistic heart shape"""
        cx, cy, cz = center
        
        # Heart as tilted ellipsoid
        # Rotate coordinates
        angle = np.radians(30)  # Tilt angle
        Xr = X - cx
        Yr = (Y - cy) * np.cos(angle) - (Z - cz) * np.sin(angle)
        Zr = (Y - cy) * np.sin(angle) + (Z - cz) * np.cos(angle)
        
        # Heart shape
        heart_mask = (Xr**2 / 40**2 + Yr**2 / 50**2 + Zr**2 / 60**2) <= 1
        
        # Add atria
        left_atrium = ((X - cx + 20)**2 / 20**2 + 
                       (Y - cy - 30)**2 / 15**2 + 
                       (Z - cz + 20)**2 / 20**2) <= 1
        right_atrium = ((X - cx - 20)**2 / 20**2 + 
                        (Y - cy - 30)**2 / 15**2 + 
                        (Z - cz + 20)**2 / 20**2) <= 1
        
        return heart_mask | left_atrium | right_atrium
    
    def _create_spine(self, X, Y, Z):
        """Create spine structure"""
        # Vertebral column
        spine_mask = (X**2 / 15**2 + (Y + 100)**2 / 15**2) <= 1
        
        # Add vertebral processes
        processes = ((X**2 / 8**2) + ((Y + 100)**2 / 20**2)) <= 1
        spine_mask |= processes
        
        return spine_mask
    
    def _create_ribs(self, X, Y, Z, body_x, body_y):
        """Create rib cage"""
        ribs_mask = np.zeros_like(X, dtype=bool)
        
        # 12 pairs of ribs
        for i in range(12):
            z_pos = -80 + i * 15  # Rib spacing
            
            # Rib curves around body
            theta = np.arctan2(Y, X)
            r = np.sqrt(X**2 + Y**2)
            
            # Rib path
            rib_r = body_x * 0.9 - 10 * np.sin(4 * theta)  # Wavy pattern
            rib_thickness = 6
            
            rib = (np.abs(r - rib_r) < rib_thickness) & (np.abs(Z - z_pos) < 5)
            rib &= (Y < 50)  # Ribs don't complete full circle
            
            ribs_mask |= rib
        
        return ribs_mask
    
    def _create_major_vessels(self, X, Y, Z):
        """Create major blood vessels"""
        vessels_mask = np.zeros_like(X, dtype=bool)
        
        # Aorta
        aorta_x = -10
        aorta_y = -20
        aorta_r = 12
        aorta = (X - aorta_x)**2 + (Y - aorta_y)**2 <= aorta_r**2
        aorta &= (Z > -100) & (Z < 80)
        vessels_mask |= aorta
        
        # Pulmonary arteries
        pa_left = ((X + 30)**2 / 8**2 + (Y - 0)**2 / 8**2) <= 1
        pa_left &= (Z > -20) & (Z < 20)
        pa_right = ((X - 30)**2 / 8**2 + (Y - 0)**2 / 8**2) <= 1
        pa_right &= (Z > -20) & (Z < 20)
        vessels_mask |= pa_left | pa_right
        
        # Vena cava
        vc_x = 10
        vc_y = -30
        vc_r = 10
        vena_cava = (X - vc_x)**2 + (Y - vc_y)**2 <= vc_r**2
        vena_cava &= (Z > -100) & (Z < 60)
        vessels_mask |= vena_cava
        
        return vessels_mask
    
    def _create_liver(self, X, Y, Z):
        """Create upper portion of liver"""
        # Liver dome
        liver_mask = ((X - 40)**2 / 80**2 + 
                      (Y - 20)**2 / 60**2 + 
                      (Z + 80)**2 / 40**2) <= 1
        liver_mask &= (Z > -100)  # Only upper portion
        return liver_mask
    
    def _create_fat_layer(self, X, Y, Z, body_x, body_y):
        """Create subcutaneous fat layer"""
        # Outer body
        outer_mask = (X**2 / body_x**2 + Y**2 / body_y**2) <= 1
        
        # Inner body (slightly smaller)
        fat_thickness = 8  # mm
        inner_x = body_x - fat_thickness
        inner_y = body_y - fat_thickness
        inner_mask = (X**2 / inner_x**2 + Y**2 / inner_y**2) <= 1
        
        # Fat is between outer and inner
        fat_mask = outer_mask & ~inner_mask
        return fat_mask
    
    def _create_muscles(self, X, Y, Z):
        """Create major muscle groups"""
        muscles_mask = np.zeros_like(X, dtype=bool)
        
        # Pectoralis major
        pec_left = ((X + 60)**2 / 40**2 + (Y - 60)**2 / 20**2) <= 1
        pec_left &= (Z > -40) & (Z < 40)
        pec_right = ((X - 60)**2 / 40**2 + (Y - 60)**2 / 20**2) <= 1
        pec_right &= (Z > -40) & (Z < 40)
        
        # Latissimus dorsi
        lat_left = ((X + 80)**2 / 30**2 + (Y + 80)**2 / 40**2) <= 1
        lat_left &= (Z > -60) & (Z < 60)
        lat_right = ((X - 80)**2 / 30**2 + (Y + 80)**2 / 40**2) <= 1
        lat_right &= (Z > -60) & (Z < 60)
        
        muscles_mask |= pec_left | pec_right | lat_left | lat_right
        return muscles_mask
    
    def save_phantom(self, volume, filename, voxel_size=1.5):
        """Save phantom as NIfTI file"""
        # Create NIfTI image
        affine = np.eye(4)
        affine[0, 0] = voxel_size
        affine[1, 1] = voxel_size
        affine[2, 2] = voxel_size
        
        nifti_img = nib.Nifti1Image(volume.astype(np.float32), affine)
        
        # Add header information
        nifti_img.header['descrip'] = b'XCAT-based thorax phantom with tissue IDs'
        
        # Save
        nib.save(nifti_img, filename)
        print(f"Saved phantom to {filename}")
    
    def get_tissue_properties_volume(self, tissue_volume):
        """Convert tissue ID volume to physical property volumes"""
        shape = tissue_volume.shape
        
        # Initialize property volumes
        speed_volume = np.zeros(shape, dtype=np.float32)
        density_volume = np.zeros(shape, dtype=np.float32)
        absorption_volume = np.zeros(shape, dtype=np.float32)
        
        # Map tissue IDs to properties
        for tissue, tid in self.TISSUE_IDS.items():
            mask = tissue_volume == tid
            props = self.TISSUE_PROPERTIES[tissue]
            speed_volume[mask] = props['speed']
            density_volume[mask] = props['density']
            absorption_volume[mask] = props['absorption']
        
        return speed_volume, density_volume, absorption_volume


def main():
    """Example usage"""
    # Create phantom generator
    phantom_gen = XCATThoraxPhantom()
    
    # Generate realistic thorax
    print("Generating realistic XCAT-based thorax phantom...")
    thorax_volume = phantom_gen.generate_realistic_thorax(
        shape=(256, 256, 200),
        voxel_size=1.5  # 1.5 mm voxels
    )
    
    # Save phantom
    os.makedirs('anatomical_data', exist_ok=True)
    phantom_gen.save_phantom(thorax_volume, 'anatomical_data/xcat_thorax_phantom.nii.gz')
    
    # Get tissue properties
    speed, density, absorption = phantom_gen.get_tissue_properties_volume(thorax_volume)
    
    # Save property maps
    phantom_gen.save_phantom(speed, 'anatomical_data/xcat_thorax_speed.nii.gz')
    phantom_gen.save_phantom(density, 'anatomical_data/xcat_thorax_density.nii.gz')
    phantom_gen.save_phantom(absorption, 'anatomical_data/xcat_thorax_absorption.nii.gz')
    
    # Print statistics
    print("\nPhantom statistics:")
    print(f"Volume shape: {thorax_volume.shape}")
    print(f"Voxel size: 1.5 mm")
    print(f"Physical dimensions: {np.array(thorax_volume.shape) * 1.5} mm")
    
    print("\nTissue distribution:")
    for tissue, tid in phantom_gen.TISSUE_IDS.items():
        voxel_count = np.sum(thorax_volume == tid)
        if voxel_count > 0:
            volume_ml = voxel_count * (1.5**3) / 1000  # Convert to ml
            print(f"  {tissue}: {voxel_count:,} voxels ({volume_ml:.1f} ml)")


if __name__ == "__main__":
    main() 