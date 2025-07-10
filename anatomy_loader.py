"""
Anatomical Data Loader

Loads and processes anatomical data for XACT simulation
"""

import numpy as np
import os

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    print("Warning: nibabel not available - limited NIFTI support")

from xcat_thorax_loader import XCATThoraxPhantom


class AnatomyLoader:
    """Loader for realistic anatomical data using XCAT phantom"""
    
    def __init__(self):
        self.phantom_gen = XCATThoraxPhantom()
        self.current_data = None
        self.current_type = None

    def load_xcat_thorax(self, generate_new=False):
        """
        Load XCAT thorax phantom with realistic tissue properties

        Parameters:
        -----------
        generate_new : bool
            If True, generate new phantom. If False, try to load existing.

        Returns:
        --------
        data : dict
            Dictionary containing:
            - 'volume': 3D tissue ID array
            - 'tissue_ids': Mapping of tissue names to IDs
            - 'tissue_properties': Physical properties per tissue
            - 'spacing': Voxel spacing in mm
        """
        phantom_file = 'anatomical_data/xcat_thorax_phantom.nii.gz'

        if not generate_new and os.path.exists(phantom_file) and NIBABEL_AVAILABLE:
            # Load existing phantom
            print("Loading existing XCAT thorax phantom...")
            nifti = nib.load(phantom_file)
            volume = nifti.get_fdata()
            spacing = nifti.header.get_zooms()[:3]
        else:
            # Generate new phantom
            print("Generating new XCAT thorax phantom...")
            volume = self.phantom_gen.generate_realistic_thorax(
                shape=(256, 256, 200),
                voxel_size=1.5
            )
            spacing = (1.5, 1.5, 1.5)

            # Save for future use
            os.makedirs('anatomical_data', exist_ok=True)
            self.phantom_gen.save_phantom(volume, phantom_file)

        # Get tissue properties
        speed, density, absorption = self.phantom_gen.get_tissue_properties_volume(volume)

        data = {
            'volume': volume.astype(np.uint8),
            'tissue_ids': self.phantom_gen.TISSUE_IDS,
            'tissue_properties': self.phantom_gen.TISSUE_PROPERTIES,
            'speed_volume': speed,
            'density_volume': density,
            'absorption_volume': absorption,
            'spacing': spacing,
            'type': 'xcat_thorax'
        }

        self.current_data = data
        self.current_type = 'xcat_thorax'

        # Print statistics
        self._print_phantom_stats(data)

        return data

    def create_synthetic_data(self, shape=(128, 128, 128), mode='thorax'):
        """Create simple synthetic data for testing"""
        if mode == 'thorax':
            # Use XCAT phantom generator but with smaller size
            print(f"Generating synthetic thorax with shape {shape}...")
            volume = self.phantom_gen.generate_realistic_thorax(
                shape=shape,
                voxel_size=2.0  # Larger voxels for synthetic
            )

            # Get tissue properties
            speed, density, absorption = self.phantom_gen.get_tissue_properties_volume(volume)

            data = {
                'volume': volume.astype(np.uint8),
                'tissue_ids': self.phantom_gen.TISSUE_IDS,
                'tissue_properties': self.phantom_gen.TISSUE_PROPERTIES,
                'speed_volume': speed,
                'density_volume': density,
                'absorption_volume': absorption,
                'spacing': (2.0, 2.0, 2.0),
                'type': 'synthetic_thorax'
            }
        else:
            # Simple sphere phantom
            data = self._create_sphere_phantom(shape)

        self.current_data = data
        self.current_type = data['type']

        return data

    def load_existing_anatomical_data(self):
        """Load existing anatomical data from the anatomical_data directory"""
        anatomical_dir = 'anatomical_data'
        existing_datasets = []

        if not os.path.exists(anatomical_dir):
            os.makedirs(anatomical_dir)
            return existing_datasets

        # Look for XCAT phantom first
        xcat_file = os.path.join(anatomical_dir, 'xcat_thorax_phantom.nii.gz')
        if os.path.exists(xcat_file):
            print(f" Found XCAT thorax phantom: {xcat_file}")
            xcat_data = self.load_xcat_thorax(generate_new=False)
            if xcat_data:
                dataset = {
                    'data': xcat_data['speed_volume'],
                    'voxel_size': xcat_data['spacing'],
                    'metadata': {
                        'description': 'XCAT Thorax Phantom',
                        'modality': 'CT',
                        'source': 'XCAT Phantom'
                    },
                    'anatomy_data': xcat_data,
                    'filepath': xcat_file
                }
                existing_datasets.append(dataset)

        # Look for other .nii.gz files
        for filename in os.listdir(anatomical_dir):
            if filename.endswith('.nii.gz') and filename != 'xcat_thorax_phantom.nii.gz':
                filepath = os.path.join(anatomical_dir, filename)
                print(f" Found anatomical file: {filepath}")

                try:
                    if NIBABEL_AVAILABLE:
                        nifti = nib.load(filepath)
                        data = nifti.get_fdata()
                        spacing = nifti.header.get_zooms()[:3]

                        dataset = {
                            'data': data.astype(np.float32),
                            'voxel_size': spacing,
                            'metadata': {
                                'description': filename.replace('.nii.gz', '').replace('_', ' ').title(),
                                'modality': 'Unknown',
                                'source': 'Local File'
                            },
                            'filepath': filepath
                        }
                        existing_datasets.append(dataset)
                except Exception as e:
                    print(f" Failed to load {filepath}: {e}")

        return existing_datasets

    def _create_sphere_phantom(self, shape):
        """Create simple sphere phantom for testing"""
        volume = np.zeros(shape, dtype=np.uint8)
        center = np.array(shape) // 2

        # Create spherical regions
        x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]

        # Outer sphere (soft tissue)
        r_outer = min(shape) // 3
        mask_outer = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= r_outer**2
        volume[mask_outer] = 1  # Soft tissue

        # Inner sphere (different tissue)
        r_inner = r_outer // 2
        mask_inner = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= r_inner**2
        volume[mask_inner] = 5  # Heart tissue

        # Get properties
        speed = np.zeros(shape, dtype=np.float32)
        density = np.zeros(shape, dtype=np.float32)
        absorption = np.zeros(shape, dtype=np.float32)

        speed[volume == 0] = 343  # Air
        speed[volume == 1] = 1540  # Soft tissue
        speed[volume == 5] = 1576  # Heart

        density[volume == 0] = 1.2
        density[volume == 1] = 1060
        density[volume == 5] = 1060

        absorption[volume == 0] = 0.0
        absorption[volume == 1] = 0.5
        absorption[volume == 5] = 0.52

        return {
            'volume': volume,
            'tissue_ids': {'air': 0, 'soft_tissue': 1, 'heart': 5},
            'tissue_properties': self.phantom_gen.TISSUE_PROPERTIES,
            'speed_volume': speed,
            'density_volume': density,
            'absorption_volume': absorption,
            'spacing': (1.0, 1.0, 1.0),
            'type': 'synthetic_sphere'
        }

    def _print_phantom_stats(self, data):
        """Print statistics about the phantom"""
        volume = data['volume']
        print(f"  Volume shape: {volume.shape}")
        print(f"  Voxel spacing: {data['spacing']} mm")
        print(f"  Tissue types: {len(data['tissue_ids'])}")
        
        unique_tissues = np.unique(volume)
        print(f"  Tissues present: {unique_tissues}")
        
        for tissue_id in unique_tissues:
            count = np.sum(volume == tissue_id)
            percentage = (count / volume.size) * 100
            print(f"    Tissue {tissue_id}: {count} voxels ({percentage:.1f}%)")


    def main():
        """Example usage"""
        loader = RealAnatomyLoader()

        # Load XCAT thorax phantom
        print("Loading XCAT thorax phantom...")
        thorax_data = loader.load_xcat_thorax(generate_new=True)

        # Example: Get lung tissue mask
        lung_mask = loader.get_tissue_mask('lung')
        print(f"\nLung volume: {np.sum(lung_mask) * np.prod(thorax_data['spacing']) / 1000:.1f} ml")

        # Example: Create smaller synthetic version
        print("\nCreating synthetic test phantom...")
        synthetic_data = loader.create_synthetic_data(shape=(64, 64, 64))


    if __name__ == "__main__":
        main() 