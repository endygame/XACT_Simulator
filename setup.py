#!/usr/bin/env python3
"""
Cross-platform setup script for XACT Simulation

Compatible with macOS, Linux, and Windows.
Automatically detects the platform and installs appropriate dependencies.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import tempfile
import urllib.request


def detect_platform():
    """Detect the operating system"""
    system = platform.system().lower()
    return {
        'darwin': 'macos',
        'linux': 'linux',
        'windows': 'windows'
    }.get(system, 'unknown')


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"Python version: {version.major}.{version.minor}.{version.micro} - OK")
    return True


def run_command(command, capture_output=True):
    """Run a system command and return result"""
    try:
        if isinstance(command, str):
            result = subprocess.run(command, shell=True, capture_output=capture_output, text=True)
        else:
            result = subprocess.run(command, capture_output=capture_output, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def check_pip():
    """Check if pip is available and upgrade if needed"""
    print("Checking pip...")
    success, stdout, stderr = run_command([sys.executable, "-m", "pip", "--version"])
    
    if not success:
        print("Error: pip not found. Please install pip first.")
        return False
    
    print("Upgrading pip...")
    success, _, _ = run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    if success:
        print("pip upgraded successfully")
    else:
        print("Warning: Could not upgrade pip, continuing anyway...")
    
    return True


def install_base_dependencies():
    """Install core dependencies required for all platforms"""
    print("Installing base dependencies...")
    
    base_packages = [
        "numpy>=1.21.0",
        "scipy>=1.7.0", 
        "matplotlib>=3.5.0",
        "scikit-image>=0.19.0",
        "h5py>=3.7.0",
        "tqdm>=4.62.0",
        "requests>=2.28.0"
    ]
    
    for package in base_packages:
        print(f"Installing {package}...")
        success, stdout, stderr = run_command([
            sys.executable, "-m", "pip", "install", package
        ])
        if not success:
            print(f"Warning: Failed to install {package}")
            print(f"Error: {stderr}")
    
    print("Base dependencies installation complete")


def install_optional_dependencies():
    """Install optional dependencies for enhanced features"""
    print("Installing optional dependencies...")
    
    optional_packages = [
        ("nibabel>=4.0.0", "Medical image format support"),
        ("soundfile>=0.10.0", "Audio file export"),
        ("opencv-python>=4.6.0", "Image processing"),
        ("plotly>=5.0.0", "Interactive plotting"),
        ("ipywidgets>=8.0.0", "Jupyter widgets"),
        ("numba>=0.56.0", "Performance acceleration"),
    ]
    
    for package, description in optional_packages:
        print(f"Installing {package} ({description})...")
        success, stdout, stderr = run_command([
            sys.executable, "-m", "pip", "install", package
        ])
        if not success:
            print(f"Warning: Failed to install {package} - {description} will not be available")


def install_platform_specific():
    """Install platform-specific dependencies"""
    platform_name = detect_platform()
    print(f"Installing platform-specific dependencies for {platform_name}...")
    
    if platform_name == 'macos':
        install_macos_dependencies()
    elif platform_name == 'linux':
        install_linux_dependencies()
    elif platform_name == 'windows':
        install_windows_dependencies()
    else:
        print("Warning: Unknown platform, skipping platform-specific dependencies")


def install_macos_dependencies():
    """Install macOS-specific dependencies"""
    print("Installing macOS-specific packages...")
    
    # Try to install PyTorch with MPS support for Apple Silicon
    try:
        import platform
        if platform.processor() == 'arm':  # Apple Silicon
            print("Detected Apple Silicon, installing PyTorch with MPS support...")
            success, _, _ = run_command([
                sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"
            ])
            if success:
                print("PyTorch with Apple Silicon support installed")
            else:
                print("Warning: Could not install PyTorch with MPS support")
    except:
        pass
    
    # VTK and PyVista for 3D visualization
    packages = [
        "vtk>=9.0.0",
        "pyvista>=0.40.0"
    ]
    
    for package in packages:
        success, _, _ = run_command([sys.executable, "-m", "pip", "install", package])
        if success:
            print(f"Installed {package}")
        else:
            print(f"Warning: Could not install {package}")


def install_linux_dependencies():
    """Install Linux-specific dependencies"""
    print("Installing Linux-specific packages...")
    
    # Install system dependencies if possible
    distro_commands = [
        ("apt-get", ["sudo", "apt-get", "update"], ["sudo", "apt-get", "install", "-y", "python3-tk", "libasound2-dev"]),
        ("yum", ["sudo", "yum", "update"], ["sudo", "yum", "install", "-y", "tkinter", "alsa-lib-devel"]),
        ("pacman", ["sudo", "pacman", "-Sy"], ["sudo", "pacman", "-S", "--noconfirm", "tk", "alsa-lib"])
    ]
    
    for manager, update_cmd, install_cmd in distro_commands:
        success, _, _ = run_command(["which", manager])
        if success:
            print(f"Detected {manager} package manager")
            print("Installing system dependencies...")
            run_command(update_cmd, capture_output=False)
            run_command(install_cmd, capture_output=False)
            break
    
    # Python packages
    packages = [
        "vtk>=9.0.0",
        "pyvista>=0.40.0",
        "torch>=1.12.0",
        "torchvision>=0.13.0"
    ]
    
    for package in packages:
        success, _, _ = run_command([sys.executable, "-m", "pip", "install", package])
        if success:
            print(f"Installed {package}")
        else:
            print(f"Warning: Could not install {package}")


def install_windows_dependencies():
    """Install Windows-specific dependencies"""
    print("Installing Windows-specific packages...")
    
    # Windows-specific packages
    packages = [
        "pywin32>=304",
        "vtk>=9.0.0",
        "pyvista>=0.40.0",
        "torch>=1.12.0",
        "torchvision>=0.13.0"
    ]
    
    for package in packages:
        success, _, _ = run_command([sys.executable, "-m", "pip", "install", package])
        if success:
            print(f"Installed {package}")
        else:
            print(f"Warning: Could not install {package}")


def create_minimal_requirements():
    """Create a minimal requirements.txt file"""
    minimal_requirements = """# Minimal requirements for XACT Simulation
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0
scikit-image>=0.19.0
h5py>=3.7.0
tqdm>=4.62.0

# Optional but recommended
nibabel>=4.0.0
soundfile>=0.10.0
numba>=0.56.0
"""
    
    with open("requirements_minimal.txt", "w") as f:
        f.write(minimal_requirements)
    
    print("Created requirements_minimal.txt")


def verify_installation():
    """Verify that key packages can be imported"""
    print("Verifying installation...")
    
    required_packages = [
        'numpy',
        'scipy', 
        'matplotlib',
        'skimage',
        'h5py'
    ]
    
    optional_packages = [
        'nibabel',
        'soundfile',
        'numba',
        'torch'
    ]
    
    all_good = True
    
    # Check required packages
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - REQUIRED")
            all_good = False
    
    # Check optional packages
    for package in optional_packages:
        try:
            __import__(package)
            print(f"✓ {package} (optional)")
        except ImportError:
            print(f"- {package} (optional, not installed)")
    
    return all_good


def setup_project_structure():
    """Create necessary project directories"""
    print("Setting up project structure...")
    
    directories = [
        "results",
        "results/archive", 
        "logs",
        "data",
        "cache",
        "sensors"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"Created directory: {dir_name}")


def main():
    """Main setup function"""
    print("XACT Simulation Setup")
    print("=" * 50)
    print(f"Platform: {detect_platform()}")
    print(f"Python: {sys.version}")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check pip
    if not check_pip():
        sys.exit(1)
    
    # Install dependencies
    try:
        install_base_dependencies()
        install_optional_dependencies()
        install_platform_specific()
    except KeyboardInterrupt:
        print("\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during installation: {e}")
        sys.exit(1)
    
    # Setup project structure
    setup_project_structure()
    
    # Create minimal requirements file
    create_minimal_requirements()
    
    # Verify installation
    print("\n" + "=" * 50)
    if verify_installation():
        print("Setup completed successfully!")
        print("\nYou can now run:")
        print("  python xact_demo.py       # Run the main demo")
        print("  python gui.py             # Launch the GUI")
        print("  python sensor_interface.py # Test sensor interface")
    else:
        print("Setup completed with some missing required packages.")
        print("Please install missing packages manually using:")
        print("  pip install -r requirements_minimal.txt")
    
    print("\nSetup complete!")


if __name__ == "__main__":
    main() 