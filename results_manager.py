"""
Results Manager for XACT Simulation

Organizes simulation results into dated folders with timestamps for better tracking.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import json


class ResultsManager:
    """Manages organization and storage of XACT simulation results"""
    
    def __init__(self, base_results_dir: str = "results"):
        self.base_results_dir = Path(base_results_dir)
        self.current_run_dir = None
        self.run_metadata = {}
        
        # Create base results directory if it doesn't exist
        self.base_results_dir.mkdir(exist_ok=True)
    
    def create_run_directory(self, run_name: Optional[str] = None) -> Path:
        """
        Create a new run directory with timestamp
        
        Args:
            run_name: Optional custom name for the run
            
        Returns:
            Path to the created run directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if run_name:
            # Sanitize run name
            run_name = "".join(c for c in run_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            run_name = run_name.replace(' ', '_')
            dir_name = f"{timestamp}_{run_name}"
        else:
            dir_name = f"{timestamp}_xact_run"
        
        self.current_run_dir = self.base_results_dir / dir_name
        self.current_run_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.current_run_dir / "images").mkdir(exist_ok=True)
        (self.current_run_dir / "audio").mkdir(exist_ok=True)
        (self.current_run_dir / "data").mkdir(exist_ok=True)
        (self.current_run_dir / "logs").mkdir(exist_ok=True)
        
        # Initialize metadata
        self.run_metadata = {
            "run_name": run_name or "xact_run",
            "timestamp": timestamp,
            "start_time": datetime.now().isoformat(),
            "version": "1.0",
            "files": {
                "images": [],
                "audio": [],
                "data": [],
                "logs": []
            }
        }
        
        return self.current_run_dir
    
    def save_image(self, image_path: str, category: str = "general") -> Path:
        """
        Save an image file to the current run directory
        
        Args:
            image_path: Path to the image file
            category: Category subdirectory (e.g., 'anatomy', 'results', 'setup')
            
        Returns:
            Path where the file was saved
        """
        if self.current_run_dir is None:
            self.create_run_directory()
        
        # Create category subdirectory
        category_dir = self.current_run_dir / "images" / category
        category_dir.mkdir(exist_ok=True)
        
        # Copy file with descriptive name
        source_path = Path(image_path)
        if source_path.exists():
            dest_path = category_dir / source_path.name
            shutil.copy2(source_path, dest_path)
            
            # Update metadata
            self.run_metadata["files"]["images"].append({
                "original_path": str(source_path),
                "saved_path": str(dest_path.relative_to(self.current_run_dir)),
                "category": category,
                "timestamp": datetime.now().isoformat()
            })
            
            return dest_path
        else:
            raise FileNotFoundError(f"Image file not found: {image_path}")
    
    def save_audio(self, audio_path: str, category: str = "sensors") -> Path:
        """
        Save an audio file to the current run directory
        
        Args:
            audio_path: Path to the audio file
            category: Category subdirectory (e.g., 'sensors', 'mixed', 'processed')
            
        Returns:
            Path where the file was saved
        """
        if self.current_run_dir is None:
            self.create_run_directory()
        
        # Create category subdirectory
        category_dir = self.current_run_dir / "audio" / category
        category_dir.mkdir(exist_ok=True)
        
        # Copy file
        source_path = Path(audio_path)
        if source_path.exists():
            dest_path = category_dir / source_path.name
            shutil.copy2(source_path, dest_path)
            
            # Update metadata
            self.run_metadata["files"]["audio"].append({
                "original_path": str(source_path),
                "saved_path": str(dest_path.relative_to(self.current_run_dir)),
                "category": category,
                "timestamp": datetime.now().isoformat()
            })
            
            return dest_path
        else:
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    def save_data(self, data_path: str, category: str = "simulation") -> Path:
        """
        Save a data file to the current run directory
        
        Args:
            data_path: Path to the data file
            category: Category subdirectory (e.g., 'simulation', 'anatomy', 'parameters')
            
        Returns:
            Path where the file was saved
        """
        if self.current_run_dir is None:
            self.create_run_directory()
        
        # Create category subdirectory
        category_dir = self.current_run_dir / "data" / category
        category_dir.mkdir(exist_ok=True)
        
        # Copy file
        source_path = Path(data_path)
        if source_path.exists():
            dest_path = category_dir / source_path.name
            shutil.copy2(source_path, dest_path)
            
            # Update metadata
            self.run_metadata["files"]["data"].append({
                "original_path": str(source_path),
                "saved_path": str(dest_path.relative_to(self.current_run_dir)),
                "category": category,
                "timestamp": datetime.now().isoformat()
            })
            
            return dest_path
        else:
            raise FileNotFoundError(f"Data file not found: {data_path}")
    
    def save_log(self, log_content: str, log_name: str = "simulation_log.txt") -> Path:
        """
        Save log content to the current run directory
        
        Args:
            log_content: Content to write to log
            log_name: Name of the log file
            
        Returns:
            Path to the saved log file
        """
        if self.current_run_dir is None:
            self.create_run_directory()
        
        log_path = self.current_run_dir / "logs" / log_name
        
        with open(log_path, 'w') as f:
            f.write(log_content)
        
        # Update metadata
        self.run_metadata["files"]["logs"].append({
            "saved_path": str(log_path.relative_to(self.current_run_dir)),
            "timestamp": datetime.now().isoformat(),
            "size_bytes": len(log_content.encode('utf-8'))
        })
        
        return log_path
    
    def save_parameters(self, parameters: Dict[str, Any]) -> Path:
        """
        Save simulation parameters as JSON
        
        Args:
            parameters: Dictionary of simulation parameters
            
        Returns:
            Path to the saved parameters file
        """
        if self.current_run_dir is None:
            self.create_run_directory()
        
        params_path = self.current_run_dir / "data" / "parameters.json"
        
        with open(params_path, 'w') as f:
            json.dump(parameters, f, indent=2, default=str)
        
        # Update metadata
        self.run_metadata["files"]["data"].append({
            "saved_path": str(params_path.relative_to(self.current_run_dir)),
            "category": "parameters",
            "timestamp": datetime.now().isoformat(),
            "type": "json"
        })
        
        return params_path
    
    def finalize_run(self, summary: Optional[str] = None) -> Path:
        """
        Finalize the current run by saving metadata and summary
        
        Args:
            summary: Optional summary text for the run
            
        Returns:
            Path to the metadata file
        """
        if self.current_run_dir is None:
            raise RuntimeError("No active run directory")
        
        # Add end time and summary
        self.run_metadata["end_time"] = datetime.now().isoformat()
        if summary:
            self.run_metadata["summary"] = summary
        
        # Calculate run duration
        start_time = datetime.fromisoformat(self.run_metadata["start_time"])
        end_time = datetime.fromisoformat(self.run_metadata["end_time"])
        duration = end_time - start_time
        self.run_metadata["duration_seconds"] = duration.total_seconds()
        
        # Save metadata
        metadata_path = self.current_run_dir / "run_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.run_metadata, f, indent=2, default=str)
        
        return metadata_path
    
    def cleanup_old_files(self, pattern: str = "*.png") -> int:
        """
        Clean up old result files matching pattern in the current directory
        
        Args:
            pattern: File pattern to match (e.g., '*.png', '*.wav')
            
        Returns:
            Number of files cleaned up
        """
        current_dir = Path(".")
        files_to_clean = list(current_dir.glob(pattern))
        
        cleaned_count = 0
        for file_path in files_to_clean:
            try:
                file_path.unlink()
                cleaned_count += 1
            except Exception as e:
                print(f"Could not remove {file_path}: {e}")
        
        return cleaned_count
    
    def get_run_summary(self) -> Dict[str, Any]:
        """Get summary of the current run"""
        if self.current_run_dir is None:
            return {}
        
        summary = self.run_metadata.copy()
        
        # Add file counts
        summary["file_counts"] = {
            "images": len(summary["files"]["images"]),
            "audio": len(summary["files"]["audio"]),
            "data": len(summary["files"]["data"]),
            "logs": len(summary["files"]["logs"])
        }
        
        return summary
    
    def list_previous_runs(self) -> List[Dict[str, Any]]:
        """List all previous runs with their metadata"""
        runs = []
        
        for run_dir in self.base_results_dir.iterdir():
            if run_dir.is_dir():
                metadata_file = run_dir / "run_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        metadata["run_directory"] = str(run_dir)
                        runs.append(metadata)
                    except Exception as e:
                        print(f"Could not read metadata for {run_dir}: {e}")
        
        # Sort by start time (newest first)
        runs.sort(key=lambda x: x.get("start_time", ""), reverse=True)
        
        return runs


# Convenience function for easy use
def create_results_manager(run_name: Optional[str] = None) -> ResultsManager:
    """
    Create and initialize a results manager
    
    Args:
        run_name: Optional name for the run
        
    Returns:
        Initialized ResultsManager instance
    """
    manager = ResultsManager()
    manager.create_run_directory(run_name)
    return manager


# Example usage
if __name__ == "__main__":
    # Create a test run
    manager = create_results_manager("test_run")
    
    print(f"Created run directory: {manager.current_run_dir}")
    
    # Save some test parameters
    test_params = {
        "beam_energy": 120,
        "num_sensors": 128,
        "simulation_time": 1e-6
    }
    manager.save_parameters(test_params)
    
    # Save a test log
    manager.save_log("Test simulation started\nSimulation complete\n", "test.log")
    
    # Finalize the run
    manager.finalize_run("Test run completed successfully")
    
    print("Run summary:")
    summary = manager.get_run_summary()
    for key, value in summary.items():
        if key != "files":  # Skip detailed file list for brevity
            print(f"  {key}: {value}")
    
    print(f"\nRun saved to: {manager.current_run_dir}") 