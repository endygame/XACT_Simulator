#!/usr/bin/env python3
"""
XACT Simulation GUI

A comprehensive GUI for X-ray-induced Acoustic Computed Tomography simulation
with anatomical data integration and customizable scan parameters.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import threading
import queue
import os
import sys
from pathlib import Path
import time

# Set matplotlib to use non-interactive backend to avoid threading issues
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xact_demo import XACTDemo
from anatomy_loader import AnatomyLoader

class XACTGUI:
    """Main GUI class for XACT simulation"""

    def __init__(self, root):
    self.root = root
    self.root.title("XACT Simulation - X-ray Acoustic Computed Tomography")
    self.root.geometry("1200x800")

    # Initialize components
    self.demo = XACTDemo()
    self.anatomy_loader = AnatomyLoader()

    # Threading
    self.simulation_thread = None
    self.message_queue = queue.Queue()
    self.simulation_running = False

    # Create GUI
    self.create_gui()

    # Start message processing
    self.process_messages()

    def create_gui(self):
    """Create the main GUI interface"""

    # Create notebook for tabs
    self.notebook = ttk.Notebook(self.root)
    self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Create tabs
    self.create_data_tab()
    self.create_simulation_tab()
    self.create_results_tab()
    self.create_preview_tab()

    # Load available datasets
    self.load_available_datasets()

    def create_data_tab(self):
    """Create the data management tab"""

    data_frame = ttk.Frame(self.notebook)
    self.notebook.add(data_frame, text=" Data Management")

    # Title
    title_label = ttk.Label(data_frame, text="Anatomical Data Management", 
    font=("Arial", 16, "bold"))
    title_label.pack(pady=10)

    # Data source selection
    source_frame = ttk.LabelFrame(data_frame, text="Data Source", padding=10)
    source_frame.pack(fill=tk.X, padx=10, pady=5)

    self.data_source = tk.StringVar(value="xcat")
    ttk.Radiobutton(source_frame, text="XCAT Thorax Phantom (Recommended)", 
    variable=self.data_source, value="xcat",
    command=self.on_data_source_change).pack(anchor=tk.W)
    ttk.Radiobutton(source_frame, text="Use Existing Data", 
    variable=self.data_source, value="existing",
    command=self.on_data_source_change).pack(anchor=tk.W)
    ttk.Radiobutton(source_frame, text="Create Synthetic Data", 
    variable=self.data_source, value="synthetic",
    command=self.on_data_source_change).pack(anchor=tk.W)

    # Dataset information
    info_frame = ttk.LabelFrame(data_frame, text="Dataset Information", padding=10)
    info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    # Treeview for datasets
    columns = ("Name", "Type", "Size", "Source")
    self.dataset_tree = ttk.Treeview(info_frame, columns=columns, show="headings", height=12)

    for col in columns:
    self.dataset_tree.heading(col, text=col)
    self.dataset_tree.column(col, width=200)

    # Scrollbars
    v_scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.dataset_tree.yview)
    h_scrollbar = ttk.Scrollbar(info_frame, orient=tk.HORIZONTAL, command=self.dataset_tree.xview)
    self.dataset_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

    # Pack treeview and scrollbars
    self.dataset_tree.grid(row=0, column=0, sticky="nsew")
    v_scrollbar.grid(row=0, column=1, sticky="ns")
    h_scrollbar.grid(row=1, column=0, sticky="ew")

    info_frame.grid_rowconfigure(0, weight=1)
    info_frame.grid_columnconfigure(0, weight=1)

    # Buttons
    button_frame = ttk.Frame(data_frame)
    button_frame.pack(fill=tk.X, padx=10, pady=5)

    ttk.Button(button_frame, text="Refresh Data", 
    command=self.load_available_datasets).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Add Data File", 
    command=self.add_data_file).pack(side=tk.LEFT, padx=5)

    def create_simulation_tab(self):
    """Create the simulation control tab"""

    sim_frame = ttk.Frame(self.notebook)
    self.notebook.add(sim_frame, text=" Simulation Control")

    # Title
    title_label = ttk.Label(sim_frame, text="XACT Simulation Control", 
    font=("Arial", 16, "bold"))
    title_label.pack(pady=10)

    # Create main container with scrollable area
    main_container = ttk.Frame(sim_frame)
    main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    # Left side - Parameters
    left_frame = ttk.Frame(main_container)
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

    # Parameters frame with scrollbar
    canvas = tk.Canvas(left_frame)
    scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # X-ray parameters
    xray_frame = ttk.LabelFrame(scrollable_frame, text="X-ray Parameters", padding=10)
    xray_frame.pack(fill=tk.X, pady=5)

    # Beam energy
    ttk.Label(xray_frame, text="Beam Energy (keV):").grid(row=0, column=0, sticky=tk.W, pady=2)
    self.beam_energy = tk.DoubleVar(value=120.0)
    energy_scale = ttk.Scale(xray_frame, from_=50, to=300, variable=self.beam_energy, 
    orient=tk.HORIZONTAL, length=200)
    energy_scale.grid(row=0, column=1, padx=5)
    self.energy_label = ttk.Label(xray_frame, text="120.0")
    self.energy_label.grid(row=0, column=2, padx=5)
    self.beam_energy.trace('w', lambda *args: self.energy_label.config(text=f"{self.beam_energy.get():.1f}"))

    # Number of projections
    ttk.Label(xray_frame, text="Number of Projections:").grid(row=1, column=0, sticky=tk.W, pady=2)
    self.num_projections = tk.IntVar(value=36)
    proj_scale = ttk.Scale(xray_frame, from_=12, to=180, variable=self.num_projections, 
    orient=tk.HORIZONTAL, length=200)
    proj_scale.grid(row=1, column=1, padx=5)
    self.proj_label = ttk.Label(xray_frame, text="36")
    self.proj_label.grid(row=1, column=2, padx=5)
    self.num_projections.trace('w', lambda *args: self.proj_label.config(text=f"{self.num_projections.get()}"))

    # Acoustic sensor parameters
    sensor_frame = ttk.LabelFrame(scrollable_frame, text="Acoustic Sensor Array", padding=10)
    sensor_frame.pack(fill=tk.X, pady=5)

    # Number of sensors
    ttk.Label(sensor_frame, text="Number of Sensors:").grid(row=0, column=0, sticky=tk.W, pady=2)
    self.num_sensors = tk.IntVar(value=128)
    sensor_scale = ttk.Scale(sensor_frame, from_=32, to=512, variable=self.num_sensors, 
    orient=tk.HORIZONTAL, length=200)
    sensor_scale.grid(row=0, column=1, padx=5)
    self.sensor_label = ttk.Label(sensor_frame, text="128")
    self.sensor_label.grid(row=0, column=2, padx=5)
    self.num_sensors.trace('w', lambda *args: self.sensor_label.config(text=f"{self.num_sensors.get()}"))

    # Center frequency
    ttk.Label(sensor_frame, text="Center Frequency (MHz):").grid(row=1, column=0, sticky=tk.W, pady=2)
    self.center_freq = tk.DoubleVar(value=2.5)
    freq_scale = ttk.Scale(sensor_frame, from_=1.0, to=10.0, variable=self.center_freq, 
    orient=tk.HORIZONTAL, length=200)
    freq_scale.grid(row=1, column=1, padx=5)
    self.freq_label = ttk.Label(sensor_frame, text="2.5")
    self.freq_label.grid(row=1, column=2, padx=5)
    self.center_freq.trace('w', lambda *args: self.freq_label.config(text=f"{self.center_freq.get():.1f}"))

    # Sample rate
    ttk.Label(sensor_frame, text="Sample Rate (kHz):").grid(row=2, column=0, sticky=tk.W, pady=2)
    self.sample_rate = tk.IntVar(value=44)
    rate_scale = ttk.Scale(sensor_frame, from_=20, to=100, variable=self.sample_rate, 
    orient=tk.HORIZONTAL, length=200)
    rate_scale.grid(row=2, column=1, padx=5)
    self.rate_label = ttk.Label(sensor_frame, text="44")
    self.rate_label.grid(row=2, column=2, padx=5)
    self.sample_rate.trace('w', lambda *args: self.rate_label.config(text=f"{self.sample_rate.get()}"))

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Right side - Control and Status
    right_frame = ttk.Frame(main_container)
    right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))

    # Status frame
    status_frame = ttk.LabelFrame(right_frame, text="Simulation Status", padding=10)
    status_frame.pack(fill=tk.X, pady=5)

    self.status_label = ttk.Label(status_frame, text="", font=("Arial", 12))
    self.status_label.pack()

    # Progress bar
    self.progress = ttk.Progressbar(status_frame, mode='indeterminate')
    self.progress.pack(fill=tk.X, pady=5)

    # Control buttons
    control_frame = ttk.Frame(right_frame)
    control_frame.pack(fill=tk.X, pady=10)

    self.start_button = ttk.Button(control_frame, text="Start Simulation", 
    command=self.start_simulation, width=15)
    self.start_button.pack(pady=2)

    self.stop_button = ttk.Button(control_frame, text="Stop Simulation", 
    command=self.stop_simulation, state=tk.DISABLED, width=15)
    self.stop_button.pack(pady=2)

    ttk.Button(control_frame, text="Reset", command=self.reset_simulation, width=15).pack(pady=2)

    # Current dataset info
    info_frame = ttk.LabelFrame(right_frame, text="Current Dataset", padding=10)
    info_frame.pack(fill=tk.BOTH, expand=True, pady=5)

    self.dataset_info = tk.StringVar(value="XCAT Thorax Phantom (Default)")
    info_text = ttk.Label(info_frame, textvariable=self.dataset_info, wraplength=200, justify=tk.LEFT)
    info_text.pack(fill=tk.BOTH, expand=True)

    def create_results_tab(self):
    """Create the results display tab"""

    results_frame = ttk.Frame(self.notebook)
    self.notebook.add(results_frame, text=" Results")

    # Title
    title_label = ttk.Label(results_frame, text="Simulation Results & Visualizations", 
    font=("Arial", 16, "bold"))
    title_label.pack(pady=10)

    # Create paned window for layout
    paned = ttk.PanedWindow(results_frame, orient=tk.HORIZONTAL)
    paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    # Left side - File list
    left_frame = ttk.Frame(paned)
    paned.add(left_frame, weight=1)

    # File list
    file_frame = ttk.LabelFrame(left_frame, text="Generated Files", padding=10)
    file_frame.pack(fill=tk.BOTH, expand=True)

    self.file_listbox = tk.Listbox(file_frame, height=15)
    file_scrollbar = ttk.Scrollbar(file_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
    self.file_listbox.configure(yscrollcommand=file_scrollbar.set)

    self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    file_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Bind double-click to view file
    self.file_listbox.bind('<Double-Button-1>', self.view_selected_file)

    # File buttons
    file_button_frame = ttk.Frame(left_frame)
    file_button_frame.pack(fill=tk.X, pady=5)

    ttk.Button(file_button_frame, text="Refresh Files", 
    command=self.refresh_files).pack(side=tk.LEFT, padx=2)
    ttk.Button(file_button_frame, text="Open File", 
    command=self.view_selected_file).pack(side=tk.LEFT, padx=2)
    ttk.Button(file_button_frame, text="Open Folder", 
    command=self.open_results_folder).pack(side=tk.LEFT, padx=2)

    # Right side - Content display
    right_frame = ttk.Frame(paned)
    paned.add(right_frame, weight=2)

    # Display area
    display_frame = ttk.LabelFrame(right_frame, text="File Content", padding=10)
    display_frame.pack(fill=tk.BOTH, expand=True)

    # Create notebook for different content types
    self.content_notebook = ttk.Notebook(display_frame)
    self.content_notebook.pack(fill=tk.BOTH, expand=True)

    # Text content tab
    text_frame = ttk.Frame(self.content_notebook)
    self.content_notebook.add(text_frame, text=" Text/Log")

    self.results_text = scrolledtext.ScrolledText(text_frame, height=20, width=60)
    self.results_text.pack(fill=tk.BOTH, expand=True)

    # Image content tab
    image_frame = ttk.Frame(self.content_notebook)
    self.content_notebook.add(image_frame, text="️ Images")

    # Create matplotlib figure for image display
    self.image_figure = Figure(figsize=(8, 6), dpi=100)
    self.image_canvas = FigureCanvasTkAgg(self.image_figure, image_frame)
    self.image_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Image info
    self.image_info_label = ttk.Label(image_frame, text="No image selected", 
    font=("Arial", 10))
    self.image_info_label.pack(pady=5)

    # Refresh files on startup
    self.refresh_files()

    def create_preview_tab(self):
    """Create the preview tab for 3D visualization"""

    preview_frame = ttk.Frame(self.notebook)
    self.notebook.add(preview_frame, text="️ Preview")

    # Title
    title_label = ttk.Label(preview_frame, text="3D Pre-Scan Setup Preview", 
    font=("Arial", 16, "bold"))
    title_label.pack(pady=10)

    # Description
    desc_label = ttk.Label(preview_frame, 
    text="3D visualization of anatomical model, sensor array, and X-ray source setup",
    font=("Arial", 10))
    desc_label.pack(pady=5)

    # Preview display
    display_frame = ttk.LabelFrame(preview_frame, text="3D Visualization", padding=10)
    display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    # Create matplotlib figure for 3D preview
    self.preview_figure = Figure(figsize=(12, 8), dpi=100)
    self.preview_canvas = FigureCanvasTkAgg(self.preview_figure, display_frame)
    self.preview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Preview info
    self.preview_info_label = ttk.Label(display_frame, 
    text="No preview available - run simulation to generate 3D setup view",
    font=("Arial", 10))
    self.preview_info_label.pack(pady=5)

    # Buttons
    button_frame = ttk.Frame(preview_frame)
    button_frame.pack(fill=tk.X, padx=10, pady=5)

    ttk.Button(button_frame, text="Generate Preview", 
    command=self.generate_preview).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Refresh Preview", 
    command=self.refresh_preview).pack(side=tk.LEFT, padx=5)

    def load_available_datasets(self):
    """Load and display available datasets"""

    # Clear existing items
    for item in self.dataset_tree.get_children():
    self.dataset_tree.delete(item)

    try:
    # Check for XCAT phantom
    xcat_file = "anatomical_data/xcat_thorax_phantom.nii.gz"
    if os.path.exists(xcat_file):
    size = os.path.getsize(xcat_file) / (1024*1024) # MB
    self.dataset_tree.insert("", "end", values=(
    "XCAT Thorax Phantom", "CT", f"{size:.1f} MB", "XCAT Phantom"
    ))

    # Check anatomical_data directory
    if os.path.exists("anatomical_data"):
    for file in os.listdir("anatomical_data"):
    if file.endswith(('.nii.gz', '.nii')):
    filepath = os.path.join("anatomical_data", file)
    size = os.path.getsize(filepath) / (1024*1024) # MB
    name = file.replace('.nii.gz', '').replace('.nii', '')
    name = name.replace('_', ' ').title()

    if "xcat" not in file.lower(): # Don't duplicate XCAT
    self.dataset_tree.insert("", "end", values=(
    name, "Unknown", f"{size:.1f} MB", "Local File"
    ))

    self.log_message(f"Found {len(self.dataset_tree.get_children())} datasets")

    except Exception as e:
    self.log_message(f"Error loading datasets: {e}")

    def add_data_file(self):
    """Add a new data file"""

    filetypes = [
    ('NIfTI files', '*.nii *.nii.gz'),
    ('All files', '*.*')
    ]

    filename = filedialog.askopenfilename(
    title="Select Anatomical Data File",
    filetypes=filetypes
    )

    if filename:
    try:
    # Copy to anatomical_data directory
    os.makedirs("anatomical_data", exist_ok=True)
    import shutil
    dest_path = os.path.join("anatomical_data", os.path.basename(filename))
    shutil.copy2(filename, dest_path)

    self.log_message(f"Added data file: {os.path.basename(filename)}")
    self.load_available_datasets()

    except Exception as e:
    messagebox.showerror("Error", f"Could not add data file: {e}")

    def on_data_source_change(self):
    """Handle data source change"""

    source = self.data_source.get()
    if source == "xcat":
    self.dataset_info.set("XCAT Thorax Phantom\n• 256×256×200 voxels\n• 1.5mm resolution\n• 10 tissue types")
    self.log_message("XCAT thorax phantom selected - realistic 3D thorax anatomy")
    elif source == "existing":
    self.dataset_info.set("Using Existing Data\n• Select from available datasets\n• Various formats supported")
    self.log_message("Existing data mode selected")
    elif source == "synthetic":
    self.dataset_info.set("Synthetic Data\n• Generated programmatically\n• Simple geometric phantoms")
    self.log_message("Synthetic data mode selected")

    def start_simulation(self):
    """Start the XACT simulation"""

    if self.simulation_running:
    messagebox.showwarning("Warning", "Simulation is already running!")
    return

    # Get parameters
    params = {
    'data_source': self.data_source.get(),
    'beam_energy': self.beam_energy.get(),
    'num_projections': self.num_projections.get(),
    'num_sensors': self.num_sensors.get(),
    'center_freq': self.center_freq.get(),
    'sample_rate': self.sample_rate.get() * 1000, # Convert to Hz
    }

    # Update UI
    self.simulation_running = True
    self.start_button.config(state=tk.DISABLED)
    self.stop_button.config(state=tk.NORMAL)
    self.progress.start()
    self.status_label.config(text="Running simulation...")

    # Clear results
    self.results_text.delete(1.0, tk.END)

    # Start simulation thread
    self.simulation_thread = threading.Thread(target=self.run_simulation, args=(params,))
    self.simulation_thread.daemon = True
    self.simulation_thread.start()

    self.log_message("Simulation started...")

    def stop_simulation(self):
    """Stop the simulation"""

    self.simulation_running = False
    self.start_button.config(state=tk.NORMAL)
    self.stop_button.config(state=tk.DISABLED)
    self.progress.stop()
    self.status_label.config(text="Stopped")

    self.log_message("Simulation stopped by user")

    def reset_simulation(self):
    """Reset simulation state"""

    if self.simulation_running:
    self.stop_simulation()

    # Reset UI
    self.results_text.delete(1.0, tk.END)
    self.status_label.config(text="")

    # Clear preview
    self.preview_figure.clear()
    self.preview_canvas.draw()
    self.preview_info_label.config(text="No preview available")

    # Clear image display
    self.image_figure.clear()
    self.image_canvas.draw()
    self.image_info_label.config(text="No image selected")

    self.log_message("Simulation reset")

    def run_simulation(self, params):
    """Run the simulation in background thread"""

    try:
    # Load dataset based on source
    source = params['data_source']

    if source == "xcat":
    self.message_queue.put({'type': 'log', 'text': 'Loading XCAT thorax phantom...'})
    dataset = self.anatomy_loader.load_xcat_thorax()
    elif source == "existing":
    self.message_queue.put({'type': 'log', 'text': 'Loading existing data...'})
    # Use first available dataset
    datasets = list(self.anatomy_loader.list_available_datasets())
    if datasets:
    dataset = self.anatomy_loader.load_dataset(datasets[0])
    else:
    self.message_queue.put({'type': 'log', 'text': 'No existing data found, using XCAT...'})
    dataset = self.anatomy_loader.load_xcat_thorax()
    else: # synthetic
    self.message_queue.put({'type': 'log', 'text': 'Creating synthetic data...'})
    dataset = self.anatomy_loader.create_synthetic_data(shape=(128, 128, 128), mode='thorax')

    if not dataset:
    raise Exception("Failed to load dataset")

    # Set dataset
    self.demo.current_dataset = dataset

    # Apply parameters
    self.demo.setup_acoustic_sensors(num_sensors=params['num_sensors'])
    self.demo.sample_rate = params['sample_rate']

    # Update scan parameters
    self.demo.scan_params = {
    'beam_energy': params['beam_energy'],
    'num_projections': params['num_projections'],
    'center_freq': params['center_freq'],
    'scan_duration': 1e-6, # 1 microsecond
    }

    self.message_queue.put({'type': 'log', 'text': f"Dataset loaded: {dataset['metadata']['description']}"})

    # Generate 3D preview
    self.message_queue.put({'type': 'log', 'text': 'Generating 3D preview...'})
    self.demo.show_3d_anatomy()

    # Run simulation
    self.message_queue.put({'type': 'log', 'text': 'Running XACT simulation...'})
    self.demo.run_xact_simulation()

    # Show results
    self.message_queue.put({'type': 'log', 'text': 'Generating results...'})
    self.demo.show_xact_results()

    self.message_queue.put({'type': 'log', 'text': 'Simulation completed successfully!'})
    self.message_queue.put({'type': 'complete'})

    except Exception as e:
    self.message_queue.put({'type': 'log', 'text': f'Simulation error: {e}'})
    self.message_queue.put({'type': 'error', 'text': str(e)})

    def generate_preview(self):
    """Generate 3D preview"""

    try:
    if not hasattr(self.demo, 'current_dataset') or not self.demo.current_dataset:
    # Load default dataset
    self.demo.current_dataset = self.anatomy_loader.load_xcat_thorax()

    self.demo.show_3d_anatomy()
    self.refresh_preview()
    self.log_message("3D preview generated")

    except Exception as e:
    self.log_message(f"Error generating preview: {e}")
    messagebox.showerror("Error", f"Could not generate preview: {e}")

    def refresh_preview(self):
    """Refresh the preview display"""

    # Look for preview images
    preview_files = [f for f in os.listdir('.') if f.startswith('3d_prescan_setup_') and f.endswith('.png')]

    if preview_files:
    # Display the most recent preview
    preview_file = max(preview_files, key=os.path.getmtime)
    self.display_preview_image(preview_file)
    else:
    self.preview_info_label.config(text="No preview images found")

    def display_preview_image(self, image_file):
    """Display preview image"""

    try:
    # Clear previous figure
    self.preview_figure.clear()

    # Load and display image
    img = plt.imread(image_file)
    ax = self.preview_figure.add_subplot(111)
    ax.imshow(img)
    ax.axis('off')

    # Update canvas
    self.preview_canvas.draw()

    # Update info
    self.preview_info_label.config(text=f"Displaying: {image_file}")

    except Exception as e:
    self.log_message(f"Error displaying preview: {e}")

    def refresh_files(self):
    """Refresh the file list"""

    self.file_listbox.delete(0, tk.END)

    # Get all relevant files
    extensions = ['.png', '.jpg', '.jpeg', '.wav', '.txt', '.log']
    files = []

    for ext in extensions:
    files.extend([f for f in os.listdir('.') if f.endswith(ext)])

    # Sort by modification time (newest first)
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    for file in files:
    self.file_listbox.insert(tk.END, file)

    self.log_message(f"Found {len(files)} result files")

    def view_selected_file(self, event=None):
    """View the selected file"""

    selection = self.file_listbox.curselection()
    if not selection:
    return

    filename = self.file_listbox.get(selection[0])

    try:
    if filename.endswith(('.png', '.jpg', '.jpeg')):
    self.display_image(filename)
    self.content_notebook.select(1) # Switch to image tab
    elif filename.endswith(('.txt', '.log')):
    self.display_text_file(filename)
    self.content_notebook.select(0) # Switch to text tab
    elif filename.endswith('.wav'):
    self.log_message(f"Audio file: {filename} (use external player)")
    else:
    self.log_message(f"Unknown file type: {filename}")

    except Exception as e:
    self.log_message(f"Error viewing file {filename}: {e}")
    messagebox.showerror("Error", f"Could not view file: {e}")

    def display_image(self, filename):
    """Display an image file"""

    try:
    # Clear previous image
    self.image_figure.clear()

    # Load and display image
    img = plt.imread(filename)
    ax = self.image_figure.add_subplot(111)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(filename, fontsize=12)

    # Update canvas
    self.image_canvas.draw()

    # Update info
    file_size = os.path.getsize(filename) / 1024 # KB
    mod_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(filename)))

    info_text = f"File: {filename} | Size: {file_size:.1f} KB | Modified: {mod_time}"
    self.image_info_label.config(text=info_text)

    except Exception as e:
    self.log_message(f"Error displaying image {filename}: {e}")

    def display_text_file(self, filename):
    """Display a text file"""

    try:
    with open(filename, 'r', encoding='utf-8') as f:
    content = f.read()

    self.results_text.delete(1.0, tk.END)
    self.results_text.insert(1.0, content)

    except Exception as e:
    self.log_message(f"Error reading text file {filename}: {e}")

    def open_results_folder(self):
    """Open the results folder in file manager"""

    try:
    import subprocess
    import platform

    if platform.system() == "Darwin": # macOS
    subprocess.run(["open", "."])
    elif platform.system() == "Windows":
    subprocess.run(["explorer", "."])
    else: # Linux
    subprocess.run(["xdg-open", "."])

    except Exception as e:
    self.log_message(f"Error opening folder: {e}")

    def log_message(self, message):
    """Log a message to the results text area"""

    timestamp = time.strftime("%H:%M:%S")
    full_message = f"[{timestamp}] {message}\n"

    self.results_text.insert(tk.END, full_message)
    self.results_text.see(tk.END)

    def process_messages(self):
    """Process messages from simulation thread"""

    try:
    while True:
    message = self.message_queue.get_nowait()

    if message['type'] == 'log':
    self.log_message(message['text'])
    elif message['type'] == 'complete':
    self.simulation_running = False
    self.start_button.config(state=tk.NORMAL)
    self.stop_button.config(state=tk.DISABLED)
    self.progress.stop()
    self.status_label.config(text="Completed")
    self.refresh_files()
    self.refresh_preview()
    elif message['type'] == 'error':
    self.simulation_running = False
    self.start_button.config(state=tk.NORMAL)
    self.stop_button.config(state=tk.DISABLED)
    self.progress.stop()
    self.status_label.config(text="Error")
    messagebox.showerror("Simulation Error", message['text'])

    except queue.Empty:
    pass

    # Schedule next check
    self.root.after(100, self.process_messages)

def main():
    """Main function to run the GUI"""

    root = tk.Tk()
    app = XACTGUI(root)

    try:
    root.mainloop()
    except KeyboardInterrupt:
    print("\nGUI closed by user")

if __name__ == "__main__":
    main() 