"""
Synthetic X-ray Computed Tomography (XCT) Slice Generator (Multi-Label)

This script generates synthetic 2D XCT slices that simulate microstructural features
of composite battery cathodes, specifically NMC particles in a solid electrolyte matrix.
It also generates a corresponding multi-label segmentation image.

Label mapping for the segmentation image:
- 0: Out of Reconstruction Volume
- 1: Background (Outer Ring)
- 2: Electrolyte (Inner Circle)
- 3: Cathode Particles
"""

import numpy as np
import cv2
from dataclasses import dataclass, field, asdict
from typing import Tuple, List, Dict, Optional, Any
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # Import patches for legend
from matplotlib.colors import ListedColormap, BoundaryNorm
from random import uniform, randint
from copy import deepcopy
import os
from tifffile import imwrite
from pathlib import Path
import pprint # For printing the config dictionary nicely
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import time

# --- Constants for Labels ---
LABEL_OUT_OF_RECONSTRUCTION = 0
LABEL_BACKGROUND = 1
LABEL_ELECTROLYTE = 2
LABEL_CATHODE = 3
# --------------------------

# --- Dataclasses for Parameter Structure (Type Hinting & Organization) ---
# These dataclasses now primarily serve for structure and type hints within classes.
# They are initialized from the main configuration dictionary.

@dataclass
class ImageParameters:
    size: int
    outer_circle_size: int
    inner_circle_size: int
    base_grey: int
    inner_circle_grey: int
    outer_circle_grey: int
    particle_grey_range: Tuple[int, int]
    num_particles: int
    attraction_radius: int
    attraction_strength: float
    cell_size: int


# --- Helper Function to Create Dataclasses from Dict ---
def dataclass_from_dict(cls, data: Dict[str, Any]):
    """Creates a dataclass instance from a dictionary, ignoring extra keys."""
    field_names = {f.name for f in cls.__dataclass_fields__.values()}
    filtered_data = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered_data)

# --- Particle Grid (Spatial Acceleration) ---
class ParticleGrid:
    """Spatial grid for efficient nearby particle lookups."""
    def __init__(self, size: int, cell_size: int):
        self.cell_size = max(1, cell_size) # Ensure cell_size is at least 1
        self.grid_width = size // self.cell_size + 1
        # Initialize grid as a dictionary of dictionaries for sparse storage
        self.grid: Dict[int, Dict[int, Tuple[int, int, int, int]]] = {i: {} for i in range(self.grid_width * self.grid_width)}

    def get_index(self, x: int, y: int) -> int:
        """Calculate the grid cell index for a given coordinate."""
        grid_x = x // self.cell_size
        grid_y = y // self.cell_size
        return int(grid_y * self.grid_width + grid_x)

    def get_nearby_particles(self, x: int, y: int, search_radius: int, size: int) -> List[Tuple[int, int, int, int]]:
        """Retrieve particles within a search radius of a given coordinate."""
        cells_to_check = set()
        radius_in_cells = (search_radius // self.cell_size) + 1

        center_idx = self.get_index(x, y)
        center_grid_x = (center_idx % self.grid_width)
        center_grid_y = (center_idx // self.grid_width)

        # Determine the grid cells to check based on the search radius
        for dx in range(-radius_in_cells, radius_in_cells + 1):
            for dy in range(-radius_in_cells, radius_in_cells + 1):
                grid_x = center_grid_x + dx
                grid_y = center_grid_y + dy
                # Ensure the cell is within the grid boundaries
                if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_width:
                    cells_to_check.add(grid_y * self.grid_width + grid_x)

        nearby = []
        # Collect particles from the identified cells
        for idx in cells_to_check:
            if idx in self.grid: # Check if index exists (might not if grid is sparse)
                 nearby.extend(self.grid[idx].values())
        return nearby

    def add_particle(self, particle_id: int, x: int, y: int, rx: int, ry: int):
        """Add a particle's information to the grid."""
        grid_idx = self.get_index(x, y)
        if grid_idx not in self.grid:
             self.grid[grid_idx] = {} # Initialize cell if it doesn't exist
        self.grid[grid_idx][particle_id] = (x, y, rx, ry)

# --- Particle Placement Logic ---
class ParticlePlacer:
    """Handles the placement logic for particles within the inner circle."""
    def __init__(self, params: ImageParameters):
        self.params = params
        # Initialize grid using parameters from the ImageParameters dataclass
        self.grid = ParticleGrid(params.size, params.cell_size)

    def generate_particle_dimensions(self) -> Tuple[int, int]:
        """Generate random radii for an elliptical particle based on a Gamma distribution
           derived from experimental equivalent diameters."""

        # --- Parameters from your Gamma fit ---
        gamma_shape_k = 4.5961
        gamma_scale_theta = 3.3044
        # ---------------------------------------

        # 1. Generate an equivalent diameter from the fitted Gamma distribution
        generated_diameter = np.random.gamma(shape=gamma_shape_k, scale=gamma_scale_theta, size=1)[0]

        # Ensure diameter is not unrealistically small (e.g., less than 1 pixel)
        generated_diameter = max(1.0, generated_diameter)

        # 2. Calculate a base radius from the diameter
        base_radius = int(round(generated_diameter / 2.0))

        # Ensure base_radius is at least 1 pixel
        base_radius = max(1, base_radius)

        # 3. Introduce slight eccentricity
        ecc = uniform(-0.15, 0.15)
        radius_x = max(1, int(round(base_radius * (1 + ecc)))) # Use round before int
        radius_y = max(1, int(round(base_radius * (1 - ecc)))) # Ensure radius is at least 1

        return radius_x, radius_y


    def try_place_particle(self, inner_radius: int, center_offset: List[int]) -> Optional[Tuple]:
        """Attempt to find a valid position for a new particle."""
        radius_x, radius_y = self.generate_particle_dimensions()
        num_candidates = 10 # Number of random positions to try
        candidates = []
        scores = []

        center_x = self.params.size // 2 + center_offset[0]
        center_y = self.params.size // 2 + center_offset[1]

        for _ in range(num_candidates):
            # Generate random position within the inner circle
            angle = uniform(0, 2 * np.pi)
            max_r = inner_radius - max(radius_x, radius_y) # Max distance from center
            if max_r <= 0: continue # Skip if particle is too large for the circle
            r = max_r * np.sqrt(uniform(0, 1)) # Uniform area sampling

            x = center_x + int(r * np.cos(angle))
            y = center_y + int(r * np.sin(angle))

            # Check for collisions and calculate attraction score
            nearby_particles = self.grid.get_nearby_particles(
                x, y, max(self.params.attraction_radius, max(radius_x, radius_y) + 10), self.params.size
            )

            score = self._calculate_score(x, y, nearby_particles)
            valid = self._check_validity(x, y, radius_x, radius_y, nearby_particles)

            if valid:
                candidates.append((x, y))
                scores.append(score)

        if not candidates:
            return None # No valid position found

        # Choose the best candidate based on scores
        chosen_idx = self._choose_position(scores)
        x, y = candidates[chosen_idx]
        angle = uniform(0, 360) # Random orientation

        return (x, y, radius_x, radius_y, angle)

    def _calculate_score(self, x: int, y: int, nearby_particles: List) -> float:
        """Calculate an attraction score based on nearby particles."""
        if not nearby_particles:
            return 1.0 # Base score if no neighbors

        distances = []
        for particle in nearby_particles:
            ex_x, ex_y = particle[:2]
            dist = np.sqrt((x - ex_x)**2 + (y - ex_y)**2)
            if dist < self.params.attraction_radius:
                distances.append(dist)

        if not distances:
            return 1.0 # Base score if no neighbors within attraction radius

        # Score higher if closer to more particles (promotes clustering)
        avg_dist = np.mean(distances)
        num_close = len(distances)
        density_factor = min(num_close / 5, 1.0) # Cap density influence
        proximity_factor = 1 - (avg_dist / self.params.attraction_radius)
        return 1.0 + (proximity_factor * density_factor * self.params.attraction_strength)

    def _check_validity(self, x: int, y: int, radius_x: int, radius_y: int,
                       nearby_particles: List) -> bool:
        """Check if the proposed particle position overlaps with existing particles."""
        check_radius = max(radius_x, radius_y)
        for particle in nearby_particles:
            ex_x, ex_y, ex_rx, ex_ry = particle
            dist = np.sqrt((x - ex_x)**2 + (y - ex_y)**2)
            # Check for overlap, allowing a small tolerance (-5 pixels)
            if dist < (check_radius + max(ex_rx, ex_ry) - 5):
                return False # Overlap detected
        return True # No overlap

    def _choose_position(self, scores: List[float]) -> int:
        """Choose a candidate position based on weighted random selection using scores."""
        total_score = sum(scores)
        if total_score == 0 or len(scores) == 0:
             # Fallback to random choice if no scores or all scores are zero
            return randint(0, len(scores) - 1) if scores else 0
        # Normalize scores to probabilities
        probs = [s / total_score for s in scores]
        # Choose index based on probability distribution
        return np.random.choice(len(scores), p=probs)

# --- Main Generator Class ---
class XCTSliceGenerator:
    """Generates the synthetic XCT slice and the multi-label segmentation image."""
    def __init__(self, params: ImageParameters):
        # Store the parameter dataclasses
        self.params = params
        
    def create_base_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """Initializes the greyscale image and the multi-label image."""
        # Greyscale image (uint16 for high dynamic range)
        img = np.full((self.params.size, self.params.size),
                     self.params.base_grey, dtype=np.uint16)

        # Multi-label image (uint8 for labels 0-3)
        # Start with everything outside the reconstruction volume (label 0)
        label_image = np.full((self.params.size, self.params.size),
                              LABEL_OUT_OF_RECONSTRUCTION, dtype=np.uint8)

        center = (self.params.size // 2, self.params.size // 2)

        # Draw outer circle (reconstruction volume boundary)
        # This area is initially the 'background' ring
        cv2.circle(img, center, self.params.outer_circle_size // 2,
                   self.params.outer_circle_grey, -1)
        cv2.circle(label_image, center, self.params.outer_circle_size // 2,
                   LABEL_BACKGROUND, -1)

        return img, label_image

    
    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate the synthetic slice and its corresponding label image."""
        img, label_image = self.create_base_images()

        # --- Draw Inner Circle (Electrolyte) ---
        # Introduce random offset for the inner circle center
        inner_circle_center_offset = [np.random.randint(-50, 50),
                                      np.random.randint(-50, 50)]
        inner_radius = self.params.inner_circle_size // 2
        center_inner = (self.params.size // 2 + inner_circle_center_offset[0],
                        self.params.size // 2 + inner_circle_center_offset[1])

        # Draw on greyscale image
        cv2.circle(img, center_inner, inner_radius,
                   self.params.inner_circle_grey, -1)
        # Draw on label image
        cv2.circle(label_image, center_inner, inner_radius,
                   LABEL_ELECTROLYTE, -1)

        # --- Place Particles (Cathode) ---
        # Pass the ImageParameters dataclass to the placer
        placer = ParticlePlacer(self.params)
        particles_placed = 0
        attempts = 0
        max_attempts = self.params.num_particles * 100 # Limit attempts to prevent infinite loops

        print("\nPlacing particles...")
        with tqdm(total=self.params.num_particles, desc="Particles", position=0, leave=True) as pbar:
            while particles_placed < self.params.num_particles and attempts < max_attempts:
                # Try to place a particle within the inner circle
                particle_data = placer.try_place_particle(inner_radius, inner_circle_center_offset)

                if particle_data:
                    x, y, rx, ry, angle = particle_data
                    # Assign random grey value within the specified range
                    grey = randint(*self.params.particle_grey_range)

                    # Draw particle on greyscale image
                    cv2.ellipse(img, (x, y), (rx, ry), angle, 0, 360, grey, -1)
                    # Draw particle on label image with cathode label
                    cv2.ellipse(label_image, (x, y), (rx, ry), angle, 0, 360, LABEL_CATHODE, -1)

                    # Add particle to the spatial grid for collision checking
                    placer.grid.add_particle(particles_placed, x, y, rx, ry)
                    particles_placed += 1
                    pbar.update(1)

                attempts += 1

        # Print placement statistics
        print(f"\nPlacement complete:")
        print(f"- Target particles: {self.params.num_particles}")
        print(f"- Particles placed: {particles_placed}")
        print(f"- Attempts made: {attempts}")
        if attempts > 0:
             print(f"- Success rate: {particles_placed / attempts * 100:.1f}%")
        else:
             print("- Success rate: N/A (no attempts made)")


        # Tomography effects removed; using base image only
        final_img_clipped = np.clip(img, 0, 65535).astype(np.uint16)

        label_image[label_image == LABEL_OUT_OF_RECONSTRUCTION] = 0

        return final_img_clipped, label_image


# --- Plotting Utility ---
def plot_results(synthetic_slice: np.ndarray, label_image: np.ndarray):
    """Plots the generated synthetic slice and its multi-label segmentation with a legend."""
    plt.style.use('default') # Use default matplotlib style
    fig, axes = plt.subplots(1, 2, figsize=(18, 9)) # Adjusted figure size

    # --- Plot Synthetic Slice ---
    ax1 = axes[0]
    im1 = ax1.imshow(synthetic_slice, cmap='gray', vmin=np.min(synthetic_slice), vmax=np.max(synthetic_slice))
    ax1.set_title('Synthetic XCT Slice (Greyscale)')
    ax1.axis('off')
    # Optional: Add colorbar for greyscale image if needed
    # fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Intensity (uint16)')


    # --- Plot Label Image ---
    ax2 = axes[1]
    # Use 'plasma' colormap with 4 distinct colors
    cmap = plt.get_cmap('plasma', 4)

    # Define the boundaries for the labels (0, 1, 2, 3)
    bounds = [0, 1, 2, 3, 4] # n_labels + 1 boundaries
    norm = BoundaryNorm(bounds, cmap.N)

    im2 = ax2.imshow(label_image, cmap=cmap, norm=norm, interpolation='nearest')
    ax2.set_title('Segmentation Label Image')
    ax2.axis('off')

    # --- Create Legend instead of Colorbar ---
    # Define labels corresponding to the integer values
    label_names = {
        LABEL_OUT_OF_RECONSTRUCTION: 'Out of Recon (0)',
        LABEL_BACKGROUND: 'Background (1)',
        LABEL_ELECTROLYTE: 'Electrolyte (2)',
        LABEL_CATHODE: 'Cathode (3)'
    }
    # Create patches as handles for the legend
    # We map the integer label value (0, 1, 2, 3) to the correct color from the colormap
    patches = [mpatches.Patch(color=cmap(label_value), label=name)
               for label_value, name in label_names.items()]

    # Add legend to the plot
    ax2.legend(handles=patches,
               loc='lower left',        # Position the legend
               bbox_to_anchor=(0.01, 0.01), # Fine-tune position (relative to axes)
               borderaxespad=0.,        # Padding around legend border
               fontsize='small',        # Adjust font size
               title="Phase Legend",    # Optional legend title
               title_fontsize='medium') # Optional title font size

    plt.tight_layout() # Adjust layout to prevent overlap
    plt.show()
def randomize_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new configuration dictionary with randomized values based on the base config.
    Only randomizes image parameters; removes all tomography/effects logic.
    """
    config = deepcopy(base_config) # Start with a copy of the base config

    # --- Randomize Image Parameters ---
    img_params = config['image']
    img_params['inner_circle_grey'] += randint(-3000, 3000)
    img_params['outer_circle_grey'] += randint(-3000, 3000)
    # Ensure particle grey range remains valid (min < max)
    particle_min_delta = randint(-1000, 1000)
    particle_max_delta = randint(-5000, 5000)
    new_min = img_params['particle_grey_range'][0] + particle_min_delta
    new_max = img_params['particle_grey_range'][1] + particle_max_delta
    img_params['particle_grey_range'] = (max(1, new_min), max(new_min + 1, new_max)) # Ensure min < max and > 0

    img_params['num_particles'] += randint(-500, 500)
    img_params['attraction_radius'] += randint(-10, 10)
    img_params['attraction_strength'] *= uniform(0.8, 1.2) # Multiplicative change
    img_params['cell_size'] += randint(-30, 30)

    # Ensure values stay within reasonable bounds
    img_params['inner_circle_grey'] = max(1, min(65535, img_params['inner_circle_grey']))
    img_params['outer_circle_grey'] = max(1, min(65535, img_params['outer_circle_grey']))
    img_params['num_particles'] = max(100, img_params['num_particles']) # Min number of particles
    img_params['attraction_radius'] = max(5, img_params['attraction_radius'])
    img_params['attraction_strength'] = max(0, img_params['attraction_strength'])
    img_params['cell_size'] = max(20, min(200, img_params['cell_size']))





    return config


# --- Function to generate a single slice (for parallel processing) ---
def generate_single_slice(args):
    i, config, data_dir, labels_dir = args
    
    # Get parameters for this slice (randomized or base)
    if config['generation']['randomize']:
        slice_config = randomize_config(config)
        # Removed print statements for parallel processing
    else:
        slice_config = deepcopy(config) # Use a copy of the base config
        # Removed print statements for parallel processing

    # --- Create Dataclass Instances from Config ---
    # This provides structure and type hints within the generator classes
    image_params = dataclass_from_dict(ImageParameters, slice_config['image'])
    
    # --- Generate Slice and Label Image ---
    generator = XCTSliceGenerator(image_params)
    synthetic_slice, label_image = generator.generate()

    # --- Crop Slice and Label Image ---
    crop_size = slice_config['generation']['crop_size']
    image_size = slice_config['image']['size'] # Get size from current config

    if crop_size > 0 and crop_size < image_size:
        start_x = (image_size - crop_size) // 2
        start_y = (image_size - crop_size) // 2
        end_x = start_x + crop_size
        end_y = start_y + crop_size
        synthetic_slice_cropped = synthetic_slice[start_y:end_y, start_x:end_x]
        label_image_cropped = label_image[start_y:end_y, start_x:end_x]
    elif crop_size > 0:
        synthetic_slice_cropped = synthetic_slice
        label_image_cropped = label_image
    else:
        synthetic_slice_cropped = synthetic_slice
        label_image_cropped = label_image

    # --- Save Results ---
    slice_filename = data_dir / f"synthetic_slice_{i:03d}.tif"
    label_filename = labels_dir / f"label_image_{i:03d}.tif" # Use TIF for labels too

    imwrite(slice_filename, synthetic_slice_cropped, imagej=True, metadata={'axes': 'YX'})
    # Ensure label image is saved as uint8
    imwrite(label_filename, label_image_cropped.astype(np.uint8), imagej=True, metadata={'axes': 'YX'})
    
    return i

# --- Main Execution Block ---
if __name__ == "__main__":
    print("Generating synthetic XCT slices with multi-label masks in parallel...")

    # --- Central Configuration Dictionary ---
    # All parameters are defined here for easy access and modification.
    config = {
        "generation": {
            "output_dir": Path("./synthetic_data_refactored"),
            "num_slices": 50,
            "crop_size": 2560, # Final output size after cropping center
            "randomize": True, # Set to False to use exact base parameters
            "plot_single_slice": True, # Disable plotting in parallel mode
            "num_workers": 12
        },
        "image": {
            "size": 2560,
            "outer_circle_size": 2420,
            "inner_circle_size": 1860,
            "base_grey": 29000,
            "inner_circle_grey": 40000,
            "outer_circle_grey": 50000,
            "particle_grey_range": (2260, 20920),
            "num_particles": 3700,
            "attraction_radius": 30,
            "attraction_strength": 0.1e5,
            "cell_size": 100 # For spatial grid efficiency
        }
    }

    # --- Setup Output Directory ---
    output_dir = config['generation']['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data and labels subfolders
    data_dir = output_dir / "data"
    labels_dir = output_dir / "labels"
    data_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    num_slices_to_generate = config['generation']['num_slices']
    num_workers = config['generation']['num_workers']
    
    print(f"Using {num_workers} parallel workers for generation")
    
    # Special case for single slice with plotting
    if num_slices_to_generate == 1 and config['generation']['plot_single_slice']:
        print("\n--- Generating a single slice with plotting ---")
        # Use the same code as before for a single slice with plotting
        slice_config = randomize_config(config) if config['generation']['randomize'] else deepcopy(config)
        print("\nUsing Parameters:")
        pprint.pprint(slice_config, indent=2)
        
        image_params = dataclass_from_dict(ImageParameters, slice_config['image'])
                
        generator = XCTSliceGenerator(image_params)
        synthetic_slice, label_image = generator.generate()
        
        crop_size = slice_config['generation']['crop_size']
        image_size = slice_config['image']['size']
        
        if crop_size > 0 and crop_size < image_size:
            print(f"\nCropping images to center {crop_size}x{crop_size} pixels...")
            start_x = (image_size - crop_size) // 2
            start_y = (image_size - crop_size) // 2
            end_x = start_x + crop_size
            end_y = start_y + crop_size
            synthetic_slice_cropped = synthetic_slice[start_y:end_y, start_x:end_x]
            label_image_cropped = label_image[start_y:end_y, start_x:end_x]
        else:
            synthetic_slice_cropped = synthetic_slice
            label_image_cropped = label_image
            
        print("\nPlotting results...")
        plot_results(synthetic_slice_cropped, label_image_cropped)
        
        slice_filename = data_dir / "synthetic_slice_000.tif"
        label_filename = labels_dir / "label_image_000.tif"
        
        print(f"\nSaving greyscale slice to: {slice_filename}")
        imwrite(slice_filename, synthetic_slice_cropped, imagej=True, metadata={'axes': 'YX'})
        
        print(f"Saving label image to: {label_filename}")
        imwrite(label_filename, label_image_cropped.astype(np.uint8), imagej=True, metadata={'axes': 'YX'})
    else:
        # --- Parallel Generation ---
        print(f"\n--- Generating {num_slices_to_generate} slices in parallel ---")
        start_time = time.time()
        
        # Prepare arguments for parallel processing
        args_list = [(i, deepcopy(config), data_dir, labels_dir) for i in range(num_slices_to_generate)]
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Map the generate_single_slice function to all arguments and track progress with tqdm
            results = list(tqdm(executor.map(generate_single_slice, args_list), 
                               total=num_slices_to_generate, 
                               desc="Generating slices"))
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        slices_per_second = num_slices_to_generate / elapsed_time
        
        print(f"\nFinished generating {num_slices_to_generate} slices in {elapsed_time:.2f} seconds")
        print(f"Average generation speed: {slices_per_second:.2f} slices/second")
        print(f"Data saved in: '{data_dir}'")
        print(f"Labels saved in: '{labels_dir}'")
