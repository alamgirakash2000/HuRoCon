import numpy as np
import transforms3d as tf3
from enum import Enum, auto

class ConstructionTypes(Enum):
    FLAT = auto()         # Regular ground
    SLOPE = auto()        # Gentle slopes
    UNEVEN = auto()       # Uneven terrain with small bumps
    BEAM = auto()         # Construction beams/planks
    HAZARD = auto()       # Hazard zones (to avoid)
    GRAVEL = auto()       # Gravel/debris areas

class ConstructionEnvironment:
    def __init__(self):
        # Basic parameters
        self.step_length = 0.35
        self.step_width = 0.15
        
        # Path curvature parameters
        self.curve_amplitude = 0.3  # Amplitude of the curve
        self.curve_period = 30      # Period of the curve
        
        # Terrain parameters
        self.max_slope_height = 0.15     
        self.uneven_max_height = 0.08    
        self.beam_width = 0.45           
        self.gravel_variation = 0.03     
        
        # Wheelchair ramp parameters (for going up)
        self.ramp_height = 1  # Total height to climb
        self.ramp_slope = 1/12   # Standard wheelchair ramp slope (1:12 ratio)
        
        # Calculate required length for ramp based on height and slope
        self.ramp_length_steps = int(np.ceil((self.ramp_height / self.ramp_slope) / self.step_length))
        
        # Stairs parameters (for going down)
        self.stair_height = 0.15  # Height of each stair
        self.stair_depth = 0.35   # Depth of each stair
        self.num_stairs = int(self.ramp_height / self.stair_height)  # Number of stairs needed
        
        # New variable to control the beam's length in steps
        self.beam_steps = 15  # Change this to 20 (or any other value) to adjust the beam length
        
        # Environment layout
        self.slope_sections = [
            (5, 5 + self.ramp_length_steps, self.ramp_height),  # Up ramp
            (5 + self.ramp_length_steps + self.beam_steps, 
             5 + self.ramp_length_steps + self.beam_steps + self.num_stairs, 
             -self.ramp_height - self.beam_steps)  # Down stairs immediately after beam
        ]
        
        self.beam_sections = [
            (5 + self.ramp_length_steps, 5 + self.ramp_length_steps + self.beam_steps)
        ]
        self.hazard_zones = []
        self.gravel_sections = []
        
        # Obstacle parameters (unchanged)
        self.num_obstacles = 10
        self.obstacle_types = [
            (0.03, 0.12, 0.10, np.array([0.5, 0.35, 0.15, 1]), "wooden_plank"),
            (0.06, 0.13, 0.17, np.array([0.7, 0.3, 0.2, 1]), "broken_brick"),
            (0.05, 0.15, 0.15, np.array([0.6, 0.6, 0.6, 1]), "concrete_piece"),
            (0.07, 0.10, 0.20, np.array([0.4, 0.4, 0.4, 1]), "stone_block"),
            (0.04, 0.14, 0.16, np.array([0.3, 0.2, 0.1, 1]), "old_timber")
        ]
        
        self.obstacle_sections = []
        np.random.seed(42)
        obstacle_start = self.slope_sections[-1][1] + 5  # 5 steps of flat ground after stairs
        base_positions = np.arange(obstacle_start, obstacle_start + self.num_obstacles)
        positions = base_positions + np.random.uniform(-0.2, 0.2, self.num_obstacles)
        
        for pos in positions:
            obstacle_type = self.obstacle_types[np.random.randint(0, len(self.obstacle_types))]
            self.obstacle_sections.append((pos, obstacle_type))

        self.plain_ground_steps = 10  # Number of plain ground steps after obstacles
        self.plain_ground_start = obstacle_start + self.num_obstacles


    def calculate_curved_position(self, x, base_y):
        """Calculate y-position for curved path."""
        curve_y = self.curve_amplitude * np.sin(2 * np.pi * x / (self.curve_period * self.step_length))
        return base_y + curve_y

    def calculate_slope_height(self, current_idx, slope_start, slope_end, total_height):
        """Calculate height for smooth slopes and stairs."""
        if total_height > 0:  # Going up (ramp)
            progress = (current_idx - slope_start) / (slope_end - slope_start)
            return total_height * progress
        else:  # Going down (stairs)
            if current_idx < slope_start:
                return self.ramp_height  # Keep at beam height before stairs
            
            step_position = current_idx - slope_start
            total_steps = slope_end - slope_start
            
            if step_position >= total_steps:
                return 0.0  # Ground level
                
            step_height = self.ramp_height / total_steps
            return self.ramp_height - (step_position * step_height)

    def generate_construction_sequence(self, construction_type=ConstructionTypes.FLAT):
        """Generate a realistic construction environment sequence."""
        sequence = []
        x = 0
        y = self.step_width
        
        # Total steps: Initial flat, ramp, beam, stairs, final flat, obstacles, plain ground
        total_steps = (5 + self.ramp_length_steps + self.beam_steps + self.num_stairs + 5 + 
                   self.num_obstacles + self.plain_ground_steps)
        
        for i in range(total_steps):
            # Base position with curve for a winding path
            curved_y = self.calculate_curved_position(x, y)
            
            # Calculate yaw angle based on curve
            dy_dx = self.curve_amplitude * (2 * np.pi / (self.curve_period * self.step_length)) * \
                    np.cos(2 * np.pi * x / (self.curve_period * self.step_length))
            yaw_angle = np.arctan2(dy_dx, 1)
            
            # Initialize height
            height = 0
            
            # Add ramps and stairs
            for start, end, slope_height in self.slope_sections:
                if start <= i <= end:
                    height += self.calculate_slope_height(i, start, end, slope_height)
                elif i > end and slope_height > 0:  # Maintain height for flat section on top
                    height = slope_height
            
            # Correct height for stairs starting from beam height
            if self.slope_sections[-1][0] <= i <= self.slope_sections[-1][1]:
                height = self.calculate_slope_height(i, self.slope_sections[-1][0],
                                                    self.slope_sections[-1][1],
                                                    -self.ramp_height)
            
            # Ensure final flat section and obstacles are at ground level
            if i > self.slope_sections[-1][1]:
                height = 0
            
            # Create step
            step = np.array([x, curved_y, height, yaw_angle])
            sequence.append(step)
            
            x += self.step_length
            y *= -1  # Alternate sides for natural walking pattern
        
        # Add stopping sequence at the end
        stopping_steps = 5  # Number of steps to stop smoothly
        for i in range(stopping_steps):
            step = np.array([x, curved_y, 0, 0])  # No height or yaw change
            sequence.append(step)
            x += self.step_length * (1 - (i / stopping_steps))  # Gradually reduce step length
        
        return np.array(sequence)

    def adjust_terrain_visualization(self, client, boxes, sequence):
        """Enhanced terrain visualization with different ramp colors."""
        sequence_extended = [np.array([0, 0, -1, 0]) for _ in range(len(boxes))]
        sequence_extended[:len(sequence)] = sequence
        
        # Calculate the transition point for ramp colors (60% mark)
        ramp_start = self.slope_sections[0][0]
        ramp_end = self.slope_sections[0][1]
        ramp_length = ramp_end - ramp_start
        transition_point = ramp_start + int(ramp_length * 0.6)  # 60% mark
        
        for idx, (box, step) in enumerate(zip(boxes, sequence_extended)):
            box_h = client.model.geom(box).size[2]
            client.model.body(box).pos[:] = step[0:3] - np.array([0, 0, box_h])
            client.model.body(box).quat[:] = tf3.euler.euler2quat(0, 0, step[3])
            
            # Determine terrain type
            is_up_ramp = (self.slope_sections[0][0] <= idx <= self.slope_sections[0][1])
            is_down_stairs = (self.slope_sections[1][0] <= idx <= self.slope_sections[1][1])
            is_beam = any(start <= idx <= end for start, end in self.beam_sections)
            
            # Check if current position is an obstacle
            obstacle_info = None
            for pos, obs_type in self.obstacle_sections:
                if abs(idx - pos) < 0.5:
                    obstacle_info = obs_type
                    break
            
            if is_up_ramp:
                if idx <= transition_point:
                    client.model.geom(box).rgba[:] = np.array([0.2, 0.2, 0.2, 1])
                    client.model.geom(box).size[:] = np.array([0.2, 2.8, box_h])
                else:
                    client.model.geom(box).rgba[:] = np.array([0.5, 0.45, 0.4, 1])
                    client.model.geom(box).size[:] = np.array([0.07, 1.4, box_h])
            elif is_down_stairs:
                client.model.geom(box).size[:] = np.array([0.2, 0.5, 0.04])
                client.model.geom(box).rgba[:] = np.array([0.7, 0.7, 0.7, 1])
            elif is_beam:
                client.model.geom(box).size[:] = np.array([0.6, 0.05, box_h])
                client.model.geom(box).rgba[:] = np.array([0.6, 0.4, 0.2, 1])
            elif obstacle_info is not None:
                # Apply obstacle properties
                height, size_x, size_y, color, _ = obstacle_info
                client.model.geom(box).size[:] = np.array([size_x, size_y, height / 2])  # Half-height for MuJoCo
                client.model.geom(box).rgba[:] = color
                
                # Adjust position so the bottom of the obstacle is at ground level
                client.model.body(box).pos[2] = height / 2  # Center of the obstacle is at half-height
                
                # Random rotation for natural appearance
                random_yaw = np.random.uniform(0, 2 * np.pi)
                client.model.body(box).quat[:] = tf3.euler.euler2quat(0, 0, random_yaw)
            else:
                # Regular ground
                client.model.geom(box).size[:] = np.array([0.2, 0.2, box_h])
                client.model.geom(box).rgba[:] = np.array([0.7, 0.9, 0.9, 0.5])