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
    # You could also add a new enum for SLIPPERY if you like.

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
        self.ramp_height = 1        # Total height to climb
        self.ramp_slope = 1/8       # Standard wheelchair ramp slope (1:12 ratio)
        
        # Calculate required length for ramp based on height and slope
        self.ramp_length_steps = int(np.ceil((self.ramp_height / self.ramp_slope) / self.step_length))
        
        # Stairs parameters (for going down)
        self.stair_height = 0.15    # Height of each stair
        self.stair_depth = 0.35     # Depth of each stair
        self.num_stairs = int(self.ramp_height / self.stair_height)  # Number of stairs needed
        
        # --------------------------------------------------------------------
        # Instead of 5 steps of flat ground after the ramp (or after the beam)
        # make it 8 or 10. Here we choose 10 for demonstration.
        self.flat_after_ramp_or_beam = 8
        # --------------------------------------------------------------------

        # Environment layout
        #
        # slope_sections is a list of tuples (start_step, end_step, total_height).
        # 1) Up ramp: from step 5 to step 5 + ramp_length_steps.
        # 2) Downstairs: from step (5 + ramp_length_steps + 8) to
        #    that plus num_stairs. 
        #
        # We'll keep the second argument as +8 for the beam region, but you can
        # also increase it if you want more steps for the beam. 
        # The only big change is how we handle the final flat region after the stairs.
        #
        # In the second slope section, we do "-self.ramp_height - 8" so that
        # we come back down from the ramp (plus the beam height).
        
        self.slope_sections = [
            (5, 5 + self.ramp_length_steps, self.ramp_height), 
            (5 + self.ramp_length_steps + 8,
             5 + self.ramp_length_steps + 8 + self.num_stairs,
             -self.ramp_height - 8)
        ]
        
        self.beam_sections = [
            (5 + self.ramp_length_steps, 5 + self.ramp_length_steps + 8)
        ]
        
        self.hazard_zones = []
        self.gravel_sections = []
        
        # Number of random obstacles after the final wet ground
        self.num_obstacles = 10
        self.obstacle_types = [
            # (height, size_x, size_y, RGBA, name)
            (0.03, 0.12, 0.10, np.array([0.5, 0.35, 0.15, 1]), "wooden_plank"),  # Dark brown
            (0.06, 0.13, 0.17, np.array([0.7, 0.3, 0.2, 1]),  "broken_brick"),   # Rusty red
            (0.05, 0.15, 0.15, np.array([0.6, 0.6, 0.6, 1]),  "concrete_piece"), # Medium gray
            (0.07, 0.10, 0.20, np.array([0.4, 0.4, 0.4, 1]),  "stone_block"),    # Slate gray
            (0.04, 0.14, 0.16, np.array([0.3, 0.2, 0.1, 1]),  "old_timber")      # Dark, desaturated brown
        ]
        
        np.random.seed(42)
        
        # --------------------------------------------------------------------
        # WET/SLIPPERY GROUND SECTION (after the last slope/stairs, before obstacles)
        # We replace the old "5 steps of flat ground" with 10 steps of
        # wet, slippery ground.
        # --------------------------------------------------------------------
        self.wet_slippery_steps = 10
        
        # The end of the second slope_section is the last stairs index:
        self.end_of_stairs = self.slope_sections[-1][1]
        
        # So the wet ground will start at 'end_of_stairs' and last for 10 steps
        self.wet_slippery_section = (
            self.end_of_stairs,
            self.end_of_stairs + self.wet_slippery_steps
        )
        
        # Now the obstacles will start AFTER this wet/slippery section
        # Instead of the old +5 offset, we use + self.wet_slippery_steps.
        obstacle_start = self.wet_slippery_section[1]
        
        # Generate obstacle positions
        base_positions = np.arange(obstacle_start, obstacle_start + self.num_obstacles)
        positions = base_positions + np.random.uniform(-0.2, 0.2, self.num_obstacles)
        
        self.obstacle_sections = []
        for pos in positions:
            obstacle_type = self.obstacle_types[np.random.randint(0, len(self.obstacle_types))]
            self.obstacle_sections.append((pos, obstacle_type))
        
        # Additional plain ground AFTER obstacles (unchanged)
        self.plain_ground_steps = 10
        self.plain_ground_start = obstacle_start + self.num_obstacles

        # --------------------------------------------------------------------
        # Example "plank" or "trench" positions in the wet/slippery section.
        # You can randomize or define them as you like.
        # Here, we place 2 planks and 1 trench in that 10-step region.
        # --------------------------------------------------------------------
        self.plank_positions = [
            self.end_of_stairs + 2,  # around step #2 in slippery region
        ]
        self.trench_positions = [
            self.end_of_stairs + 8   # around step #8 in slippery region
        ]
        # --------------------------------------------------------------------

    def calculate_curved_position(self, x, base_y):
        """Calculate y-position for curved path."""
        curve_y = self.curve_amplitude * np.sin(
            2 * np.pi * x / (self.curve_period * self.step_length)
        )
        return base_y + curve_y

    def calculate_slope_height(self, current_idx, slope_start, slope_end, total_height):
        """Calculate height for smooth slopes and for stairs."""
        if total_height > 0:  # Going up (ramp)
            progress = (current_idx - slope_start) / (slope_end - slope_start)
            return total_height * progress
        else:  # Going down (stairs)
            if current_idx < slope_start:
                return self.ramp_height  # Maintain ramp height until stairs start
            step_position = current_idx - slope_start
            total_steps = slope_end - slope_start
            if step_position >= total_steps:
                return 0.0  # back to ground level
            step_height = self.ramp_height / total_steps
            return self.ramp_height - (step_position * step_height)

    def generate_construction_sequence(self, construction_type=ConstructionTypes.FLAT):
        """Generate a realistic construction environment sequence."""
        sequence = []
        x = 0
        y = self.step_width
        
        # --------------------------------------------------------------------
        # Total steps now includes:
        #   5 initial flat steps + up ramp + beam + down stairs +
        #   wet/slippery section (10 steps) + obstacles (self.num_obstacles) +
        #   final plain ground steps (10).
        # --------------------------------------------------------------------
        total_steps = (
            5 +                         # initial flat ground
            self.ramp_length_steps +    # up ramp
            8 +                         # beam section
            self.num_stairs +           # down stairs
            self.wet_slippery_steps +   # new wet/slippery ground
            self.num_obstacles +        # obstacles region
            self.plain_ground_steps     # final plain ground
        )
        
        for i in range(total_steps):
            # Base position with curve for a winding path
            curved_y = self.calculate_curved_position(x, y)
            
            # Calculate yaw angle based on curve
            dy_dx = (self.curve_amplitude *
                     (2 * np.pi / (self.curve_period * self.step_length)) *
                     np.cos(2 * np.pi * x / (self.curve_period * self.step_length)))
            yaw_angle = np.arctan2(dy_dx, 1)
            
            # Initialize height
            height = 0
            
            # Add ramps and stairs
            for start, end, slope_height in self.slope_sections:
                if start <= i <= end:
                    height += self.calculate_slope_height(i, start, end, slope_height)
                elif i > end and slope_height > 0:
                    # maintain ramp height in the flat region on top if needed
                    height = slope_height
            
            # Correct height for the second slope (stairs).
            if self.slope_sections[-1][0] <= i <= self.slope_sections[-1][1]:
                height = self.calculate_slope_height(
                    i,
                    self.slope_sections[-1][0],
                    self.slope_sections[-1][1],
                    -self.ramp_height
                )
            
            # Everything after the last stairs is ground level (height=0)
            if i > self.slope_sections[-1][1]:
                height = 0
            
            # Create step
            step = np.array([x, curved_y, height, yaw_angle])
            sequence.append(step)
            
            x += self.step_length
            y *= -1  # alternate sides for a walking pattern
        
        # Add stopping sequence at the end
        stopping_steps = 5
        for i in range(stopping_steps):
            step = np.array([x, curved_y, 0, 0])  # no height or yaw change
            sequence.append(step)
            x += self.step_length * (1 - (i / stopping_steps))  # gradually reduce step size
        
        return np.array(sequence)

    def adjust_terrain_visualization(self, client, boxes, sequence):
        """Enhanced terrain visualization, including wet/slippery ground, planks, and trenches."""
        sequence_extended = [np.array([0, 0, -1, 0]) for _ in range(len(boxes))]
        sequence_extended[:len(sequence)] = sequence
        
        # Calculate the transition point for ramp colors (60% mark)
        ramp_start = self.slope_sections[0][0]
        ramp_end   = self.slope_sections[0][1]
        ramp_length = ramp_end - ramp_start
        transition_point = ramp_start + int(ramp_length * 0.6)  # 60% mark
        
        for idx, (box, step) in enumerate(zip(boxes, sequence_extended)):
            box_h = client.model.geom(box).size[2]
            client.model.body(box).pos[:] = step[0:3] - np.array([0, 0, box_h])
            client.model.body(box).quat[:] = tf3.euler.euler2quat(0, 0, step[3])
            
            # Determine if we're on the up ramp, down stairs, or beam
            is_up_ramp    = (self.slope_sections[0][0] <= idx <= self.slope_sections[0][1])
            is_down_stairs = (self.slope_sections[1][0] <= idx <= self.slope_sections[1][1])
            is_beam       = any(start <= idx <= end for start, end in self.beam_sections)
            
            # Check if the current position is an obstacle
            obstacle_info = None
            for pos, obs_type in self.obstacle_sections:
                # If idx is close to the obstacle position
                if abs(idx - pos) < 0.5:
                    obstacle_info = obs_type
                    break
            
            # 1) UP RAMP SECTION
            if is_up_ramp:
                if idx <= transition_point:
                    # Darker color near bottom
                    client.model.geom(box).rgba[:] = np.array([0.2, 0.2, 0.2, 1])
                    client.model.geom(box).size[:] = np.array([0.2, 2.8, box_h])
                else:
                    # Lighter color near top
                    client.model.geom(box).rgba[:] = np.array([0.5, 0.45, 0.4, 1])
                    client.model.geom(box).size[:] = np.array([0.08, 1.4, box_h])
            
            # 2) DOWN STAIRS SECTION
            elif is_down_stairs:
                client.model.geom(box).rgba[:] = np.array([0.7, 0.7, 0.7, 1])
                client.model.geom(box).size[:] = np.array([0.2, 0.5, 0.04])
            
            # 3) BEAM SECTION
            elif is_beam:
                client.model.geom(box).rgba[:] = np.array([0.6, 0.4, 0.2, 1])
                client.model.geom(box).size[:] = np.array([0.6, 0.1, box_h])
            
            # 4) OBSTACLES
            elif obstacle_info is not None:
                height, size_x, size_y, color, _ = obstacle_info
                client.model.geom(box).size[:] = np.array([size_x, size_y, height / 2])  
                client.model.geom(box).rgba[:] = color
                # Lift it so the obstacle sits on the ground:
                client.model.body(box).pos[2] = height / 2
                # Give it some random yaw
                random_yaw = np.random.uniform(0, 2 * np.pi)
                client.model.body(box).quat[:] = tf3.euler.euler2quat(0, 0, random_yaw)
            
            # 5) WET/SLIPPERY GROUND SECTION (the newly-added region)
            elif self.wet_slippery_section[0] <= idx < self.wet_slippery_section[1]:
                # Make it look 'wet' or different
                client.model.geom(box).rgba[:] = np.array([0.4, 0.4, 0.9, 0.8]) 
                
                # Decrease friction to simulate slipperiness (MuJoCo friction param)
                # friction = [sliding_friction, torsional_friction, rolling_friction]
                client.model.geom(box).friction[:] = [0.1, 0.005, 0.0001]
                
                # Default size for normal ground
                client.model.geom(box).size[:] = np.array([0.2, 0.2, box_h])
                
                # Check if this idx has a plank or a trench
                # (In reality, you'd want to handle shape collisions carefully.)
                if any(abs(idx - p) < 0.5 for p in self.plank_positions):
                    # Make a wide/narrow "plank" geometry above the wet floor
                    client.model.geom(box).size[:] = np.array([0.015, 0.2, box_h+0.025 ])
                    client.model.geom(box).rgba[:] = np.array([0.5, 0.25, 0.1, 1])  # Wooden color
                    # Raise the plank slightly
                    client.model.body(box).pos[2] += 0.02 
                
                elif any(abs(idx - t) < 0.5 for t in self.trench_positions):
                    # Make a "trench" by lowering the geometry
                    # We'll cheat by simply changing color and reducing height.
                    # A real trench might be negative geometry or an actual hole.
                    client.model.geom(box).rgba[:] = np.array([0.1, 0.1, 0.1, 1])
                    client.model.geom(box).size[:] = np.array([0.015, 0.2,box_h+0.025])
                    client.model.body(box).pos[2] -= 0.015
            
            # 6) REGULAR GROUND (all other places)
            else:
                client.model.geom(box).rgba[:] = np.array([0.7, 0.9, 0.9, 0.5])
                client.model.geom(box).size[:] = np.array([0.2, 0.2, box_h])
