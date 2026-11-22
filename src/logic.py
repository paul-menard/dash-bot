### the python script that handles the logics of the different modes that the character can have
### we will use an OOP approach to code the logics of the different characters
### then depending on the logic of each character change how they should behave based on what is seen on the capture
### logic.py reacts to what is seen and acts on it through controls.py that is called in each class
### this program does not aim to do any detection or stuff like this

from controls import click
from controls import long_click
from controls import stationary_click
from controls import open_key, close_key
import time
import torch
import cv2 as cv 
import numpy as np


"""
from ultralytics import YOLO
model = YOLO('runs/detect/train/weights/best.pt')
print(model.names)
exit()
"""


def character_cls():
    ## 2 = Cube
    ## 5 = Rocket Cube
    return [2, 5]

def obstacles_cls():
    return torch.tensor([0, 4, 7], device="cuda:0")

def convert_to_cls(cls, w, h):
    ## will return a class so that we can instanciate 
    if cls == 2:
        return Cube(w, h)
    if cls == 5:
        return Rocket(w, h)

### base class with base methods that we will find in each sub-class
class Character():

    def __init__(self, w, h):
        self.x = 0
        self.y = 0
        self.speed = torch.zeros(2)
        self.cls_id = None
        self.rectangle = None
        self.screen_w = w
        self.screen_h = h

    def position(self, rectangle):
        ## for the following methods we use the format rectangle = (x1, y1, x2, y2)
        ## for the position of the character, the right will be used for distances and for obstacles the left one
        return (rectangle[2], rectangle[3]) ## = (x2, y2) so the bottom right corner
    


    ## convention, distance positive = devant | au dessus et sinon en dessous | derrière

    def distance_x_left(self, rectangle):
        if rectangle.dim() == 1:
            return rectangle[0] - self.x
        return rectangle[:, 0] - self.x
    
    def distance_x_right(self, rectangle):
        if rectangle.dim() == 1:
            return rectangle[2] - self.x 
        return rectangle[: , 2] - self.x

    def distance_y_top(self, rectangle):
        if rectangle.dim() == 1:
            return rectangle[1] - self.y
        return rectangle[:, 1] - self.y 

    def distance_y_bottom(self, rectangle):
        if rectangle.dim() == 1:
            return rectangle[3] - self.y
        return rectangle[:, 3] - self.y

    def jump(self):
        click()
    pass


class Cube(Character):
    def __init__(self, w, h):
        super().__init__(w, h)
        self.cls_id = 2
        self.threshold_x = 60
        self.threshold_y = 20
        self.max_speed = None


    def check_type(self, results):
        if 2. in results[0].boxes.cls:
            return True
        return False
    
    def is_jump_safe(self, cube_rect, obstacles_rects):
        ground_clearance = 60 
        
        danger_zone = torch.tensor([
            cube_rect[0], 
            cube_rect[1] - 120,       
            cube_rect[0] + 200,         
            cube_rect[3] - ground_clearance
        ])
        
        inter_x1 = torch.maximum(danger_zone[0], obstacles_rects[:, 0])
        inter_y1 = torch.maximum(danger_zone[1], obstacles_rects[:, 1])
        inter_x2 = torch.minimum(danger_zone[2], obstacles_rects[:, 2])
        inter_y2 = torch.minimum(danger_zone[3], obstacles_rects[:, 3])
        
        # Overlap exists if width and height of intersection are positive
        overlap = (inter_x1 < inter_x2) & (inter_y1 < inter_y2)
        
        return not overlap.any()
    
    """
    def behave(self, rectangles, cls_list):
        for i in range(len(cls_list)):
            if cls_list[i] == 7 :
                if self.distance_x_left(rectangles[i]) < self.threshold_x and self.distance_x_left(rectangles[i]) > 0:
                    self.jump()
                    print("Cube jump")
                    ## it works so far !! the methods works, but we lose a lot of fps when nearing the obstacles
                    ## we need to find a more efficient way of coding all of this.
            if cls_list[i] == 8:
                #print("edge detected | distance :", self.distance_y_top(rectangles[i])-15)
                if self.distance_y_bottom(rectangles[i]) > 0:
                    if self.distance_x_left(rectangles[i]) < self.threshold_x and self.distance_x_left(rectangles[i]) > 0:
                        self.jump()
                        print("Cube jump")
                if self.distance_y_top(rectangles[i]) < 0: ## small margin just in case ?
                    if self.distance_x_right(rectangles[i]) < self.threshold_x and self.distance_x_right(rectangles[i]) > 0:
                        self.jump()
                        print("Cube jump")
            if cls_list[i] == 0:
                if self.distance_x_right(rectangles[i]) < 2*self.threshold_x and self.distance_x_right(rectangles[i]) > 0:
                    self.jump()
                    print("Cube jump", self.distance_x_right(rectangles[i]))
    """
    def behave(self, rectangles, cls):
        ## behaviour for the spikes (simple version)
        left_distances = self.distance_x_left(rectangles)
        right_distances = self.distance_x_right(rectangles)
        bottom_distances = self.distance_y_bottom(rectangles)
        jump_safe = self.is_jump_safe(self.rectangle, rectangles[cls == 7] )

        if jump_safe:
            #spikes = (cls == 7) & (left_distances > 0) & (left_distances < self.threshold_x) & (bottom_distances <= 0)
            spikes = (cls == 7) & (left_distances > 0) & (left_distances < self.threshold_x) & (bottom_distances < 10)
            if spikes.any():
                self.jump()
                #print("Cube jump on spike")

            ## behaviour for the blocks
            #blocks = (cls == 0) & (left_distances > 0) & (left_distances < 0.5*self.threshold_x) & (bottom_distances <= 0) & (bottom_distances < self.threshold_y)
            blocks = (cls == 0) & (right_distances > 0) & (right_distances < 1.75*self.threshold_x) & (bottom_distances < 50) & (bottom_distances > 0)
            if blocks.any():
                self.jump()
                #print("Cube jump on block", self.y)

            ## behaviour for edges
            #edges = (cls == 8) & (left_distances > 0) & (left_distances < self.threshold_x) & (bottom_distances > 0)
            edges_up = (cls == 8) & (left_distances > 0) & (left_distances < 1.5*self.threshold_x) & (bottom_distances < 0)
            if edges_up.any():
                #print("edge up detected")
                self.jump()
                #print("Cube jump on block")

            edges_down = (cls == 8) & (right_distances > 0) & (right_distances < 0.5*self.threshold_x) & (bottom_distances > 0)
            if edges_down.any():
                #print("edge down detected")
                self.jump()
                #print("Cube jump on block")


    pass



class Rocket(Character):
    def __init__(self, w, h):
        super().__init__(w, h)
        self.cls_id = 5
        self.last_t = 0.0
        self.compute_window = 250 # px
        self.cuda0 = torch.device('cuda:0')
        self.barycenter = torch.tensor([0,0], device=self.cuda0)
        self.delta = 20
        self.alpha = 0
        self.thrust = - 2291.16/3 ##px/s^2 for my specific resolution
        self.gravity = 894.11/3 ##px/s^2
        
        self.max_speed = None
        self.speed = [0.0, 0.0] # Ensure speed is initialized
        
        # Asymmetric damping parameters
        self.K_up = 0.35   # Stronger damping when ascending to fight high thrust
        self.K_down = 0.15 # Lighter damping when falling
        
        # Target smoothing parameter
        self.ema_target = None
        self.beta = 0.85 # 0.0 = no smoothing, 0.99 = extreme smoothing
        
        self.dt = 0
        self.last_y_pos = self.barycenter[1] 
        self.gas_open = False
        self.dtype = torch.int64
        self.penalty_buffer = torch.zeros(self.screen_h + 1, device=self.cuda0, dtype=self.dtype)
        self.raw_map  = torch.abs( torch.arange(0, self.screen_h, device=self.cuda0, dtype=self.dtype)
                            - (self.screen_h//2) * torch.ones(self.screen_h, device=self.cuda0, dtype=self.dtype))//3
        
        self.latest_cost_map_cpu = None
    
    def check_type(self, results):
        if 5. in results[0].boxes.cls:
            return True
        return False
    
    def gas(self, alpha):
        long_click(alpha)

    def stationary_gas(self):
        stationary_click()

    def open_gas(self):
        open_key()
        self.gas_open = True

    def close_gas(self):
        close_key()
        self.gas_open = False
    

    def get_barycenter(self, rectangle):
        x1, y1, x2, y2 = rectangle
        return torch.tensor([(x1 + x2) / 2, (y1 + y2) / 2], device=self.cuda0)
    
    def distance_to_barycenter(self, target):
        ## we want to see if the barycenter gets away from the navigation at 
        ## a given x (which is the one of the barycenter)
        y = self.barycenter[1]
        return target - y
    ## faudrait ptet check pour qu'on ait bien un ptit tenseur sah
    
    def get_obstacles(self, rectangles, cls):
        left_distances = self.distance_x_left(rectangles)
        right_distances = self.distance_x_right(rectangles)
        
        is_obstacle = torch.isin(cls, obstacles_cls())
        
        # 1. right_distances > 0 ensures the right edge is still ahead of us (not completely behind).
        # 2. left_distances < self.compute_window ensures the obstacle starts within our vision.
        # This naturally keeps long obstacles that we are currently inside (left < 0 but right > 0).
        valid_position = (right_distances > 0) & (left_distances < self.compute_window)
        
        obstacles = is_obstacle & valid_position
        
        return rectangles[obstacles]


    def navigation(self, obstacles):
        ## this method generates the path to follow for the rocket
        delta = 15 ## px
        nav_padd = torch.zeros([int(self.x)], device=self.cuda0) 
        linear_traj = 240*torch.ones([int(self.x) + self.compute_window], device=self.cuda0)
        if self.check_intersect(obstacles, linear_traj):
            ## then we go for a non linear trajectory haaha
            return linear_traj
        return linear_traj
    
    def new_navigation(self, obstacles, screen_size):
        # Base cost map
        cost_map = torch.zeros(screen_size, device=self.cuda0, dtype=torch.float32)

        # If no obstacles are visible, just pull gently to the center to stop random drifting
        if len(obstacles) == 0:
            y_coords = torch.arange(0, screen_size, device=self.cuda0, dtype=torch.float32)
            cost_map += torch.abs(y_coords - screen_size // 2) * 0.1
            self.latest_cost_map_cpu = cost_map.detach().cpu().numpy()
            return cost_map

        # Ship's current X position
        ship_x = self.barycenter[0]

        for i in range(len(obstacles)):
            x1 = obstacles[i, 0]
            y1 = max(0, int(obstacles[i, 1].item()))
            y2 = min(screen_size - 1, int(obstacles[i, 3].item()))
            
            # 1. Depth Weighting (Urgency)
            # Obstacles right in front of us have a weight of 1.0. 
            # Obstacles at the edge of the compute_window have a weight near 0.1.
            dist_x = float(x1 - ship_x)
            urgency = max(0.1, (self.compute_window - dist_x) / self.compute_window) if dist_x > 0 else 1.0
            
            # Hard-fill the physical interior of the obstacle
            cost_map[y1:y2+1] += 10000.0 * urgency
            
            # 2. Build the "Pit" (Local Repulsion Field)
            # We hand-draw a parabolic slope that extends exactly 60 pixels 
            # above and below the obstacle, creating a perfectly smooth, stable center in gaps.
            buffer_px = 60 
            
            if y1 > 0: # Top Repulsion Pit
                start_idx = max(0, y1 - buffer_px)
                length = y1 - start_idx
                if length > 0:
                    y_idx = torch.arange(length, device=self.cuda0, dtype=torch.float32)
                    # Parabola: highest at the edge of the box, drops to 0 at the edge of the buffer
                    repulsion = ((y_idx + 1) / length) ** 2 
                    cost_map[start_idx:y1] += repulsion * 500.0 * urgency
                    
            if y2 < screen_size - 1: # Bottom Repulsion Pit
                end_idx = min(screen_size, y2 + 1 + buffer_px)
                length = end_idx - (y2 + 1)
                if length > 0:
                    y_idx = torch.arange(length, device=self.cuda0, dtype=torch.float32)
                    # Parabola: highest at the box edge, fading downwards
                    repulsion = (1.0 - (y_idx / length)) ** 2
                    cost_map[y2 + 1:end_idx] += repulsion * 500.0 * urgency

        self.latest_cost_map_cpu = cost_map.detach().cpu().numpy()
        return cost_map
    
    
    def get_heatmap_frame(self, bar_width=150, legend_width=70):
        if self.latest_cost_map_cpu is None:
            return None

        h = self.latest_cost_map_cpu.shape[0]
        cm_clipped = np.clip(self.latest_cost_map_cpu, 0, 1000.0)
        normalized = (cm_clipped / 1000.0 * 255).astype(np.uint8)

        # widened cost bar
        heatmap_bar = np.tile(normalized[:, np.newaxis], (1, bar_width))
        colored_bar = cv.applyColorMap(heatmap_bar, cv.COLORMAP_TURBO)

        # marker for the rocket's current height, handy for calibrating by eye
        y_pos = int(max(0, min(h - 1, self.barycenter[1].item())))
        cv.line(colored_bar, (0, y_pos), (bar_width, y_pos), (255, 255, 255), 2)

        # legend: same colormap, gradient 0 (top, safe) -> 1000 (bottom, danger)
        gradient = np.linspace(0, 255, h, dtype=np.uint8)
        gradient_bar = cv.applyColorMap(np.tile(gradient[:, np.newaxis], (1, 30)), cv.COLORMAP_TURBO)

        legend = np.full((h, legend_width, 3), 30, dtype=np.uint8)
        legend[:, :30] = gradient_bar
        cv.putText(legend, "SAFE", (32, 20), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv.LINE_AA)
        cv.putText(legend, "DANGER", (32, h - 10), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv.LINE_AA)

        return np.hstack([colored_bar, legend])

    
    def behave(self, rectangles, cls):
        self.dt = time.time() - self.last_t
        
        # Prevent division by zero on the very first frame
        if self.dt <= 0: 
            self.dt = 0.001 
            
        self.speed[1] = (self.barycenter[1] - self.last_y_pos) / self.dt
        self.last_y_pos = self.barycenter[1].clone() if torch.is_tensor(self.barycenter[1]) else self.barycenter[1]

        obstacles = self.get_obstacles(rectangles, cls)
        cost_map = self.new_navigation(obstacles, self.screen_h)
        
       # 1. Target Smoothing (EMA) with Danger Override
        raw_target = torch.argmin(cost_map).float()
        
        if self.ema_target is None:
            self.ema_target = raw_target
        else:
            # Check the cost map at the current smoothed target's position
            current_target_idx = int(self.ema_target.clamp(0, self.screen_h - 1).item())
            
            # Condition 1: Is the smoothed target literally inside a newly detected wall? (Cost >= 1000)
            # Condition 2: Did the safe zone jump by a massive amount suddenly? (> 100 pixels)
            if cost_map[current_target_idx] >= 1000.0 or abs(raw_target - self.ema_target) > 100:
                # Emergency snap: Abandon smoothing to survive
                self.ema_target = raw_target
            else:
                # Safe gliding: Apply standard EMA
                self.ema_target = self.beta * self.ema_target + (1 - self.beta) * raw_target
        
        # dist_to_curve: 
        # Negative means target is ABOVE the rocket (needs thrust)
        # Positive means target is BELOW the rocket (needs to fall)
        dist_to_curve = float(self.ema_target - self.barycenter[1])

        # 2. Asymmetric Damping (PD Control)
        # speed[1] < 0 means moving UP (towards y=0)
        # speed[1] > 0 means moving DOWN (towards y=screen_h)
        if self.speed[1] < 0:
            damping = self.K_up * self.speed[1]
        else:
            damping = self.K_down * self.speed[1]

        # The Proportional-Derivative error term
        pd_error = dist_to_curve - damping

        # 3. Actuation Logic
        # Regulate update frequency to prevent overloading input buffers
        if time.time() - self.last_t > 0.02: 
            
            # Hover state (within safe tunnel)
            if abs(dist_to_curve) < self.delta:
                self.alpha = 6
                self.stationary_gas()
                
            # Rocket is too low (Target is ABOVE, pd_error is very negative)
            elif pd_error < -self.delta:
                if pd_error < -200:
                    self.alpha = 5 # Full power for large corrections
                elif pd_error < -125:
                    self.alpha = 4
                elif pd_error < -50:
                    self.alpha = 3 # Slow down
                else:
                    self.alpha = 2 # Precision steering
                
                self.gas(self.alpha)
                
            # Rocket is too high (Target is BELOW, pd_error is positive)
            elif pd_error > self.delta:
                # The original code lacked an explicit drop command here
                self.close_gas() 
                self.alpha = 0 # Indicates freefall

            self.last_t = time.time()

    """
    def behave(self, rectangles, cls):
        self.dt = time.time() - self.last_t
        
        if not hasattr(self, 'vy'):
            self.vy = 0.0  

        # 1. Sélection des accélérations (repère Y vers le bas)
        if self.gas_open:
            acceleration = -1117.64328 if self.vy <= -50.0 else -1397.0491
        else:
            acceleration = -1341.1719 if self.vy <= -50.0 else 894.11464
            
        # 2. Adaptation dynamique du gain K selon le signe de l'accélération
        # Si acceleration < 0 -> On pousse vers le haut (Montée)
        # Si acceleration > 0 -> La gravité nous tire vers le bas (Descente)
        current_K = self.K_up if acceleration < 0 else self.K_down

        # 3. Mise à jour et bridage de la vitesse
        raw_speed = self.vy + acceleration * self.dt
        self.vy = max(min(raw_speed, 345.6), -432.0) 
        
        if self.dt > 0.01:
            obstacles = self.get_obstacles(rectangles, cls)
            target = torch.argmin(self.new_navigation(obstacles, self.screen_h))
            dist_to_curve = int(self.distance_to_barycenter(target))

            # 4. Commande anticipée avec le gain commuté
            sliding_surface = dist_to_curve + current_K * self.vy
            
            if sliding_surface < 0:
                self.open_gas()
            else:
                self.close_gas()

        self.last_t = time.time()
    """
    """
    def behave(self, rectangles, cls):
        self.dt = time.time() - self.last_t
        acceleration = (self.thrust if self.gas_open else 0) + self.gravity
        raw_speed = self.speed + self.dt * acceleration
        self.speed = max(min(raw_speed, 432.0), -345.6)
        if self.dt > 0.01:
            obstacles = self.get_obstacles(rectangles, cls)
            #curve = self.navigation(obstacles)
            target = torch.argmin(self.new_navigation(obstacles, self.screen_h))
            dist_to_curve = int(self.distance_to_barycenter(target))
            print(dist_to_curve, "target :", target)
            #print(self.barycenter)

            if dist_to_curve + self.K*self.speed < 0 :
                self.open_gas()
            else :
                self.close_gas()

        self.last_t = time.time() 
    """

    """
    if dist_to_curve + self.delta < 0:
        if dist_to_curve > 200:
            self.alpha = 6 # Full power for large corrections
        elif dist_to_curve > 100:
            self.alpha = 3 # Slow down
        else:
            self.alpha = 2 # Precision steering
        self.gas(self.alpha)
        #alpha = min(-dist_to_curve//10, 5)
        print("not in delta tunnel", self.alpha)
    if abs(dist_to_curve) < self.delta :
        self.alpha = 5
        self.stationary_gas()
    """
        


### for the mean time, this is more than enough, we will maybe add new class
### later on when we see that the bot already performs well
