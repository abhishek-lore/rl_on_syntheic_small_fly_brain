import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FlyFlightEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, arena_length: float = 200.0, arena_width: float = 80.0):
        super().__init__()
        self.arena_length = arena_length
        self.arena_width = arena_width
        self.max_ray_dist = 45.0

        self.ray_vectors = []
        for az in np.linspace(-np.pi * 0.45, np.pi * 0.45, 8):
            self.ray_vectors.append(np.array([np.cos(az), np.sin(az), 0.0], dtype=np.float32))
        for az in np.linspace(-np.pi * 0.35, np.pi * 0.35, 6):
            self.ray_vectors.append(np.array([np.cos(az) * 0.88, np.sin(az) * 0.88, 0.47], dtype=np.float32))
        for az in np.linspace(-np.pi * 0.35, np.pi * 0.35, 6):
            self.ray_vectors.append(np.array([np.cos(az) * 0.88, np.sin(az) * 0.88, -0.47], dtype=np.float32))
        self.ray_vectors.extend([
            np.array([0.0, 1.0, 0.0], dtype=np.float32), np.array([0.0, -1.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, -1.0], dtype=np.float32), np.array([0.0, 0.0, 1.0], dtype=np.float32)
        ])

        self.num_rays = len(self.ray_vectors)
        # Precomputed once, reused every call to _cast_3d_compound_eye instead
        # of rebuilding/looping in Python each time.
        self.ray_vectors_arr = np.stack(self.ray_vectors).astype(np.float32)  # (num_rays, 3)
        self._ray_sample_d = np.linspace(0.5, self.max_ray_dist, 32, dtype=np.float32)  # (32,)
        self._front_ray_idx = np.array([3, 4, 10, 11, 16, 17], dtype=np.int64)

        self.obs_dim = self.num_rays + 7
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.resting_height = 0.35
        self.ideal_cruise_clearance = 2.5
        self.ceiling_height = 24.0
        
        
        self.fly_radius = 0.55
        self.dt = 0.04
  
        self.cruise_speed = 1.2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._build_procedural_complex_city()

        self.surface_base_z = 0.0
        spawn_x = np.random.uniform(2.0, 10.0)
        spawn_y = np.random.uniform(-15.0, 15.0)

        self.pos = np.array([spawn_x, spawn_y, self.surface_base_z + 1.2], dtype=np.float32)
        self.start_x = float(spawn_x)
        self.vel = np.array([self.cruise_speed, 0.0, 1.2], dtype=np.float32)
        self.yaw = float(np.random.uniform(-0.05, 0.05))
        self.pitch = 0.04
        self.steps = 0
        self.wing_l = 0.6
        self.wing_r = 0.6
  
        self.episode_shaping_accum = 0.0

        return self._get_obs(), {}

    def _build_procedural_complex_city(self):
        self.obstacles = []
        self.low_roof_candidates = []

        edges = [-36.0, 36.0]
        mids = [-18.0, 0.0, 18.0]

        for x in np.arange(35.0, self.arena_length - 20.0, 22.0):
            chosen_edges = np.random.choice(edges, size=np.random.randint(1, 3), replace=False)
            chosen_mids = np.random.choice(mids, size=np.random.randint(1, 3), replace=False)
            chosen_zones = np.concatenate([chosen_edges, chosen_mids])

            for y in chosen_zones:
                w = np.random.uniform(7.0, 10.0)
                h = np.random.uniform(5.0, 30.0)
                x_off = x + np.random.uniform(-3.0, 3.0)
                y_off = y + np.random.uniform(-1.0, 1.0)
                
                b_data = {"x": float(x_off), "y": float(y_off), "w": float(w), "h": float(h)}
                self.obstacles.append(b_data)
                if h <= 8.0 and x < 240.0:
                    self.low_roof_candidates.append(b_data)

     .
        if len(self.obstacles) > 0:
            self.obs_x = np.array([o["x"] for o in self.obstacles], dtype=np.float32)
            self.obs_y = np.array([o["y"] for o in self.obstacles], dtype=np.float32)
            self.obs_w = np.array([o["w"] for o in self.obstacles], dtype=np.float32)
            self.obs_h = np.array([o["h"] for o in self.obstacles], dtype=np.float32)
        else:
            self.obs_x = np.zeros(0, dtype=np.float32)
            self.obs_y = np.zeros(0, dtype=np.float32)
            self.obs_w = np.zeros(0, dtype=np.float32)
            self.obs_h = np.zeros(0, dtype=np.float32)

    def _get_local_surface_height(self, px: float, py: float) -> float:
        surface_z = 0.0
        for obs in self.obstacles:
            if (obs["x"] - obs["w"]/2 <= px <= obs["x"] + obs["w"]/2 and
                obs["y"] - obs["w"]/2 <= py <= obs["y"] + obs["w"]/2):
                if obs["h"] > surface_z: surface_z = obs["h"]
        return surface_z

    def _cast_3d_compound_eye(self):
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        R = np.array([[cy * cp, -sy, cy * sp], [sy * cp, cy, sy * sp], [-sp, 0, cp]], dtype=np.float32)

        # (num_rays, 3): rotate every local ray into world space in one matmul
        # instead of one R @ local_ray per ray in a Python loop.
        world_rays = self.ray_vectors_arr @ R.T

        d_vals = self._ray_sample_d  

        # (num_rays, num_samples, 3): every sample point along every ray, all
        # at once via broadcasting.
        points = self.pos[None, None, :] + world_rays[:, None, :] * d_vals[None, :, None]
        px = points[..., 0]; py = points[..., 1]; pz = points[..., 2]  # each (num_rays, num_samples)

        ground_hit = pz <= 0.0

        if self.obs_x.shape[0] > 0:
            ox = self.obs_x[:, None, None]; oy = self.obs_y[:, None, None]
            ow = self.obs_w[:, None, None]; oh = self.obs_h[:, None, None]
            # (num_obstacles, num_rays, num_samples): test every obstacle
            # against every sample point of every ray in one shot.
            in_box = (
                (px[None, :, :] >= ox - ow * 0.5) & (px[None, :, :] <= ox + ow * 0.5) &
                (py[None, :, :] >= oy - ow * 0.5) & (py[None, :, :] <= oy + ow * 0.5) &
                (pz[None, :, :] <= oh)
            )
            obstacle_hit = in_box.any(axis=0)  # (num_rays, num_samples)
        else:
            obstacle_hit = np.zeros_like(ground_hit)

        combined_hit = ground_hit | obstacle_hit  

      
        hit_exists = combined_hit.any(axis=1)
        first_idx = np.argmax(combined_hit, axis=1)
        distances = np.where(hit_exists, d_vals[first_idx], self.max_ray_dist).astype(np.float32)

        min_front_dist = float(np.min(distances[self._front_ray_idx]))

        return distances / self.max_ray_dist, min_front_dist

    def _get_obs(self):
        norm_rays, min_front_dist = self._cast_3d_compound_eye()
        local_surf = self._get_local_surface_height(self.pos[0], self.pos[1])
        clearance = np.clip((self.pos[2] - local_surf) / 10.0, 0.0, 3.0)
        kinematics = np.array([
            self.vel[0], self.vel[1], self.vel[2],
            self.yaw, self.pitch, clearance,
            min_front_dist / self.max_ray_dist
        ], dtype=np.float32)
        return np.concatenate([norm_rays, kinematics])

    def step(self, action):
        self.steps += 1
        u_left, u_right, u_pitch_lift = np.clip(action, -1.0, 1.0)

        self.wing_l = float(np.clip(0.5 + u_left * 0.5, 0.2, 1.0))
        self.wing_r = float(np.clip(0.5 + u_right * 0.5, 0.2, 1.0))

        thrust = (self.wing_l + self.wing_r) * 0.5 * 3.6
        
       
        yaw_torque = (self.wing_l - self.wing_r) * 8.0 

        drag_x = -0.85 * self.vel[0]; drag_y = -2.0 * self.vel[1]; drag_z = -1.5 * self.vel[2]

      
        self.yaw += yaw_torque * self.dt
        self.yaw *= 0.85 
        self.yaw = np.clip(self.yaw, -1.2, 1.2)
        
        self.pitch = np.clip(self.pitch + u_pitch_lift * 0.8 * self.dt - self.pitch * 0.25, -0.35, 0.35)

        prev_x = float(self.pos[0])
        self.pos[0] = np.clip(self.pos[0] + self.vel[0] * self.dt * 6.0, 0.0, self.arena_length + 10)
        self.pos[1] = np.clip(self.pos[1] + self.vel[1] * self.dt * 6.0, -self.arena_width, self.arena_width)
        
        local_surf = self._get_local_surface_height(self.pos[0], self.pos[1])
        clearance = self.pos[2] - local_surf

        excess_clearance = max(0.0, clearance - 5.0)
        downward_wash = excess_clearance * 2.8
        lift_accel = 9.81 + (u_pitch_lift * 4.6) - downward_wash

        ax = thrust * np.cos(self.yaw) * np.cos(self.pitch) + drag_x
        ay = thrust * np.sin(self.yaw) * np.cos(self.pitch) + drag_y
        az = (lift_accel - 9.81) + thrust * np.sin(self.pitch) + drag_z

        self.vel[0] = np.clip(self.vel[0] + ax * self.dt, self.cruise_speed, 4.8)
        self.vel[1] = np.clip(self.vel[1] + ay * self.dt, -2.5, 2.5)
        self.vel[2] = np.clip(self.vel[2] + az * self.dt, -3.0, 3.0)
        self.pos[2] += self.vel[2] * self.dt * 6.0

        min_allowed_z = local_surf + self.resting_height
        if self.pos[2] < min_allowed_z:
            self.pos[2] = min_allowed_z
            if self.vel[2] < 0: self.vel[2] = 0.5
        if self.pos[2] > self.ceiling_height:
            self.pos[2] = self.ceiling_height
            if self.vel[2] > 0: self.vel[2] = -0.5

        clearance = self.pos[2] - local_surf
        hit = False
        termination_reason = "none"

        if abs(self.pos[1]) >= self.arena_width / 2.0:
            hit = True; termination_reason = "arena_bounds"
        else:
            for obs in self.obstacles:
                if self.pos[2] <= (obs["h"] - 0.05):
                    if (obs["x"] - obs["w"]/2 - self.fly_radius <= self.pos[0] <= obs["x"] + obs["w"]/2 + self.fly_radius and
                        obs["y"] - obs["w"]/2 - self.fly_radius <= self.pos[1] <= obs["y"] + obs["w"]/2 + self.fly_radius):
                        hit = True; termination_reason = "wall_impact"; break

        norm_rays, min_front_dist = self._cast_3d_compound_eye()
        raw_ray_dists = norm_rays * self.max_ray_dist

        dx = max(0.0, self.pos[0] - prev_x)
        r_flight = dx * 3.4 * float(np.exp(- ((clearance - self.ideal_cruise_clearance) ** 2) / 2.5))
       
        r_heading = 0.05 * np.cos(self.yaw)
        r_sky_penalty = - 0.20 * ((clearance - 6.0) ** 2) if clearance > 6.0 else 0.0

        threats = np.maximum(0.0, 1.0 - (raw_ray_dists[:22] / 4.5))
        r_barrier = - 0.85 * float(np.sum(threats ** 2))

        right_dist = np.mean(raw_ray_dists[0:4])
        left_dist = np.mean(raw_ray_dists[4:8])
        escape_direction = np.tanh((left_dist - right_dist) / 3.0)

        r_saccade = 0.0; r_looming = 0.0
        
        if min_front_dist < 12.0:
            r_looming = - 0.6 * (self.vel[0] / max(min_front_dist, 0.5))

        if min_front_dist < 10.0:
         
            r_saccade = 0.8 * (yaw_torque * escape_direction)

        r_smooth = - 0.02 * (yaw_torque ** 2) - 0.006 * (u_left**2 + u_right**2 + u_pitch_lift**2)

      
        r_edge = 0.0
        if abs(self.pos[1]) > 28.0:
            r_edge = -0.5 * (abs(self.pos[1]) - 28.0)

        reward = r_flight + r_heading + r_sky_penalty + r_barrier + r_looming + r_saccade + r_smooth + r_edge
    
        reward = float(np.clip(reward, -2.0, 2.0))

        terminated = hit or self.pos[0] >= self.arena_length
        truncated = self.steps >= 1000

        if hit:
          
            progress_frac = min(1.0, max(0.0, self.pos[0] / self.arena_length))
            base_penalty = -100.0 - 100.0 * (1.0 - progress_frac)
         
            crash_ceiling = -10.0
            projected_total = self.episode_shaping_accum + base_penalty
            if projected_total > crash_ceiling:
                reward = crash_ceiling - self.episode_shaping_accum
            else:
                reward = base_penalty
        elif self.pos[0] >= self.arena_length:
            termination_reason = "goal_reached"
     
            goal_bonus = 500.0
            success_floor = 100.0
            projected_total = self.episode_shaping_accum + goal_bonus
            if projected_total < success_floor:
                reward = goal_bonus + (success_floor - projected_total)
            else:
                reward = goal_bonus
        elif truncated:
            termination_reason = "step_timeout"
        else:
          
            self.episode_shaping_accum += reward

        info = {
            "flight_stats": {
                "distance_flown": float(self.pos[0] - self.start_x),
                "final_x": float(self.pos[0]),
                "final_z": float(self.pos[2]),
                "clearance": float(clearance),
                "steps": int(self.steps),
                "reason": termination_reason
            }
        }
        return self._get_obs(), float(reward), terminated, truncated, info