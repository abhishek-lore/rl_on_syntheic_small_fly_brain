import os
import time
import numpy as np
import torch
from ursina import *
import matplotlib.pyplot as plt

from fly_env import FlyFlightEnv
from train import ConnectomeActor

# 1. 3D LIVE NEURAL DASHBOARD SETUP

def setup_3d_brain():
    plt.ion()
    fig = plt.figure(figsize=(7, 9))
    fig.canvas.manager.set_window_title('Live 3D Neural Connectome')
    fig.patch.set_facecolor('#0d0d14')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0d0d14')

    x_lptc = np.linspace(-2.5, 2.5, 16); y_lptc = np.zeros(16); z_lptc = np.full(16, 2.0)
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    x_epg = np.cos(angles) * 1.5; y_epg = np.sin(angles) * 1.5; z_epg = np.full(32, 1.0)
    x_dn = [-1.5, 1.5, 0.0]; y_dn = [0.0, 0.0, -0.5]; z_dn = [0.0, 0.0, 0.0]

    scat_lptc = ax.scatter(x_lptc, y_lptc, z_lptc, c='#004444', s=100, alpha=0.9, edgecolors='w')
    scat_epg = ax.scatter(x_epg, y_epg, z_epg, c='#444400', s=100, alpha=0.9, edgecolors='w')
    scat_dn = ax.scatter(x_dn, y_dn, z_dn, c='#440000', s=300, alpha=1.0, edgecolors='w')

    for i in range(16): ax.plot([x_lptc[i], 0], [y_lptc[i], 0], [2.0, 1.0], color='#00ffcc', alpha=0.10)
    for i in range(32):
        ax.plot([x_epg[i], x_dn[0]], [y_epg[i], y_dn[0]], [1.0, 0.0], color='#ffcc00', alpha=0.05)
        ax.plot([x_epg[i], x_dn[1]], [y_epg[i], y_dn[1]], [1.0, 0.0], color='#ffcc00', alpha=0.05)

    ax.text2D(0.5, 0.95, "LPTC: Compound Vision (Input)", transform=ax.transAxes, color='#00ffcc', ha='center', fontsize=11, fontweight='bold')
    ax.text2D(0.5, 0.50, "E-PG: Spatial Compass (Hidden)", transform=ax.transAxes, color='#ffcc00', ha='center', fontsize=11, fontweight='bold')
    ax.text2D(0.5, 0.05, "DNs: Motor Actuators (Action)", transform=ax.transAxes, color='#ff3366', ha='center', fontsize=11, fontweight='bold')

    status_text = ax.text2D(0.5, 0.80, "SYSTEM ONLINE", transform=ax.transAxes, color='white', 
                            ha='center', fontsize=13, fontweight='bold', 
                            bbox=dict(facecolor='black', alpha=0.6, edgecolor='#00ffcc'))

    ax.text(x_dn[0], y_dn[0], z_dn[0] - 0.2, "L-Wing", color='white', fontsize=10, ha='center')
    ax.text(x_dn[1], y_dn[1], z_dn[1] - 0.2, "R-Wing", color='white', fontsize=10, ha='center')
    ax.text(x_dn[2], y_dn[2], z_dn[2] - 0.2, "Thrust", color='white', fontsize=10, ha='center')

    ax.set_axis_off(); ax.set_xlim(-3.0, 3.0); ax.set_ylim(-3.0, 3.0); ax.set_zlim(-0.5, 2.5)
    ax.view_init(elev=20, azim=-75)
    plt.tight_layout()
    return fig, scat_lptc, scat_epg, scat_dn, status_text

def update_3d_brain(firing, scat_lptc, scat_epg, scat_dn, status_text, fig):
    lptc_fires = firing[:16]
    lptc_colors = np.zeros((16, 4))
    lptc_colors[:, 1] = lptc_fires * 0.9 + 0.1 
    lptc_colors[:, 2] = lptc_fires * 0.9 + 0.1 
    lptc_colors[:, 3] = 0.9
    scat_lptc.set_color(lptc_colors)
    scat_lptc.set_sizes(lptc_fires * 800 + 100)

    epg_fires = firing[16:48]
    epg_colors = np.zeros((32, 4))
    epg_colors[:, 0] = epg_fires * 0.9 + 0.1 
    epg_colors[:, 1] = epg_fires * 0.7 + 0.1
    epg_colors[:, 3] = 0.9
    scat_epg.set_color(epg_colors)
    scat_epg.set_sizes(epg_fires * 800 + 100)

    dn_fires = firing[80:83]
    dn_colors = np.zeros((3, 4))
    dn_colors[0] = [dn_fires[0] * 0.8 + 0.2, 0.1, 0.1, 1.0] 
    dn_colors[1] = [dn_fires[1] * 0.8 + 0.2, 0.1, 0.1, 1.0] 
    dn_colors[2] = [0.1, 0.5, dn_fires[2] * 0.8 + 0.2, 1.0] 
    scat_dn.set_color(dn_colors)
    scat_dn.set_sizes(dn_fires * 1500 + 300)

    vis_max = np.max(lptc_fires)
    sight_str = "PATH CLEAR"
    # SYNCED HUD ALARM
    if vis_max > 0.82:
        sight_str = "⚠️ WALL DETECTED"
        
    action_str = "Cruising Forward ^^"
    if dn_fires[0] > dn_fires[1] + 0.05:
        action_str = "<< Dodging LEFT"
    elif dn_fires[1] > dn_fires[0] + 0.05:
        action_str = "Dodging RIGHT >>"

    status_text.set_text(f"VISION: {sight_str}\nACTION: {action_str}")
    status_text.set_color('#00ffcc' if vis_max < 0.82 else '#ff3366')

    fig.canvas.draw_idle()
    fig.canvas.flush_events()


# 2. URSINA ENGINE & RL ENVIRONMENT SETUP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Initializing Smooth Ursina + 3D Connectome on: {device}")

env = FlyFlightEnv(arena_length=200.0, arena_width=80.0)
actor_model = ConnectomeActor(obs_dim=env.observation_space.shape[0], act_dim=env.action_space.shape[0]).to(device)

model_path = "fly_connectome_actor.pt"
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    actor_model.load_state_dict(checkpoint["actor_state_dict"])
    actor_model.eval()

def extract_live_neural_firing(observation):
    with torch.no_grad():
        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        sensory_act = torch.relu(actor_model.sensory_projector(obs_t))
        w_eff = actor_model.bio_adj * actor_model.synaptic_gain
        h = torch.zeros((1, 83), device=device)
        leak = 0.65
        for _ in range(4):
            h = h.clone()
            h[:, :16] = sensory_act
            h = (1.0 - leak) * h + leak * torch.sigmoid(torch.matmul(h, w_eff))
        return h.squeeze(0).cpu().numpy()

obs, _ = env.reset()

app = Ursina(title="Bio-Inspired 3D Fly Simulator", borderless=False)

window.color = color.rgb(18, 22, 34)
Sky(color=color.rgb(30, 45, 60))

ground = Entity(model='plane', scale=(env.arena_width * 2.2, 1, env.arena_length + 120), position=(0, 0, env.arena_length / 2), color=color.rgb(35, 40, 50), texture='white_cube', texture_scale=(60, 400))

building_entities = []
def sync_buildings():
    global building_entities
    for b in building_entities: destroy(b)
    building_entities.clear()

    for obs_box in env.obstacles:
        b_color = color.rgb(110, 105, 90) if obs_box["h"] <= 8.0 else color.rgb(75, 85, 100)
        b = Entity(model='cube', position=(obs_box["y"], obs_box["h"] / 2, obs_box["x"]), scale=(obs_box["w"], obs_box["h"], obs_box["w"]), color=b_color, texture='white_cube')
        building_entities.append(b)

sync_buildings()
Entity(model='cube', position=(0, 12, env.arena_length), scale=(env.arena_width, 24, 2), color=color.rgba(0, 255, 160, 95))

fly_root = Entity(position=(env.pos[1], env.pos[2], env.pos[0]))
fly_thorax = Entity(parent=fly_root, model='sphere', color=color.black, scale=(0.9, 0.6, 1.4))
fly_head = Entity(parent=fly_thorax, model='sphere', color=color.dark_gray, scale=(0.7, 0.7, 0.6), position=(0, 0.1, 0.7))
Entity(parent=fly_head, model='sphere', color=color.red, scale=(0.38, 0.38, 0.38), position=(-0.4, 0.2, 0.3))
Entity(parent=fly_head, model='sphere', color=color.red, scale=(0.38, 0.38, 0.38), position=(0.4, 0.2, 0.3))

wing_left_hinge = Entity(parent=fly_thorax, position=(-0.4, 0.25, 0.0))
wing_left = Entity(parent=wing_left_hinge, model='plane', color=color.rgba(210, 235, 255, 175), scale=(1.2, 1, 0.6), position=(-0.6, 0, -0.2))
wing_right_hinge = Entity(parent=fly_thorax, position=(0.4, 0.25, 0.0))
wing_right = Entity(parent=wing_right_hinge, model='plane', color=color.rgba(210, 235, 255, 175), scale=(1.2, 1, 0.6), position=(0.6, 0, -0.2))

DirectionalLight(y=40, z=-40, shadows=True)
AmbientLight(color=color.rgba(140, 150, 170, 200))

fig, scat_lptc, scat_epg, scat_dn, status_text = setup_3d_brain()

camera_mode = 0
camera_names = ["Smooth Spring Follow", "Locked Cockpit", "Free-Cam Spectator"]
free_cam_pos = Vec3(0, 15, -20)
free_cam_rot = Vec3(20, 0, 0)
mouse_sensitivity = 40.0; free_cam_speed = 35.0

time_scale = 0.4
sim_timer = 0.0
step_dt = 0.04
wing_stroke_phase = 0.0
frame_counter = 0

prev_render_pos = Vec3(env.pos[1], env.pos[2], env.pos[0])
target_render_pos = Vec3(env.pos[1], env.pos[2], env.pos[0])
prev_yaw = env.yaw; target_yaw = env.yaw
prev_pitch = env.pitch; target_pitch = env.pitch
last_reset_reason = "Spawned"


RESET_HOLD_SECONDS = 0.6
pending_reset_reason = None
pending_reset_timer = 0.0


hud_text = Text(text="", position=(-0.85, 0.45), scale=1.1, color=color.black)

def input(key):
    global camera_mode, time_scale, free_cam_pos, free_cam_rot
    if key == 'c':
        camera_mode = (camera_mode + 1) % 3
        if camera_mode == 2:
            free_cam_pos = Vec3(camera.x, camera.y, camera.z)
            free_cam_rot = Vec3(camera.rotation_x, camera.rotation_y, 0)
    elif key == ']' or key == 'up arrow': time_scale = min(time_scale + 0.05, 1.5)
    elif key == '[' or key == 'down arrow': time_scale = max(time_scale - 0.05, 0.05)

def reset_simulation(reason="Reset"):
    global obs, prev_render_pos, target_render_pos, prev_yaw, target_yaw, prev_pitch, target_pitch, last_reset_reason
    last_reset_reason = reason
    obs, _ = env.reset()
    sync_buildings()
    
    target_render_pos = Vec3(env.pos[1], env.pos[2], env.pos[0])
    prev_render_pos = target_render_pos
    target_yaw = env.yaw; prev_yaw = env.yaw
    target_pitch = env.pitch; prev_pitch = env.pitch
    
    fly_root.position = target_render_pos
    fly_thorax.rotation_y = np.degrees(target_yaw)
    fly_thorax.rotation_x = -np.degrees(target_pitch)
    if camera_mode == 0: camera.position = target_render_pos + Vec3(0, 4.0, -10.0)

def update():
    global obs, sim_timer, free_cam_pos, free_cam_rot, wing_stroke_phase, frame_counter
    global prev_render_pos, target_render_pos, prev_yaw, target_yaw, prev_pitch, target_pitch
    global pending_reset_reason, pending_reset_timer

    if pending_reset_reason is not None:
        # Episode already ended: target_render_pos below is already the exact
        # collision/finish pose (sim_timer was snapped to step_dt when it was
        # detected, so fraction == 1.0 and the fly sits exactly there). Just
        # count down real time before actually swapping in a new episode.
        pending_reset_timer -= time.dt
        if pending_reset_timer <= 0.0:
            reason = pending_reset_reason
            pending_reset_reason = None
            reset_simulation(reason)
    else:
        sim_timer += time.dt * time_scale

        if sim_timer >= step_dt:
            sim_timer -= step_dt

            prev_render_pos = target_render_pos
            prev_yaw = target_yaw
            prev_pitch = target_pitch

            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = actor_model.get_deterministic_action(obs_t).squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action)

            target_render_pos = Vec3(env.pos[1], env.pos[2], env.pos[0])
            target_yaw = env.yaw
            target_pitch = env.pitch

            if terminated or truncated:
                # Snap straight to the terminal pose (instead of leaving the
                # fly mid-interpolation) and hold there before resetting.
                sim_timer = step_dt
                pending_reset_reason = info.get("flight_stats", {}).get("reason", "Episode End")
                pending_reset_timer = RESET_HOLD_SECONDS

    fraction = min(sim_timer / step_dt, 1.0)
    
    fly_root.position = lerp(prev_render_pos, target_render_pos, fraction)
    curr_yaw_deg = np.degrees(lerp(prev_yaw, target_yaw, fraction))
    curr_pitch_deg = -np.degrees(lerp(prev_pitch, target_pitch, fraction))
    
    fly_thorax.rotation_y = curr_yaw_deg
    fly_thorax.rotation_x = curr_pitch_deg

    wing_stroke_phase += time.dt * 45.0
    wing_left_hinge.rotation_z = np.sin(wing_stroke_phase) * (34.0 * env.wing_l)
    wing_right_hinge.rotation_z = -np.sin(wing_stroke_phase) * (34.0 * env.wing_r)

    if camera_mode == 0:
        fly_thorax.visible = True
        cam_dist = 8.5; cam_height = 3.5
        target_cam_x = fly_root.x - np.sin(np.radians(curr_yaw_deg)) * cam_dist
        target_cam_z = fly_root.z - np.cos(np.radians(curr_yaw_deg)) * cam_dist
        target_cam_y = fly_root.y + cam_height

        camera.position = lerp(camera.position, Vec3(target_cam_x, target_cam_y, target_cam_z), time.dt * 8.0)
        look_target = fly_root.position + Vec3(np.sin(np.radians(curr_yaw_deg))*4.0, 0.5, np.cos(np.radians(curr_yaw_deg))*4.0)
        camera.look_at(look_target)

    elif camera_mode == 1:
        fly_thorax.visible = False
        camera.position = fly_root.position + Vec3(0, 0.45, 0.7)
        camera.rotation_y = curr_yaw_deg
        camera.rotation_x = curr_pitch_deg
        camera.rotation_z = 0

    elif camera_mode == 2:
        fly_thorax.visible = True
        if mouse.right:
            free_cam_rot.x -= mouse.velocity[1] * mouse_sensitivity
            free_cam_rot.y += mouse.velocity[0] * mouse_sensitivity
            free_cam_rot.x = clamp(free_cam_rot.x, -89, 89)
        camera.rotation = free_cam_rot
        move_speed = free_cam_speed * (2.5 if held_keys['shift'] else 1.0) * time.dt
        move_vec = Vec3((held_keys['d'] - held_keys['a']), (held_keys['e'] - held_keys['q']), (held_keys['w'] - held_keys['s'])).normalized()
        free_cam_pos += (camera.forward * move_vec.z + camera.right * move_vec.x + Vec3(0, 1, 0) * move_vec.y) * move_speed
        camera.position = free_cam_pos

    clearance = env.pos[2] - env._get_local_surface_height(env.pos[0], env.pos[1])
    if pending_reset_reason is not None:
        event_line = f">>> {pending_reset_reason.upper()} <<< (respawning in {max(0.0, pending_reset_timer):.1f}s)"
    else:
        event_line = f"Event: {last_reset_reason}"
    hud_text.text = f"[C] Mode: {camera_names[camera_mode]} | {event_line}\nWorld X: {env.pos[0]:.1f}m / {env.arena_length:.0f}m | Alt: {env.pos[2]:.1f}m\nSpeed: {env.vel[0]:.2f} m/s | Steps: {env.steps}/1000"

    frame_counter += 1
    if frame_counter % 12 == 0:
        update_3d_brain(extract_live_neural_firing(obs), scat_lptc, scat_epg, scat_dn, status_text, fig)

app.run()