import os
import time
import numpy as np
import torch
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

from fly_env import FlyFlightEnv
from train import ConnectomeActor

def euler_to_quaternion(yaw, pitch, roll=0.0):
    cy = np.cos(yaw * 0.5); sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5); sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5); sr = np.sin(roll * 0.5)
    return [cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy]

def build_mujoco_xml():
    xml = """
    <mujoco model="biological_fruit_fly">
      <compiler angle="radian"/>
      <option gravity="0 0 -9.81" timestep="0.002"/>

      <visual>
        <global azimuth="140" elevation="-30"/>
        <headlight ambient="0.5 0.5 0.5" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>
        <rgba haze="0.4 0.5 0.6 1"/>
      </visual>

      <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0.1 0.1 0.15" width="512" height="512"/>
        <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.25 0.3" rgb2="0.25 0.3 0.35" width="300" height="300" mark="edge" markrgb="0.8 0.8 0.8"/>
        <material name="floor_mat" texture="grid" texrepeat="60 12" texuniform="true"/>
      </asset>

      <worldbody>
        <light pos="0 0 25" dir="0 0 -1" diffuse="0.9 0.9 0.9" castshadow="true"/>
        <geom name="floor" type="plane" size="100 60 0.1" pos="100 0 0" material="floor_mat"/>
        <geom name="finish" type="box" size="2 60 12" pos="200 0 12" rgba="0 1 0.6 0.4"/>
    """
    
    for i in range(250):
        xml += f'    <geom name="b{i}" type="box" size="5 5 5" pos="0 0 -20" rgba="0.35 0.4 0.5 1"/>\n'

    xml += """
        <body name="fly_torso" pos="0 0 1.2">
          <joint name="root" type="free"/>
          <camera name="First_Person" pos="0.45 0 0.05" xyaxes="0 -1 0 0 0 1"/>
          
          <geom name="thorax" type="ellipsoid" size="0.45 0.25 0.22" rgba="0.1 0.1 0.1 1"/>
          <geom name="head" type="sphere" size="0.18" pos="0.35 0 0.05" rgba="0.2 0.2 0.2 1"/>
          <geom name="eye_l" type="sphere" size="0.08" pos="0.4 0.15 0.1" rgba="0.85 0.1 0.1 1"/>
          <geom name="eye_r" type="sphere" size="0.08" pos="0.4 -0.15 0.1" rgba="0.85 0.1 0.1 1"/>
          
          <body name="wing_left" pos="0 0.3 0.1">
            <joint name="hinge_l" type="hinge" axis="1 0 0" range="-1.5 1.5"/>
            <geom name="wing_geom_l" type="box" size="0.15 0.5 0.01" pos="0 0.25 0" rgba="0.8 0.95 1 0.85"/>
          </body>
          <body name="wing_right" pos="0 -0.3 0.1">
            <joint name="hinge_r" type="hinge" axis="1 0 0" range="-1.5 1.5"/>
            <geom name="wing_geom_r" type="box" size="0.15 0.5 0.01" pos="0 -0.25 0" rgba="0.8 0.95 1 0.85"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    return xml

def extract_live_neural_firing(actor_model, observation, device):
    with torch.no_grad():
        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        sensory_act = torch.relu(actor_model.sensory_projector(obs_t))
        w_eff = actor_model.bio_adj * actor_model.synaptic_gain
        h = torch.zeros((1, 83), device=device)
        leak = 0.65
        for _ in range(4):
            h = h.clone()
            h[:, :16] = sensory_act
            synaptic_current = torch.matmul(h, w_eff)
            h = (1.0 - leak) * h + leak * torch.sigmoid(synaptic_current)
        return h.squeeze(0).cpu().numpy()

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

def run_mujoco():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing Native MuJoCo + Neural Dashboard on: {device}")

    # 200m Showcase
    env = FlyFlightEnv(arena_length=200.0, arena_width=80.0)
    actor = ConnectomeActor(obs_dim=env.observation_space.shape[0], act_dim=env.action_space.shape[0]).to(device)

    model_path = "fly_connectome_actor.pt"
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
        actor.load_state_dict(ckpt["actor_state_dict"])
        actor.eval()

    obs, _ = env.reset()
    xml_string = build_mujoco_xml()
    mj_model = mujoco.MjModel.from_xml_string(xml_string)
    mj_data = mujoco.MjData(mj_model)
    
    for i in range(250):
        geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"b{i}")
        if i < len(env.obstacles):
            obs_data = env.obstacles[i]
            mj_model.geom_pos[geom_id] = [obs_data["x"], obs_data["y"], obs_data["h"]/2.0]
            mj_model.geom_size[geom_id] = [obs_data["w"]/2.0, obs_data["w"]/2.0, obs_data["h"]/2.0]
        else:
            mj_model.geom_pos[geom_id] = [0, 0, -20]

    fig, scat_lptc, scat_epg, scat_dn, status_text = setup_3d_brain()
    cam_state = {"mode": 0}

    def key_callback(keycode):
        if keycode == 67 or keycode == 99:
            cam_state["mode"] = (cam_state["mode"] + 1) % 3

    try: viewer = mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_callback)
    except TypeError: viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

    step_dt = 0.04
    time_scale = 0.4
    sim_timer = 0.0

    prev_pos = np.array(env.pos, dtype=np.float64); target_pos = np.array(env.pos, dtype=np.float64)
    prev_yaw = env.yaw; target_yaw = env.yaw
    prev_pitch = env.pitch; target_pitch = env.pitch
    last_u_left, last_u_right = 0.0, 0.0

   
    RESET_HOLD_SECONDS = 0.6
    pending_reset_reason = None
    pending_reset_timer = 0.0

    def rebuild_city_geoms():
        for i in range(250):
            geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"b{i}")
            if i < len(env.obstacles):
                obs_data = env.obstacles[i]
                mj_model.geom_pos[geom_id] = [obs_data["x"], obs_data["y"], obs_data["h"]/2.0]
                mj_model.geom_size[geom_id] = [obs_data["w"]/2.0, obs_data["w"]/2.0, obs_data["h"]/2.0]
                mj_model.geom_rgba[geom_id] = [0.5, 0.48, 0.4, 1.0] if obs_data["h"] <= 8.0 else [0.35, 0.4, 0.5, 1.0]
            else:
                mj_model.geom_pos[geom_id] = [0, 0, -20]

    with viewer:
        wing_phase = 0.0; last_mode = -1; step_counter = 0
        last_wall_time = time.perf_counter()

        while viewer.is_running():
            now = time.perf_counter()
            frame_dt = now - last_wall_time
            last_wall_time = now

            if cam_state["mode"] != last_mode:
                last_mode = cam_state["mode"]
                if last_mode == 0:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    viewer.cam.trackbodyid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "fly_torso")
                    viewer.cam.distance = 5.5 
                    viewer.cam.elevation = -12.0
                    viewer.cam.azimuth = 180.0
                elif last_mode == 1:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "First_Person")
                elif last_mode == 2:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

            if pending_reset_reason is not None:
             
                pending_reset_timer -= frame_dt
                if pending_reset_timer <= 0.0:
                    pending_reset_reason = None
                    obs, _ = env.reset()
                    rebuild_city_geoms()
                    prev_pos = np.array(env.pos, dtype=np.float64); target_pos = np.array(env.pos, dtype=np.float64)
                    prev_yaw = env.yaw; target_yaw = env.yaw
                    prev_pitch = env.pitch; target_pitch = env.pitch
            else:
                sim_timer += frame_dt * time_scale

                if sim_timer >= step_dt:
                    sim_timer -= step_dt

                    prev_pos = target_pos.copy()
                    prev_yaw = target_yaw
                    prev_pitch = target_pitch

                    with torch.no_grad():
                        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                        action = actor.get_deterministic_action(obs_t).squeeze(0).cpu().numpy()

                    obs, _, terminated, truncated, info = env.step(action)
                    last_u_left, last_u_right, _ = action

                    target_pos = np.array(env.pos, dtype=np.float64)
                    target_yaw = env.yaw
                    target_pitch = env.pitch

                    if terminated or truncated:
                        
                        sim_timer = step_dt
                        pending_reset_reason = info.get("flight_stats", {}).get("reason", "Episode End")
                        pending_reset_timer = RESET_HOLD_SECONDS

            fraction = min(sim_timer / step_dt, 1.0) if step_dt > 0 else 1.0
            interp_pos = prev_pos + (target_pos - prev_pos) * fraction
            interp_yaw = prev_yaw + (target_yaw - prev_yaw) * fraction
            interp_pitch = prev_pitch + (target_pitch - prev_pitch) * fraction

            mj_data.qpos[0] = interp_pos[0]; mj_data.qpos[1] = interp_pos[1]; mj_data.qpos[2] = interp_pos[2]
            mj_data.qpos[3:7] = euler_to_quaternion(interp_yaw, -interp_pitch)

           
            wing_phase += 0.8
            mj_data.qpos[7] = np.sin(wing_phase) * (0.6 + last_u_left * 0.4)
            mj_data.qpos[8] = -np.sin(wing_phase) * (0.6 + last_u_right * 0.4)

            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

            if step_counter % 12 == 0:
                firing = extract_live_neural_firing(actor, obs, device)
                update_3d_brain(firing, scat_lptc, scat_epg, scat_dn, status_text, fig)
            
            step_counter += 1
            time.sleep(0.015)

if __name__ == "__main__":
    run_mujoco()