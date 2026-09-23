import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

from fly_env import FlyFlightEnv
from connectome_data import get_flight_circuit

class ConnectomeActor(nn.Module):
    def __init__(self, obs_dim=31, act_dim=3):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        circuit_adj = get_flight_circuit("real_fly_circuit.pt")
        self.register_buffer("bio_adj", circuit_adj)
        self.synaptic_gain = nn.Parameter(torch.ones_like(circuit_adj) * 0.08)
        self.sensory_projector = nn.Linear(obs_dim, 16)

        self.feature_net = nn.Sequential(nn.Linear(3 + obs_dim, 96), nn.LayerNorm(96), nn.ReLU(), nn.Linear(96, 64), nn.ReLU())
        self.mean_layer = nn.Linear(64, act_dim)
        self.log_std_layer = nn.Linear(64, act_dim)
        
        self.world_model_head = nn.Sequential(nn.Linear(64 + act_dim, 128), nn.ReLU(), nn.Linear(128, obs_dim))

    def forward_features(self, obs: torch.Tensor):
        batch_size = obs.shape[0]
        device = obs.device
        sensory_act = torch.relu(self.sensory_projector(obs))
        w_eff = self.bio_adj * self.synaptic_gain
        h = torch.zeros((batch_size, 83), device=device)
        leak = 0.65

        for _ in range(4):
            h = h.clone()
            h[:, :16] = sensory_act 
            synaptic_current = torch.matmul(h, w_eff)
            h = (1.0 - leak) * h + leak * torch.sigmoid(synaptic_current)

        motor_signals = h[:, 80:83]
        fused = torch.cat([motor_signals, obs], dim=-1)
        return self.feature_net(fused)

    def sample(self, obs: torch.Tensor):
        feats = self.forward_features(obs)
        mean = self.mean_layer(feats)
        log_std = torch.clamp(self.log_std_layer(feats), -20.0, 2.0)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def predict_next_state(self, obs: torch.Tensor, act: torch.Tensor):
        feats = self.forward_features(obs)
        wm_input = torch.cat([feats, act], dim=-1)
        return self.world_model_head(wm_input)

    def get_deterministic_action(self, obs: torch.Tensor):
        with torch.no_grad():
            feats = self.forward_features(obs)
            return torch.tanh(self.mean_layer(feats))


class TwinQCritic(nn.Module):
    def __init__(self, obs_dim=31, act_dim=3):
        super().__init__()
        self.q1 = nn.Sequential(nn.Linear(obs_dim + act_dim, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
        self.q2 = nn.Sequential(nn.Linear(obs_dim + act_dim, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: torch.device):
        self.capacity = capacity; self.device = device; self.ptr = 0; self.size = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs; self.actions[self.ptr] = action; self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs; self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity; self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (torch.as_tensor(self.obs[idx], device=self.device), torch.as_tensor(self.actions[idx], device=self.device),
                torch.as_tensor(self.rewards[idx], device=self.device), torch.as_tensor(self.next_obs[idx], device=self.device),
                torch.as_tensor(self.dones[idx], device=self.device))

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Connectome SAC (World Model Policy Gradient) on {device}")

    # SHOWCASE ARENA: 200m
    env = FlyFlightEnv(arena_length=200.0, arena_width=80.0)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    actor = ConnectomeActor(obs_dim, act_dim).to(device)
    critic = TwinQCritic(obs_dim, act_dim).to(device)
    critic_target = TwinQCritic(obs_dim, act_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())

    target_entropy = -float(act_dim)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optim = optim.Adam([log_alpha], lr=5e-4)

    # ACCELERATED LEARNING RATE
    actor_optim = optim.Adam(actor.parameters(), lr=5e-4)
    critic_optim = optim.Adam(critic.parameters(), lr=5e-4)

    buffer = ReplayBuffer(capacity=200_000, obs_dim=obs_dim, act_dim=act_dim, device=device)

    total_timesteps = 60_000
    learning_starts = 800    
    batch_size = 256          
    tau = 0.005

    obs, _ = env.reset()
    episode_reward = 0.0

    print("[*] Training active: 200m Showcase Track...")

    for step in range(1, total_timesteps + 1):
        if step < learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                act_t, _ = actor.sample(obs_t)
                action = act_t.squeeze(0).cpu().numpy()

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward

        buffer.add(obs, action, reward, next_obs, terminated)
        obs = next_obs

        if step >= learning_starts:
            b_obs, b_act, b_rew, b_next_obs, b_done = buffer.sample(batch_size)
            alpha = log_alpha.exp().detach()

            with torch.no_grad():
                next_act, next_log_prob = actor.sample(b_next_obs)
                q1_targ, q2_targ = critic_target(b_next_obs, next_act)
                q_targ = torch.min(q1_targ, q2_targ) - alpha * next_log_prob
                target_q = b_rew + (1.0 - b_done) * gamma * q_targ

            curr_q1, curr_q2 = critic(b_obs, b_act)
            critic_loss = F.mse_loss(curr_q1, target_q) + F.mse_loss(curr_q2, target_q)

            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            new_act, new_log_prob = actor.sample(b_obs)
            q1_curr, q2_curr = critic(b_obs, new_act)
            q_curr = torch.min(q1_curr, q2_curr)
            
            pred_next_obs = actor.predict_next_state(b_obs, new_act)
            wm_loss = F.mse_loss(pred_next_obs, b_next_obs)
            
            actor_loss = (alpha * new_log_prob - q_curr).mean() + 0.5 * wm_loss

            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            alpha_loss = -(log_alpha.exp() * (new_log_prob.detach() + target_entropy)).mean()
            alpha_optim.zero_grad()
            alpha_loss.backward()
            alpha_optim.step()

            with torch.no_grad():
                for param, targ_param in zip(critic.parameters(), critic_target.parameters()):
                    targ_param.data.copy_(tau * param.data + (1.0 - tau) * targ_param.data)

        if done:
            stats = info["flight_stats"]
            pct = min(1.0, max(0.0, stats['final_x'] / env.arena_length))
            bar = "█" * int(pct * 25) + "-" * (25 - int(pct * 25))
            print(f"[FlyFlight] [{bar}] Flown: {stats['distance_flown']:5.1f}m (X: {stats['final_x']:5.1f}m) | Alt: {stats['final_z']:4.1f}m | Steps: {stats['steps']:4d} | Rew: {episode_reward:5.1f} | {stats['reason']}")
            obs, _ = env.reset()
            episode_reward = 0.0

    torch.save({
        "actor_state_dict": actor.state_dict(),
        "synaptic_gain": actor.synaptic_gain.data.cpu()
    }, "fly_connectome_actor.pt")
    print("[+] Trained model saved to fly_connectome_actor.pt")

if __name__ == "__main__":
    train()