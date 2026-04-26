






import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import savemat

from hybrid_env import TwoRFreeEnv




DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[Device]", DEVICE)






TRAJ_PATH = r"bridge\station_traj_fourmods.npy"
traj = np.load(TRAJ_PATH)
_, cx, cy, d, tau_v, phi, sigma = traj.T

s_vis = np.stack([d, tau_v, phi, sigma], axis=1).astype(np.float32)







class Actor(nn.Module):
    def __init__(self, in_dim=16, hidden=128, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
            nn.Tanh()
        )

        self.tau_max = torch.tensor([20.0, 20.0, 20.0], dtype=torch.float32, device=DEVICE)

    def forward(self, s):

        return self.tau_max * self.net(s)


class Critic(nn.Module):
    def __init__(self, in_dim=16, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, s):
        return self.net(s)





class PPOAgent:
    def __init__(self, in_dim, act_dim, lr=3e-4, gamma=0.99, lam=0.95, clip_eps=0.2):
        self.actor = Actor(in_dim=in_dim, out_dim=act_dim).to(DEVICE)
        self.critic = Critic(in_dim=in_dim).to(DEVICE)

        self.opt_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps

    def get_action(self, s):
        s = torch.tensor(s, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            a = self.actor(s)
        return a.cpu().numpy()

    def update(self, states, actions, rewards, dones):
        states_t  = torch.tensor(states,  dtype=torch.float32, device=DEVICE)
        actions_t = torch.tensor(actions, dtype=torch.float32, device=DEVICE)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        dones_t   = torch.tensor(dones,   dtype=torch.float32, device=DEVICE)


        with torch.no_grad():
            values = self.critic(states_t).squeeze()


            a_pred = self.actor(states_t)
            logp_old = -torch.sum((a_pred - actions_t) ** 2, dim=1)


        adv = []
        gae = 0.0
        values_ext = torch.cat([values, torch.tensor([0.0], device=DEVICE)])
        for t in reversed(range(len(rewards))):
            delta = rewards_t[t] + self.gamma * values_ext[t + 1] * (1 - dones_t[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones_t[t]) * gae
            adv.insert(0, gae)
        adv = torch.stack(adv)
        ret = adv + values

        adv = (adv - adv.mean()) / (adv.std() + 1e-5)


        K_epochs = 10
        for _ in range(K_epochs):

            a_pred = self.actor(states_t)
            logp = -torch.sum((a_pred - actions_t) ** 2, dim=1)

            ratio = torch.exp(logp - logp_old)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
            actor_loss = -torch.mean(torch.min(surr1, surr2))

            v_pred = self.critic(states_t).squeeze()
            critic_loss = torch.mean((v_pred - ret) ** 2)

            loss = actor_loss + 0.5 * critic_loss

            self.opt_actor.zero_grad()
            self.opt_critic.zero_grad()
            loss.backward()
            self.opt_actor.step()
            self.opt_critic.step()











def export_matlab_actor(actor, save_path):
    sd = actor.state_dict()


    W1 = sd['net.0.weight'].cpu().numpy()
    b1 = sd['net.0.bias'  ].cpu().numpy()

    W2 = sd['net.2.weight'].cpu().numpy()
    b2 = sd['net.2.bias'  ].cpu().numpy()

    W3 = sd['net.4.weight'].cpu().numpy()
    b3 = sd['net.4.bias'  ].cpu().numpy()

    D = W1.shape[1]
    out_dim = W3.shape[0]


    s = {
        'mean': np.zeros((D, 1), dtype=np.float32),
        'std':  np.ones((D, 1),  dtype=np.float32),
    }


    net = {
        'W1': W1,
        'b1': b1.reshape(-1, 1),
        'W2': W2,
        'b2': b2.reshape(-1, 1),
        'W3': W3,
        'b3': b3.reshape(-1, 1),
    }


    act = {
        'scale': np.ones((out_dim, 1), dtype=np.float32),
        'bias':  np.zeros((out_dim, 1), dtype=np.float32),
    }

    savemat(save_path, {'s': s, 'net': net, 'act': act})
    print(f"\n[Export] Saved MATLAB actor to: {save_path}")





def train():
    env = TwoRFreeEnv()
    in_dim = 12 + 4
    act_dim = 3

    agent = PPOAgent(in_dim=in_dim, act_dim=act_dim)

    MAX_EP = 300
    global_step = 0

    for ep in range(MAX_EP):
        s_env = env.reset()
        done = False

        ep_states = []
        ep_actions = []
        ep_rewards = []
        ep_dones = []

        vis_id = 0

        while not done:

            fvis = s_vis[min(vis_id, len(s_vis) - 1)]
            vis_id += 1


            s_input = np.concatenate([s_env, fvis], dtype=np.float32)

            a = agent.get_action(s_input)

            s_env2, r, done, info = env.step(a)

            ep_states.append(s_input)
            ep_actions.append(a)
            ep_rewards.append(r)
            ep_dones.append(done)

            s_env = s_env2
            global_step += 1

        agent.update(ep_states, ep_actions, ep_rewards, ep_dones)

        ep_return = np.sum(ep_rewards)
        print(f"[EP {ep+1:03d}] return = {ep_return:8.3f}, steps = {len(ep_rewards)}")


    save_path = r"bridge\rl_actor.mat"
    export_matlab_actor(agent.actor, save_path)





if __name__ == "__main__":
    train()
