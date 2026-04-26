













import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from scipy.io import savemat

from hybrid_env import TwoRFreeEnv


SAVE_PATH = os.path.abspath("rl_actor_second.mat")
MAX_EP     = 600
MAX_STEPS  = 1000


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {DEVICE}")


def _to_np1d(x, dim_expected):
    """Convert any observation to a float32 1D vector and align dimensions by truncation or zero padding."""
    x = np.array(x, dtype=np.float32).reshape(-1)
    if x.size != dim_expected:
        if x.size > dim_expected:
            x = x[:dim_expected]
        else:
            x = np.pad(x, (0, dim_expected - x.size))
    return x

class RunningNorm:
    """Accumulate mean/std online using Welford merging for state normalization and export to .mat."""
    def __init__(self, dim, eps=1e-8):
        self.mean = np.zeros((dim,), dtype=np.float64)
        self.var  = np.ones((dim,), dtype=np.float64)
        self.count = eps
    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1: x = x[None, :]
        bm = x.mean(axis=0)
        bv = x.var(axis=0) + 1e-8
        bc = x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        new_mean = self.mean + delta * bc / tot
        m_a = self.var * self.count
        m_b = bv * bc
        M2 = m_a + m_b + delta**2 * self.count * bc / tot
        new_var = M2 / tot
        self.mean, self.var, self.count = new_mean, new_var, tot
    @property
    def std(self): return np.sqrt(self.var + 1e-8)


class Actor(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=3,
                 tau_max=None, log_std=-0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)
        self.tanh = nn.Tanh()

        if tau_max is None:
            tau_max = np.ones((out_dim,), dtype=np.float32)
        self.register_buffer("tau_scale", torch.tensor(tau_max, dtype=torch.float32))
        self.register_buffer("tau_bias",  torch.zeros(out_dim, dtype=torch.float32))
        self.log_std = nn.Parameter(torch.ones(out_dim) * log_std)
    def forward(self, s_norm):
        h = self.tanh(self.fc1(s_norm))
        h = self.tanh(self.fc2(h))
        pre = self.fc3(h)
        a = self.tanh(pre)
        mu  = self.tau_scale * a + self.tau_bias
        std = torch.exp(self.log_std)
        return mu, std, pre
    def act(self, s_norm):
        with torch.no_grad():
            mu, std, _ = self.forward(s_norm)
            dist = torch.distributions.Normal(mu, std)
            a = dist.sample()
            logp = dist.log_prob(a).sum(dim=-1)
        return a, logp

class Critic(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.v   = nn.Linear(hidden, 1)
        self.tanh = nn.Tanh()
    def forward(self, s_norm):
        h = self.tanh(self.fc1(s_norm))
        h = self.tanh(self.fc2(h))
        return self.v(h).squeeze(-1)


@dataclass
class PPOCfg:
    gamma: float = 0.99
    lam:   float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-4
    train_iters: int = 10

class PPOAgent:
    def __init__(self, state_dim, act_dim, cfg: PPOCfg, tau_max=None):
        self.actor  = Actor(state_dim, out_dim=act_dim, tau_max=tau_max).to(DEVICE)
        self.critic = Critic(state_dim).to(DEVICE)
        self.opt_a = optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.opt_c = optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.cfg = cfg

    def compute_gae(self, rewards, values, dones, gamma, lam):
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            mask  = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t+1] * mask - values[t]
            last_gae = delta + gamma * lam * mask * last_gae
            adv[t] = last_gae
        ret = adv + values[:-1]
        return adv, ret

    def update(self, buf, state_norm):
        s, a, logp_old, r, v, done = buf
        s = torch.tensor((s - state_norm.mean)/state_norm.std, dtype=torch.float32, device=DEVICE)
        a = torch.tensor(a, dtype=torch.float32, device=DEVICE)
        logp_old = torch.tensor(logp_old, dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            v_pred = self.critic(s).cpu().numpy()
        values_plus = np.concatenate([v_pred, np.array([0.0], dtype=np.float32)], axis=0)
        adv, ret = self.compute_gae(r, values_plus, done, self.cfg.gamma, self.cfg.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = torch.tensor(adv, dtype=torch.float32, device=DEVICE)
        ret = torch.tensor(ret, dtype=torch.float32, device=DEVICE)

        for _ in range(self.cfg.train_iters):
            mu, std, _ = self.actor(s)
            dist = torch.distributions.Normal(mu, std)
            logp = dist.log_prob(a).sum(dim=-1)
            ratio = torch.exp(logp - logp_old)

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1-self.cfg.clip_eps, 1+self.cfg.clip_eps) * adv
            loss_a = -torch.min(surr1, surr2).mean()

            v_now = self.critic(s)
            loss_c = ((v_now - ret)**2).mean()

            self.opt_a.zero_grad(); self.opt_c.zero_grad()
            (loss_a + 0.5*loss_c).backward()
            self.opt_a.step(); self.opt_c.step()


def train():


    state_dim = 12


    act_dim_guess = 3

    try:
        env = TwoRFreeEnv(state_dim=state_dim, action_dim=act_dim_guess)
    except TypeError:
        env = TwoRFreeEnv()



    if hasattr(env, "state_dim"):
        state_dim = int(env.state_dim)
    elif hasattr(env, "observation_space"):
        shape = getattr(env.observation_space, "shape", (state_dim,))
        state_dim = int(np.prod(shape))


    if hasattr(env, "action_dim"):
        act_dim = int(env.action_dim)
    elif hasattr(env, "tau_max"):
        act_dim = int(np.size(env.tau_max))
    elif hasattr(env, "action_space"):
        shape = getattr(env.action_space, "shape", (act_dim_guess,))
        act_dim = int(np.prod(shape))
    else:
        act_dim = act_dim_guess


    tau_max = None
    if hasattr(env, "tau_max"):
        tm = np.array(env.tau_max, dtype=np.float32).reshape(-1)
        if tm.size == act_dim:
            tau_max = tm
    print(f"[Env] state_dim={state_dim}, action_dim={act_dim}, tau_max={tau_max}")


    cfg = PPOCfg()
    agent = PPOAgent(state_dim, act_dim, cfg, tau_max=tau_max)
    s_norm = RunningNorm(state_dim)


    for ep in range(1, MAX_EP + 1):
        ret = env.reset()
        s = ret[0] if isinstance(ret, (tuple, list)) else ret
        s = _to_np1d(s, state_dim)

        ep_r, done, t = 0.0, False, 0
        buf_s, buf_a, buf_logp, buf_r, buf_done = [], [], [], [], []

        while (not done) and (t < MAX_STEPS):
            s_norm.update(s)
            s_in = torch.tensor((s - s_norm.mean)/s_norm.std, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            a, logp = agent.actor.act(s_in)
            a = a.squeeze(0).cpu().numpy().astype(np.float32).reshape(-1)


            if a.size > act_dim:
                a = a[:act_dim]

            out = env.step(a)

            if isinstance(out, (tuple, list)) and len(out) == 5:
                s2, r, terminated, truncated, info = out
                done = bool(terminated or truncated)
            else:
                s2, r, done, info = out

            s2 = _to_np1d(s2, state_dim)
            buf_s.append(s); buf_a.append(a); buf_logp.append(float(logp.item()))
            buf_r.append(float(r)); buf_done.append(float(done))

            s = s2; ep_r += r; t += 1


        s_batch = np.array(buf_s, dtype=np.float32)
        with torch.no_grad():
            v_batch = agent.critic(torch.tensor((s_batch - s_norm.mean)/s_norm.std,
                                                dtype=torch.float32, device=DEVICE)).cpu().numpy()
        agent.update((
            s_batch,
            np.array(buf_a, dtype=np.float32),
            np.array(buf_logp, dtype=np.float32),
            np.array(buf_r, dtype=np.float32),
            np.array(v_batch, dtype=np.float32),
            np.array(buf_done, dtype=np.float32)
        ), s_norm)

        print(f"[EP {ep:03d}] steps={t:04d}  reward={ep_r:8.3f}")



    W1 = agent.actor.fc1.weight.detach().cpu().numpy()
    b1 = agent.actor.fc1.bias.detach().cpu().numpy().reshape(-1,1)
    W2 = agent.actor.fc2.weight.detach().cpu().numpy()
    b2 = agent.actor.fc2.bias.detach().cpu().numpy().reshape(-1,1)
    W3 = agent.actor.fc3.weight.detach().cpu().numpy()
    b3 = agent.actor.fc3.bias.detach().cpu().numpy().reshape(-1,1)

    act_scale = agent.actor.tau_scale.detach().cpu().numpy().reshape(-1,1)
    act_bias  = agent.actor.tau_bias.detach().cpu().numpy().reshape(-1,1)


    if act_dim == 2:
        H = W3.shape[1]
        W3 = np.vstack([W3, np.zeros((1, H), dtype=np.float32)])
        b3 = np.vstack([b3, np.zeros((1,1), dtype=np.float32)])
        act_scale = np.vstack([act_scale, np.zeros((1,1), dtype=np.float32)])
        act_bias  = np.vstack([act_bias,  np.zeros((1,1), dtype=np.float32)])


    s_mean = s_norm.mean.reshape(-1,1).astype(np.float64)
    s_std  = s_norm.std.reshape(-1,1).astype(np.float64)

    out = {
        "s":   {"mean": s_mean, "std": s_std},
        "net": {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3},
        "act": {"scale": act_scale, "bias": act_bias},

        "s_mean": s_mean, "s_std": s_std,
        "W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3,
        "act_scale": act_scale, "act_bias": act_bias,
    }
    savemat(SAVE_PATH, out)
    print(f"\n✅ Saved to {SAVE_PATH}")
    print("keys:", list(out.keys()))

if __name__ == "__main__":
    train()
