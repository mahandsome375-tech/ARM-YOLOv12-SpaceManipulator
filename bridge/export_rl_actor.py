"""
export_rl_actor.py

Running this script in PyCharm will:
1) Train an off-policy RL Actor in the free-floating 2R space manipulator environment, with parameters from the paper and MATLAB code
2) Export the Actor network and state-normalization parameters to rl_actor.mat
   Structure:
       s.mean[12x1], s.std[12x1]
       net.W1[H x 12], net.b1[H x 1]
       net.W2[H x H ], net.b2[H x 1]
       net.W3[3 x H ], net.b3[3 x 1]
       act.scale[3 x 1], act.bias[3 x 1]
   It can be read normally by policy_infer_rl in MATLAB.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import Tuple
from scipy.io import savemat





STATE_DIM = 12
ACTION_DIM = 3
HIDDEN_DIM = 64

DT = 0.01
GAMMA = 0.99
TAU = 0.005

NUM_EPISODES = 200
MAX_STEPS = 400
BEHAVIOR_NOISE_STD = 0.5

CRITIC_LR = 1e-3
ACTOR_LR = 1e-3
BATCH_SIZE = 256
TRAIN_ITERS = 5000

REPLAY_CAPACITY = 200000


ACTION_MAX = np.array([5.0, 5.0, 5.0], dtype=np.float32)
ACTION_MIN = -ACTION_MAX

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")






class SpaceArmEnv:
    """
    Two-dimensional 2R space manipulator environment in free-floating mode.
    Physical parameters are taken from the paper and MATLAB code.
    """

    def __init__(self, dt: float = DT, episode_steps: int = MAX_STEPS):
        self.dt = dt
        self.max_steps = episode_steps


        self.m0 = 40.0
        self.m1 = 4.0
        self.m2 = 3.0
        self.b0 = 0.5
        self.a1 = 0.5
        self.b1 = 0.5
        self.a2 = 0.5
        self.b2 = 0.5
        self.J0 = 6.667
        self.J1 = 0.333
        self.J2 = 0.25

        self.l1 = self.a1 + self.b1
        self.l2 = self.a2 + self.b2


        M = self.m0 + self.m1 + self.m2
        self.ak1 = (self.b0 * self.m0) / M
        self.ak2 = (self.a1 * self.m0 + self.b1 * (self.m0 + self.m1)) / M
        self.ak3 = (self.a2 * (self.m0 + self.m1) + self.b2 * (self.m0 + self.m1 + self.m2)) / M


        self.ad1 = 3.0
        self.ad2 = 1.0
        self.ad3 = 1.0
        self.ad4 = 20.0
        self.ad5 = 4.0
        self.ad6 = 1.0


        self.q = np.zeros(3, dtype=np.float32)
        self.qdot = np.zeros(3, dtype=np.float32)
        self.step_count = 0
        self.t = 0.0



    def _forward_kinematics(self, q: np.ndarray) -> Tuple[float, float]:
        """
        Use the geometric form from Appendix 2 here:
          xe = -b0*sin(q0) - l1*sin(q0+q1) - l2*sin(q0+q1+q2)
          ye =  b0*cos(q0) + l1*cos(q0+q1) + l2*cos(q0+q1+q2)
        """
        q0, q1, q2 = q
        s0 = np.sin(q0)
        c0 = np.cos(q0)
        s01 = np.sin(q0 + q1)
        c01 = np.cos(q0 + q1)
        s012 = np.sin(q0 + q1 + q2)
        c012 = np.cos(q0 + q1 + q2)

        x = -self.b0 * s0 - self.l1 * s01 - self.l2 * s012
        y = self.b0 * c0 + self.l1 * c01 + self.l2 * c012
        return x, y

    def _jacobian(self, q: np.ndarray) -> np.ndarray:
        q0, q1, q2 = q
        s0 = np.sin(q0)
        c0 = np.cos(q0)
        s01 = np.sin(q0 + q1)
        c01 = np.cos(q0 + q1)
        s012 = np.sin(q0 + q1 + q2)
        c012 = np.cos(q0 + q1 + q2)

        dx_dq0 = -self.b0 * c0 - self.l1 * c01 - self.l2 * c012
        dx_dq1 = -self.l1 * c01 - self.l2 * c012
        dx_dq2 = -self.l2 * c012

        dy_dq0 = -self.b0 * s0 - self.l1 * s01 - self.l2 * s012
        dy_dq1 = -self.l1 * s01 - self.l2 * s012
        dy_dq2 = -self.l2 * s012

        J = np.array([[dx_dq0, dx_dq1, dx_dq2],
                      [dy_dq0, dy_dq1, dy_dq2]], dtype=np.float32)
        return J



    def _calc_M_C(self, q: np.ndarray, qdot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        q0, q1, q2 = q
        q0dot, q1dot, q2dot = qdot

        ad1, ad2, ad3 = self.ad1, self.ad2, self.ad3
        ad4, ad5, ad6 = self.ad4, self.ad5, self.ad6

        Mbb = 2 * ad1 * np.cos(q1) + 2 * ad2 * np.cos(q2) + 2 * ad3 * np.cos(q1 + q2) + ad4
        Mbm = np.array([
            ad1 * np.cos(q1) + 2 * ad2 * np.cos(q2) + ad3 * np.cos(q1 + q2) + ad5,
            ad2 * np.cos(q2) + ad3 * np.cos(q1 + q2) + ad6
        ], dtype=np.float32)
        Mmm = np.array([
            [2 * ad2 * np.cos(q2) + ad5, ad2 * np.cos(q2) + ad6],
            [ad2 * np.cos(q2) + ad6, ad6]
        ], dtype=np.float32)

        M = np.zeros((3, 3), dtype=np.float32)
        M[0, 0] = Mbb
        M[0, 1:] = Mbm
        M[1:, 0] = Mbm
        M[1:, 1:] = Mmm

        Cbb = -ad1 * np.sin(q1) * q1dot - ad2 * np.sin(q2) * q2dot - ad3 * np.sin(q1 + q2) * (q1dot + q2dot)
        Cbm = np.array([
            -ad1 * np.sin(q1) * (q0dot + q1dot) - ad2 * np.sin(q2) * q2dot - ad3 * np.sin(q1 + q2) * (q0dot + q1dot + q2dot),
            -(ad2 * np.sin(q2) + ad3 * np.sin(q1 + q2)) * (q0dot + q1dot + q2dot)
        ], dtype=np.float32)
        Cmb = np.array([
            ad1 * np.sin(q1) * q0dot - ad2 * np.sin(q2) * q2dot + ad3 * np.sin(q1 + q2) * q0dot,
            ad2 * np.sin(q2) * (q0dot + q1dot) + ad3 * np.sin(q1 + q2) * q0dot
        ], dtype=np.float32)
        Cmm = np.array([
            [-ad2 * np.sin(q2) * q2dot, -ad2 * np.sin(q2) * (q0dot + q1dot + q2dot)],
            [ad2 * np.sin(q2) * (q0dot + q1dot), 0.0]
        ], dtype=np.float32)

        C = np.zeros((3, 3), dtype=np.float32)
        C[0, 0] = Cbb
        C[0, 1:] = Cbm
        C[1:, 0] = Cmb
        C[1:, 1:] = Cmm

        return M, C


    def _ref_traj(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        xd = 0.2 + 0.05 * np.sin(3.0 * t)
        yd = 0.4 + 0.05 * np.cos(3.0 * t)

        xdot_d = 0.15 * np.cos(3.0 * t)
        ydot_d = -0.15 * np.sin(3.0 * t)
        return np.array([xd, yd], dtype=np.float32), np.array([xdot_d, ydot_d], dtype=np.float32)


    def reset(self) -> np.ndarray:

        self.q = np.array([0.0, np.pi / 6.0, np.pi / 3.0], dtype=np.float32)
        self.qdot = np.zeros(3, dtype=np.float32)
        self.t = 0.0
        self.step_count = 0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        x, y = self._forward_kinematics(self.q)
        J = self._jacobian(self.q)
        xdot_vec = J @ self.qdot
        xdot, ydot = float(xdot_vec[0]), float(xdot_vec[1])

        xd_vec, _ = self._ref_traj(self.t)
        ex_ey = np.array([x - xd_vec[0], y - xd_vec[1]], dtype=np.float32)

        s = np.array([
            x, y,
            xdot, ydot,
            ex_ey[0], ex_ey[1],
            self.q[0], self.q[1], self.q[2],
            self.qdot[0], self.qdot[1], self.qdot[2]
        ], dtype=np.float32)
        return s

    def step(self, tau: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        tau = np.clip(tau, ACTION_MIN, ACTION_MAX).astype(np.float32)

        M, C = self._calc_M_C(self.q, self.qdot)
        qddot = np.linalg.solve(M, tau - C @ self.qdot)

        self.qdot = self.qdot + qddot * self.dt
        self.q = self.q + self.qdot * self.dt
        self.t += self.dt
        self.step_count += 1

        x, y = self._forward_kinematics(self.q)
        J = self._jacobian(self.q)
        xdot_vec = J @ self.qdot

        xd_vec, xdot_d = self._ref_traj(self.t)
        e_vec = np.array([x - xd_vec[0], y - xd_vec[1]], dtype=np.float32)

        pos_err2 = float(e_vec @ e_vec)
        vel_err2 = float((xdot_vec - xdot_d) @ (xdot_vec - xdot_d))
        torque2 = float(tau @ tau)

        reward = -(10.0 * pos_err2 + 1.0 * vel_err2 + 0.01 * torque2)

        done = False
        if self.step_count >= self.max_steps:
            done = True
        if np.any(np.abs(self.q) > 4 * np.pi):
            done = True

        s_next = self._get_state()
        return s_next, reward, done, {}






class ActorNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        h1 = self.tanh(self.fc1(s))
        h2 = self.tanh(self.fc2(h1))
        a = self.tanh(self.fc3(h2))
        return a


class CriticNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a], dim=-1)
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        q = self.fc3(h2)
        return q






@dataclass
class Transition:
    s: np.ndarray
    a: np.ndarray
    r: float
    s_next: np.ndarray
    done: float


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, tr: Transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(tr)
        else:
            self.buffer[self.pos] = tr
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        s = np.stack([b.s for b in batch], axis=0)
        a = np.stack([b.a for b in batch], axis=0)
        r = np.array([b.r for b in batch], dtype=np.float32)
        s_next = np.stack([b.s_next for b in batch], axis=0)
        done = np.array([b.done for b in batch], dtype=np.float32)
        return s, a, r, s_next, done

    def __len__(self):
        return len(self.buffer)






def collect_offpolicy_data(env: SpaceArmEnv,
                           actor: ActorNet,
                           buffer: ReplayBuffer,
                           num_episodes: int,
                           max_steps: int,
                           noise_std: float):
    actor.eval()
    for ep in range(num_episodes):
        s = env.reset()
        for t in range(max_steps):
            s_tensor = torch.from_numpy(s).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                a_tensor = actor(s_tensor).squeeze(0)
            a = a_tensor.cpu().numpy()
            a = a + np.random.normal(scale=noise_std, size=a.shape)
            a = np.clip(a, -1.0, 1.0)

            tau = ACTION_MIN + (a + 1.0) * 0.5 * (ACTION_MAX - ACTION_MIN)

            s_next, r, done, _ = env.step(tau.astype(np.float32))
            buffer.push(Transition(s=s, a=a.astype(np.float32), r=r,
                                   s_next=s_next, done=float(done)))
            s = s_next
            if done:
                break


def train_offpolicy_integral_rl(env: SpaceArmEnv):
    actor = ActorNet(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)
    critic = CriticNet(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)
    target_actor = ActorNet(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)
    target_critic = CriticNet(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)

    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=ACTOR_LR)
    critic_opt = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    replay = ReplayBuffer(REPLAY_CAPACITY)

    print("Collecting offline data with initial random policy ...")
    collect_offpolicy_data(env, actor, replay,
                           num_episodes=NUM_EPISODES,
                           max_steps=MAX_STEPS,
                           noise_std=BEHAVIOR_NOISE_STD)
    print(f"Collected {len(replay)} transitions.")

    actor.train()
    critic.train()

    for it in range(1, TRAIN_ITERS + 1):
        if len(replay) < BATCH_SIZE:
            continue

        s, a, r, s_next, done = replay.sample(BATCH_SIZE)
        s_tensor = torch.from_numpy(s).float().to(DEVICE)
        a_tensor = torch.from_numpy(a).float().to(DEVICE)
        r_tensor = torch.from_numpy(r).float().unsqueeze(1).to(DEVICE)
        s_next_tensor = torch.from_numpy(s_next).float().to(DEVICE)
        done_tensor = torch.from_numpy(done).float().unsqueeze(1).to(DEVICE)

        with torch.no_grad():
            a_next = target_actor(s_next_tensor)
            q_next = target_critic(s_next_tensor, a_next)
            q_target = r_tensor + GAMMA * (1.0 - done_tensor) * q_next

        q_pred = critic(s_tensor, a_tensor)
        critic_loss = nn.MSELoss()(q_pred, q_target)

        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        actor_opt.zero_grad()
        a_pred = actor(s_tensor)
        q_for_actor = critic(s_tensor, a_pred)
        actor_loss = -q_for_actor.mean()
        actor_loss.backward()
        actor_opt.step()

        with torch.no_grad():
            for p, tp in zip(actor.parameters(), target_actor.parameters()):
                tp.data.mul_(1.0 - TAU).add_(TAU * p.data)
            for p, tp in zip(critic.parameters(), target_critic.parameters()):
                tp.data.mul_(1.0 - TAU).add_(TAU * p.data)

        if it % 500 == 0 or it == TRAIN_ITERS:
            print(f"Iter {it}/{TRAIN_ITERS}, CriticLoss={critic_loss.item():.4e}, "
                  f"ActorLoss={actor_loss.item():.4e}")

    print("Training finished.")
    return actor, replay






def export_actor_to_mat(actor: ActorNet, replay: ReplayBuffer, filename: str = "rl_actor.mat"):
    actor_cpu = ActorNet(STATE_DIM, ACTION_DIM, HIDDEN_DIM)
    actor_cpu.load_state_dict(actor.cpu().state_dict())

    all_states = np.stack([tr.s for tr in replay.buffer], axis=0)
    s_mean = all_states.mean(axis=0).astype(np.float32)
    s_std = all_states.std(axis=0).astype(np.float32) + 1e-6

    W1 = actor_cpu.fc1.weight.detach().numpy().astype(np.float32)
    b1 = actor_cpu.fc1.bias.detach().numpy().reshape(-1, 1).astype(np.float32)
    W2 = actor_cpu.fc2.weight.detach().numpy().astype(np.float32)
    b2 = actor_cpu.fc2.bias.detach().numpy().reshape(-1, 1).astype(np.float32)
    W3 = actor_cpu.fc3.weight.detach().numpy().astype(np.float32)
    b3 = actor_cpu.fc3.bias.detach().numpy().reshape(-1, 1).astype(np.float32)

    act_scale = ACTION_MAX.reshape(-1, 1).astype(np.float32)
    act_bias = np.zeros((ACTION_DIM, 1), dtype=np.float32)

    savemat(filename, {
        "s": {"mean": s_mean.reshape(-1, 1),
              "std": s_std.reshape(-1, 1)},
        "net": {"W1": W1, "b1": b1,
                "W2": W2, "b2": b2,
                "W3": W3, "b3": b3},
        "act": {"scale": act_scale,
                "bias": act_bias}
    })

    print(f"Exported trained actor to {filename}")






def main():
    env = SpaceArmEnv(dt=DT, episode_steps=MAX_STEPS)
    actor, replay = train_offpolicy_integral_rl(env)
    export_actor_to_mat(actor, replay, filename="rl_actor.mat")


if __name__ == "__main__":
    main()
