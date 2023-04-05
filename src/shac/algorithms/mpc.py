import numpy as np
from typing import Optional
from scipy.interpolate import CubicSpline


class Policy:
    def __init__(
        self,
        num_actions: int,
        horizon: float = 0.5,
        dt: float = 1.0 / 60.0,
        max_steps: int = 512,
        params: Optional[np.ndarray] = None,
        policy_type: str = "linear",
        step: float = 0.0,
    ):
        self.num_actions = num_actions
        self.horizon = horizon
        self.step = step
        self.dt = dt
        self.max_steps = max_steps

        # Spline points
        steps = int(min(horizon / dt + 1, max_steps))
        self.timesteps = np.linspace(0, horizon, steps)
        self.params = params or np.zeros((steps, num_actions))
        self.policy_type = policy_type

    def get_policy(self):
        pol = None
        if self.policy_type == "cubic":
            pol = lambda x, params: CubicSpline(self.timesteps, params)(x)
        elif self.policy_type == "zero":
            pol = lambda x, params: params[
                :, np.argwhere(x > self.timesteps)[-1].item()
            ]
        else:
            assert self.policy_type == "linear"
            pol = lambda x, params: np.stack(
                [
                    np.interp(x, self.timesteps, params[:, i])
                    for i in range(self.num_actions)
                ]
            )
        return pol

    def action(self, t, params=None):
        params = self.params if params is None else params
        return self.get_policy()(t, params)
