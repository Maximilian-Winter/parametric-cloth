"""A minimal pure-NumPy Adam optimizer.

Every loss in this package ships an analytic gradient, so there's no need for
a full autodiff framework's optimizer -- this is the standard Adam update rule
(Kingma & Ba, 2015) operating on an arbitrary-shaped parameter array.
"""

from __future__ import annotations

import numpy as np


class Adam:
    def __init__(self, shape, lr: float = 0.01, beta1: float = 0.9,
                beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
