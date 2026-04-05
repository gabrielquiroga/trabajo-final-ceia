import numpy as np

class TrajectoryGenerator:
    """Class with static methods to generate diverse planar trajectories"""

    @staticmethod
    def linear(n_points, start= (0,0), angle=0, length=1.0):
        t = np.linspace(0, length, n_points)
        x = start[0] + t * np.cos(np.radians(angle))
        y = start[1] + t * np.sin(np.radians(angle))
        return np.stack([x, y], axis=1)

    @staticmethod
    def sinusoidal(n_points, amp=0.5, freq=1.0, phase=0.0):
        x = np.linspace(0, 2, n_points)
        y = amp * np.sin(2 * np.pi * freq * x + phase)
        return np.stack([x, y], axis=1)

    @staticmethod
    def circular(n_points, radius=1.0, arc_ratio=1.0, start_angle=0.0):
        angles = np.linspace(0, 2 * np.pi * arc_ratio, n_points) + np.radians(start_angle)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        return np.stack([x, y], axis=1)
    
    @staticmethod
    def parabolic(n_points, a=1.0, h=0, k=0, x_range=(-1, 1)):
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = a * (x - h)**2 + k
        return np.stack([x, y], axis=1)

    @staticmethod
    def exponential(n_points, a=1.0, b=1.0, x_range=(0, 2)):
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = a * np.exp(b * x)
        return np.stack([x, y], axis=1)

    @staticmethod
    def sigmoid(n_points, L=1.0, k=10, x0=1.0):
        x = np.linspace(0, 2, n_points)
        y = L / (1 + np.exp(-k * (x - x0)))
        return np.stack([x, y], axis=1)

    @staticmethod
    def staggered_step(n_points, n_steps=3, step_height=0.5):
        x = np.linspace(0, 2, n_points)
        # Crear escalones usando redondeo y multiplicación
        y = np.floor(x * (n_steps / 2)) * step_height
        return np.stack([x, y], axis=1)

    @staticmethod
    def spiral(n_points, b=0.1, theta_max=4*np.pi):
        theta = np.linspace(0, theta_max, n_points)
        r = b * theta
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.stack([x, y], axis=1)

    @staticmethod
    def lemniscate(n_points, a=1.5):
        # Figura en 8 (Lemniscata de Bernoulli)
        t = np.linspace(0, 2 * np.pi, n_points)
        x = (a * np.cos(t)) / (1 + np.sin(t)**2)
        y = (a * np.sin(t) * np.cos(t)) / (1 + np.sin(t)**2)
        return np.stack([x, y], axis=1)

    @staticmethod
    def lissajous(n_points, A=1, B=1, a=3, b=2, delta=np.pi/2):
        t = np.linspace(0, 2 * np.pi, n_points)
        x = A * np.sin(a * t + delta)
        y = B * np.sin(b * t)
        return np.stack([x, y], axis=1)