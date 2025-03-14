import numpy as np

class MotionModel:
    def __init__(self):
        self.eps = 1e-6  # Small value for zero comparisons

    def update_particles(self, particles, v, w, dt, motion_noise=None):
        """
        Update particle positions based on velocity commands
        
        Args:
            particles: Nx3 array of particles [x, y, theta]
            v: Linear velocity (m/s)
            w: Angular velocity (rad/s)
            dt: Time step (s)
            motion_noise: Optional [x, y, theta] noise parameters
        
        Returns:
            Updated Nx3 array of particles
        """
        # Make a copy to avoid modifying original array
        updated_particles = particles.copy()
        
        # Convert scalar velocities to arrays if noise is specified
        if motion_noise is not None:
            v = v + np.random.normal(0, motion_noise[0], len(particles))
            w = w + np.random.normal(0, motion_noise[2], len(particles))
        else:
            # Broadcast scalar values to arrays
            v = np.full(len(particles), v)
            w = np.full(len(particles), w)
        
        # Create masks for straight motion and turning
        straight_mask = np.abs(w) < self.eps
        turning_mask = ~straight_mask
        
        # Update straight-moving particles
        if np.any(straight_mask):
            updated_particles[straight_mask, 0] += v[straight_mask] * \
                np.cos(particles[straight_mask, 2]) * dt
            updated_particles[straight_mask, 1] += v[straight_mask] * \
                np.sin(particles[straight_mask, 2]) * dt
        
        # Update turning particles
        if np.any(turning_mask):
            v_w = np.divide(v[turning_mask], w[turning_mask])
            theta = particles[turning_mask, 2]
            theta_new = theta + w[turning_mask] * dt
            
            updated_particles[turning_mask, 0] += v_w * \
                (np.sin(theta_new) - np.sin(theta))
            updated_particles[turning_mask, 1] += v_w * \
                (-np.cos(theta_new) + np.cos(theta))
            updated_particles[turning_mask, 2] = theta_new
        
        # Normalize angles to [-pi, pi]
        updated_particles[:, 2] = np.arctan2(np.sin(updated_particles[:, 2]), 
                                           np.cos(updated_particles[:, 2]))
        
        return updated_particles

    def get_motion_stats(self, particles, initial_poses):
        """
        Calculate motion statistics for debugging
        
        Args:
            particles: Current Nx3 array of particles
            initial_poses: Initial Nx3 array of particles
            
        Returns:
            Dictionary with motion statistics
        """
        dx = particles[:, 0] - initial_poses[:, 0]
        dy = particles[:, 1] - initial_poses[:, 1]
        dtheta = particles[:, 2] - initial_poses[:, 2]
        
        # Normalize angle differences to [-pi, pi]
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
        
        return {
            'mean_dx': np.mean(dx),
            'mean_dy': np.mean(dy),
            'mean_dtheta': np.mean(dtheta),
            'std_dx': np.std(dx),
            'std_dy': np.std(dy),
            'std_dtheta': np.std(dtheta)
        }