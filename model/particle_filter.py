"""
Particle filter for ground truth enhancement
"""

import numpy as np
import torch


class ParticleFilterGT:
    # particle filter for gt enhancement during training
    def __init__(self, num_particles=200, device='cpu'):
        self.num_particles = num_particles
        self.device = device
        
        self.particles = None  
        self.weights = None    
        self.initialized = False
        
        self.velocity_std = 3.0
        self.position_std = 2.0
        self.size_std = 1.5
        self.velocity_decay = 0.92 
        
        # image and object constraints
        self.max_velocity = 30.0
        self.min_size = 5.0
        self.max_size = 120.0
        self.image_width = 304
        self.image_height = 240
        

    # initialize particles around first valid gt observation
    def initialize_particles(self, center, size):
        if center is None or size is None:
            return False
            
        # Convert to numpy
        x, y = center[0].item(), center[1].item()
        w, h = size[0].item(), size[1].item()
        
        self.particles = np.zeros((self.num_particles, 6))
        
        self.particles[:, 0] = np.random.normal(x, 2.0, self.num_particles)
        self.particles[:, 1] = np.random.normal(y, 2.0, self.num_particles)
        
        self.particles[:, 2] = np.random.normal(0, 3.0, self.num_particles)
        self.particles[:, 3] = np.random.normal(0, 3.0, self.num_particles)
        
        self.particles[:, 4] = np.random.normal(w, 1.0, self.num_particles)
        self.particles[:, 5] = np.random.normal(h, 1.0, self.num_particles)
        
        self._apply_constraints()
        
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.initialized = True
        
        return True
        

    # predict particle motion and add noise
    def motion_model(self, dt=0.005):
        if not self.initialized:
            return
            
        # x = x + v * dt
        self.particles[:, 0] += self.particles[:, 2] * dt
        self.particles[:, 1] += self.particles[:, 3] * dt
        
        # velocity damping
        self.particles[:, 2] *= self.velocity_decay
        self.particles[:, 3] *= self.velocity_decay
        
        noise = np.random.normal(0, 1, (self.num_particles, 6))
        noise[:, 0] *= self.position_std  # x noise
        noise[:, 1] *= self.position_std  # y noise
        noise[:, 2] *= self.velocity_std  # vx noise
        noise[:, 3] *= self.velocity_std  # vy noise
        noise[:, 4] *= self.size_std      # width noise
        noise[:, 5] *= self.size_std      # height noise
        
        self.particles += noise
        
        self._apply_constraints()
        

    # enforce physical bounds on particles
    def _apply_constraints(self):
        self.particles[:, 0] = np.clip(self.particles[:, 0], 10, self.image_width - 10)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 10, self.image_height - 10)
        
        # velocity bounds
        self.particles[:, 2] = np.clip(self.particles[:, 2], -self.max_velocity, self.max_velocity)
        self.particles[:, 3] = np.clip(self.particles[:, 3], -self.max_velocity, self.max_velocity)
        
        # size bounds
        self.particles[:, 4] = np.clip(self.particles[:, 4], self.min_size, self.max_size)
        self.particles[:, 5] = np.clip(self.particles[:, 5], self.min_size, self.max_size)
        

    # update particle weights based on observation
    def update_with_observation(self, center, size):
        if not self.initialized or center is None or size is None:
            return
            
        x_obs, y_obs = center[0].item(), center[1].item()
        w_obs, h_obs = size[0].item(), size[1].item()
        
        # likelihood cal per particle
        for i in range(self.num_particles):
            px, py, _, _, pw, ph = self.particles[i]
            

            pos_dist = np.sqrt((px - x_obs)**2 + (py - y_obs)**2)
            pos_likelihood = np.exp(-0.5 * (pos_dist**2) / (8.0**2))
            
            size_dist = np.sqrt((pw - w_obs)**2 + (ph - h_obs)**2)
            size_likelihood = np.exp(-0.5 * (size_dist**2) / (5.0**2))
            
            self.weights[i] *= pos_likelihood * size_likelihood
        
        weight_sum = np.sum(self.weights)
        if weight_sum > 1e-10:
            self.weights /= weight_sum
        else:
            self.weights = np.ones(self.num_particles) / self.num_particles
            
        eff_size = 1.0 / np.sum(self.weights**2)
        if eff_size < self.num_particles / 3:
            self._resample()
            

    # resample particles based on weights
    def _resample(self):
        indices = np.random.choice(
            self.num_particles, 
            self.num_particles, 
            p=self.weights
        )
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles
        

    # get weighted average prediction from particles
    def get_enhanced_prediction(self):
        if not self.initialized:
            return None, None, 0.0
            
        pred_x = np.average(self.particles[:, 0], weights=self.weights)
        pred_y = np.average(self.particles[:, 1], weights=self.weights)
        pred_w = np.average(self.particles[:, 4], weights=self.weights)
        pred_h = np.average(self.particles[:, 5], weights=self.weights)
        
        var_x = np.average((self.particles[:, 0] - pred_x)**2, weights=self.weights)
        var_y = np.average((self.particles[:, 1] - pred_y)**2, weights=self.weights)
        spread = np.sqrt(var_x + var_y)
        
        confidence = 1.0 / (1.0 + spread / 10.0)
        
        enhanced_center = torch.tensor([pred_x, pred_y], dtype=torch.float32, device=self.device)
        enhanced_size = torch.tensor([pred_w, pred_h], dtype=torch.float32, device=self.device)
        
        return enhanced_center, enhanced_size, confidence
    
    
    def enhance_batch_with_particle_filter(center_seq, bbox_seq, confidence_seq, has_plane_seq):
        batch_size, seq_len = center_seq.shape[:2]
        device = center_seq.device
        
        enhanced_centers = []
        enhanced_bboxes = []
        enhanced_confidences = []
        
        # process each sequence in the batch
        for b in range(batch_size):
            pf = ParticleFilterGT(num_particles=100, device=device)
            
            seq_centers = []
            seq_bboxes = []
            seq_confidences = []
            
            # process each timestep in the sequence
            for t in range(seq_len):
                has_plane = has_plane_seq[b, t].item()
                original_center = center_seq[b, t]
                original_bbox = bbox_seq[b, t]
                original_conf = confidence_seq[b, t].item()
                
                bbox_w = original_bbox[2] - original_bbox[0]
                bbox_h = original_bbox[3] - original_bbox[1]
                original_size = torch.tensor([bbox_w, bbox_h], dtype=torch.float32, device=device)
                
                # enhance only valid plane detections
                if has_plane and not (original_center[0].item() == 0 and original_center[1].item() == 0):
                    if not pf.initialized:
                        pf.initialize_particles(original_center, original_size)
                        enhanced_center = original_center
                        enhanced_size = original_size
                        enhanced_conf = original_conf
                    else:
                        pf.motion_model(dt=0.005)
                        
                        pf.update_with_observation(original_center, original_size)
                        
                        pf_center, pf_size, pf_conf = pf.get_enhanced_prediction()
                        
                        if pf_center is not None and pf_size is not None:
                            enhanced_center = original_center
                            
                            size_blend_weight = 0.8 + 0.15 * original_conf
                            
                            enhanced_size = torch.zeros_like(original_size)
                            enhanced_size[0] = size_blend_weight * original_size[0] + (1-size_blend_weight) * pf_size[0]
                            enhanced_size[1] = size_blend_weight * original_size[1] + (1-size_blend_weight) * pf_size[1]
                            
                            enhanced_conf = (original_conf * 3.0 + pf_conf) / 4.0
                        else:
                            enhanced_center = original_center
                            enhanced_size = original_size
                            enhanced_conf = original_conf
                    
                    cx, cy = enhanced_center[0], enhanced_center[1]
                    w, h = enhanced_size[0], enhanced_size[1]
                    
                    enhanced_bbox = torch.tensor([
                        cx - w/2, cy - h/2, cx + w/2, cy + h/2
                    ], dtype=torch.float32, device=device)
                    
                else:
                    enhanced_center = original_center
                    enhanced_bbox = original_bbox
                    enhanced_conf = original_conf
                        
                
                seq_centers.append(enhanced_center)
                seq_bboxes.append(enhanced_bbox)
                seq_confidences.append(enhanced_conf)
            
            enhanced_centers.append(torch.stack(seq_centers))
            enhanced_bboxes.append(torch.stack(seq_bboxes))
            enhanced_confidences.append(torch.tensor(seq_confidences, dtype=torch.float32, device=device))
        
        enhanced_centers = torch.stack(enhanced_centers)
        enhanced_bboxes = torch.stack(enhanced_bboxes)
        enhanced_confidences = torch.stack(enhanced_confidences)
        
        return enhanced_centers, enhanced_bboxes, enhanced_confidences