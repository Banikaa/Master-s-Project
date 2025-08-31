import torch
import torch.nn as nn
import torch.optim as optim
import snntorch as snn
from snntorch import surrogate
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import os

from particle_filter import ParticleFilterGT
from model import SNN_tracker_model
from dataset import get_dataloader


# trainer class with particle filter enhanced ground truth
class ParticleFilterTrainer:
    def __init__(self, model, train_loader, val_loader, device='cpu', use_particle_filter=True):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_particle_filter = use_particle_filter
        
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        
        self.detection_loss = nn.BCELoss(reduction='none')
        self.center_loss = nn.MSELoss(reduction='none')
        self.size_loss = nn.MSELoss(reduction='none')
        self.velocity_loss = nn.MSELoss(reduction='none')
        self.iou_loss = self.iou_loss
        
        self.train_losses = []
        self.val_losses = []
        self.detailed_train_losses = []
        self.detailed_val_losses = []
        self.best_val_loss = float('inf')
    
    # one epoch training
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        detection_loss_total = 0
        center_loss_total = 0
        size_loss_total = 0
        velocity_loss_total = 0
        iou_loss_total = 0
        prediction_errors = []
        
        pf_enhancement_count = 0
        pf_confidence_scores = []
        original_vs_enhanced_errors = []
        total_frames_processed = 0
        
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Training Epoch {epoch}")):
            spike_sequence = batch['spike_sequence'].to(self.device)
            has_plane_seq = batch['has_plane_sequence'].float().to(self.device)
            center_seq = batch['center_sequence'].to(self.device)
            bbox_seq = batch['bbox_sequence'].to(self.device)
            confidence_seq = batch['confidence_sequence'].to(self.device)
            
            original_center_seq = center_seq.clone()
            original_bbox_seq = bbox_seq.clone()
            original_confidence_seq = confidence_seq.clone()
            
            # enhance ground truth with particle filter if enabled
            if self.use_particle_filter:
                center_seq, bbox_seq, confidence_seq = ParticleFilterGT.enhance_batch_with_particle_filter(
                    center_seq, bbox_seq, confidence_seq, has_plane_seq
                )
                
                # track enhancement statistics too see how many annotations the particle filter will change
                with torch.no_grad():
                    bbox_changes = torch.norm(bbox_seq.view(bbox_seq.shape[0], bbox_seq.shape[1], -1) - 
                                            original_bbox_seq.view(original_bbox_seq.shape[0], original_bbox_seq.shape[1], -1), dim=-1)
                    enhancement_mask = bbox_changes > 1e-4
                    pf_enhancement_count += enhancement_mask.sum().item()
                    total_frames_processed += bbox_seq.shape[0] * bbox_seq.shape[1]
                    
                    conf_diff = confidence_seq - original_confidence_seq
                    positive_improvements = conf_diff[conf_diff > 0]
                    if len(positive_improvements) > 0:
                        pf_confidence_scores.extend(positive_improvements.cpu().numpy())
                    
                    for b in range(bbox_seq.shape[0]):
                        if bbox_seq.shape[1] > 2:
                            orig_sizes = torch.stack([
                                original_bbox_seq[b, :, 2] - original_bbox_seq[b, :, 0],
                                original_bbox_seq[b, :, 3] - original_bbox_seq[b, :, 1]
                            ], dim=-1)
                            
                            enh_sizes = torch.stack([
                                bbox_seq[b, :, 2] - bbox_seq[b, :, 0],
                                bbox_seq[b, :, 3] - bbox_seq[b, :, 1]
                            ], dim=-1)
                            
                            orig_size_vel = orig_sizes[1:] - orig_sizes[:-1]
                            orig_size_accel = orig_size_vel[1:] - orig_size_vel[:-1]
                            orig_smoothness = torch.norm(orig_size_accel, dim=-1).var().item() if len(orig_size_accel) > 0 else 0
                            
                            enh_size_vel = enh_sizes[1:] - enh_sizes[:-1]
                            enh_size_accel = enh_size_vel[1:] - enh_size_vel[:-1]
                            enh_smoothness = torch.norm(enh_size_accel, dim=-1).var().item() if len(enh_size_accel) > 0 else 0
                            
                            if orig_smoothness > 0:
                                smoothness_improvement = (orig_smoothness - enh_smoothness) / orig_smoothness
                                original_vs_enhanced_errors.append(smoothness_improvement)

            
            center_seq_normalized = center_seq.clone()
            center_seq_normalized[:, :, 0] /= self.model.width
            center_seq_normalized[:, :, 1] /= self.model.height
            
            bbox_seq_normalized = bbox_seq.clone()
            bbox_seq_normalized[:, :, 0] /= self.model.width
            bbox_seq_normalized[:, :, 1] /= self.model.height
            bbox_seq_normalized[:, :, 2] /= self.model.width
            bbox_seq_normalized[:, :, 3] /= self.model.height
            
            gt_width = bbox_seq_normalized[:, :, 2] - bbox_seq_normalized[:, :, 0]
            gt_height = bbox_seq_normalized[:, :, 3] - bbox_seq_normalized[:, :, 1]
            gt_size = torch.stack([gt_width, gt_height], dim=-1)
            
            gt_velocity = torch.zeros_like(center_seq_normalized)
            gt_velocity[:, 1:] = center_seq_normalized[:, 1:] - center_seq_normalized[:, :-1]
            
            velocity_magnitudes = torch.norm(gt_velocity, dim=-1)
            valid_velocities = velocity_magnitudes[velocity_magnitudes > 1e-6]
            
            if len(valid_velocities) > 0:
                max_vel_scale = max(min(torch.quantile(valid_velocities, 0.95).item(), 0.05), 0.001)
            else:
                max_vel_scale = 0.01
            
            gt_velocity_scaled = torch.clamp(gt_velocity / max_vel_scale, -1.0, 1.0)
            
            self.optimizer.zero_grad()
            detection_pred, center_pred, size_pred, velocity_pred, bbox_pred = self.model(spike_sequence)
            
            seq_len = detection_pred.shape[1]
            time_weights = torch.linspace(0.5, 1.0, seq_len).to(self.device)
            
            loss_dict = self._calculate_losses(
                detection_pred, center_pred, size_pred, velocity_pred,
                has_plane_seq, center_seq_normalized, gt_size, gt_velocity_scaled,
                confidence_seq, time_weights
            )
            
            batch_loss = loss_dict['total_loss']
            
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += batch_loss.item()
            detection_loss_total += loss_dict['detection_loss'].item()
            center_loss_total += loss_dict['center_loss'].item()
            size_loss_total += loss_dict['size_loss'].item()
            velocity_loss_total += loss_dict['velocity_loss'].item()
            iou_loss_total += loss_dict['iou_loss'].item()
            
            with torch.no_grad():
                for t in range(seq_len - 1):
                    plane_mask = has_plane_seq[:, t].bool()
                    if plane_mask.sum() > 0:
                        current_center = center_pred[:, t][plane_mask]
                        pred_velocity = velocity_pred[:, t][plane_mask]
                        predicted_next = current_center + pred_velocity * 0.005
                        
                        actual_next = center_pred[:, t+1][plane_mask]
                        
                        pred_error = torch.norm(predicted_next - actual_next, dim=1).mean().item()
                        pred_error_pixels = pred_error * self.model.width
                        prediction_errors.append(pred_error_pixels)
            
            if batch_idx % 50 == 0:
                avg_pred_error = np.mean(prediction_errors) if prediction_errors else 0
                avg_pf_conf = np.mean(pf_confidence_scores) if pf_confidence_scores else 0
                avg_smoothness_improvement = np.mean(original_vs_enhanced_errors) if original_vs_enhanced_errors else 0
                enhancement_rate = (pf_enhancement_count / max(1, total_frames_processed)) * 100
                
                print(f"\nBatch {batch_idx}/{num_batches}:")
                print(f"  Total Loss: {batch_loss.item():.4f}")
                print(f"  Detection: {loss_dict['detection_loss'].item():.4f}")
                print(f"  Center: {loss_dict['center_loss'].item():.4f}")
                print(f"  Size: {loss_dict['size_loss'].item():.4f}")
                print(f"  Velocity: {loss_dict['velocity_loss'].item():.4f}")
                print(f"  IoU: {loss_dict['iou_loss'].item():.4f}")
                print(f"  Pred Error: {avg_pred_error:.2f} pixels")
                if self.use_particle_filter:
                    print(f"  PF Enhanced: {enhancement_rate:.1f}% of frames")
                    print(f"  PF Confidence: +{avg_pf_conf:.3f}")
                    print(f"  PF Smoothness: {avg_smoothness_improvement:.3f}")
        
        pf_enhancement_rate = (pf_enhancement_count / max(1, total_frames_processed)) * 100 if self.use_particle_filter and total_frames_processed > 0 else 0
        pf_confidence_improvement = np.mean(pf_confidence_scores) if pf_confidence_scores else 0
        pf_smoothness_improvement = np.mean(original_vs_enhanced_errors) if original_vs_enhanced_errors else 0
        
        avg_losses = {
            'total_loss': total_loss / num_batches,
            'detection_loss': detection_loss_total / num_batches,
            'center_loss': center_loss_total / num_batches,
            'size_loss': size_loss_total / num_batches,
            'velocity_loss': velocity_loss_total / num_batches,
            'iou_loss': iou_loss_total / num_batches,
            'prediction_error': np.mean(prediction_errors) if prediction_errors else 0,
            'pf_enhancement_rate': pf_enhancement_rate,
            'pf_confidence_improvement': pf_confidence_improvement,
            'pf_smoothness_improvement': pf_smoothness_improvement
        }
        return avg_losses
    

    # validate model performance
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        detection_loss_total = 0
        center_loss_total = 0
        size_loss_total = 0
        velocity_loss_total = 0
        iou_loss_total = 0
        prediction_errors = []
        
        pf_enhancement_count = 0
        pf_confidence_scores = []
        original_vs_enhanced_errors = []
        total_frames_processed = 0
        
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Validation Epoch {epoch}"):
                spike_sequence = batch['spike_sequence'].to(self.device)
                has_plane_seq = batch['has_plane_sequence'].float().to(self.device)
                center_seq = batch['center_sequence'].to(self.device)
                bbox_seq = batch['bbox_sequence'].to(self.device)
                confidence_seq = batch['confidence_sequence'].to(self.device)
                
                total_frames_processed += center_seq.numel() // 2
                
                center_seq_normalized = center_seq.clone()
                center_seq_normalized[:, :, 0] /= self.model.width
                center_seq_normalized[:, :, 1] /= self.model.height
                
                bbox_seq_normalized = bbox_seq.clone()
                bbox_seq_normalized[:, :, 0] /= self.model.width
                bbox_seq_normalized[:, :, 1] /= self.model.height
                bbox_seq_normalized[:, :, 2] /= self.model.width
                bbox_seq_normalized[:, :, 3] /= self.model.height
                
                gt_width = bbox_seq_normalized[:, :, 2] - bbox_seq_normalized[:, :, 0]
                gt_height = bbox_seq_normalized[:, :, 3] - bbox_seq_normalized[:, :, 1]
                gt_size = torch.stack([gt_width, gt_height], dim=-1)
                
                gt_velocity = torch.zeros_like(center_seq_normalized)
                gt_velocity[:, 1:] = center_seq_normalized[:, 1:] - center_seq_normalized[:, :-1]
                
                velocity_magnitudes = torch.norm(gt_velocity, dim=-1)
                valid_velocities = velocity_magnitudes[velocity_magnitudes > 1e-6]
                
                if len(valid_velocities) > 0:
                    max_vel_scale = max(min(torch.quantile(valid_velocities, 0.95).item(), 0.05), 0.001)
                else:
                    max_vel_scale = 0.01
                
                gt_velocity_scaled = torch.clamp(gt_velocity / max_vel_scale, -1.0, 1.0)
                
                detection_pred, center_pred, size_pred, velocity_pred, bbox_pred = self.model(spike_sequence)
                
                seq_len = detection_pred.shape[1]
                time_weights = torch.linspace(0.5, 1.0, seq_len).to(self.device)
                
                loss_dict = self._calculate_losses(
                    detection_pred, center_pred, size_pred, velocity_pred,
                    has_plane_seq, center_seq_normalized, gt_size, gt_velocity_scaled,
                    confidence_seq, time_weights
                )
                
                total_loss += loss_dict['total_loss'].item()
                detection_loss_total += loss_dict['detection_loss'].item()
                center_loss_total += loss_dict['center_loss'].item()
                size_loss_total += loss_dict['size_loss'].item()
                velocity_loss_total += loss_dict['velocity_loss'].item()
                iou_loss_total += loss_dict['iou_loss'].item()
                
                for t in range(seq_len - 1):
                    plane_mask = has_plane_seq[:, t].bool()
                    if plane_mask.sum() > 0:
                        current_center = center_pred[:, t][plane_mask]
                        pred_velocity = velocity_pred[:, t][plane_mask]
                        predicted_next = current_center + pred_velocity * 0.005
                        actual_next = center_pred[:, t+1][plane_mask]
                        pred_error = torch.norm(predicted_next - actual_next, dim=1).mean().item()
                        pred_error_pixels = pred_error * self.model.width
                        prediction_errors.append(pred_error_pixels)
        
        pf_enhancement_rate = (pf_enhancement_count / max(1, total_frames_processed)) * 100 if self.use_particle_filter and total_frames_processed > 0 else 0
        pf_confidence_improvement = np.mean(pf_confidence_scores) if pf_confidence_scores else 0
        pf_smoothness_improvement = np.mean(original_vs_enhanced_errors) if original_vs_enhanced_errors else 0
        
        avg_losses = {
            'total_loss': total_loss / num_batches,
            'detection_loss': detection_loss_total / num_batches,
            'center_loss': center_loss_total / num_batches,
            'size_loss': size_loss_total / num_batches,
            'velocity_loss': velocity_loss_total / num_batches,
            'iou_loss': iou_loss_total / num_batches,
            'prediction_error': np.mean(prediction_errors) if prediction_errors else 0,
            'pf_enhancement_rate': pf_enhancement_rate,
            'pf_confidence_improvement': pf_confidence_improvement,
            'pf_smoothness_improvement': pf_smoothness_improvement
        }
        return avg_losses
    

    # calculate all loss components for training
    def _calculate_losses(self, detection_pred, center_pred, size_pred, velocity_pred,
                         has_plane_seq, center_gt, size_gt, velocity_gt,
                         confidence_seq, time_weights):
        seq_len = detection_pred.shape[1]
        
        # calculate detection loss for all timesteps
        detection_losses = []
        for t in range(seq_len):
            det_loss = self.detection_loss(
                detection_pred[:, t].squeeze(-1),
                has_plane_seq[:, t]
            )
            detection_losses.append(det_loss.mean())
        detection_loss = torch.stack(detection_losses).mean()
        
        # calculate center loss only for frames with planes
        center_losses = []
        for t in range(seq_len):
            plane_mask = has_plane_seq[:, t].bool()
            if plane_mask.sum() > 0:
                center_loss_t = self.center_loss(
                    center_pred[:, t][plane_mask],
                    center_gt[:, t][plane_mask]
                ).mean()
                center_losses.append(center_loss_t)
        center_loss = torch.stack(center_losses).mean() if center_losses else torch.tensor(0.0, device=self.device)
        
        # calculate size loss only for frames with planes
        size_losses = []
        for t in range(seq_len):
            plane_mask = has_plane_seq[:, t].bool()
            if plane_mask.sum() > 0:
                size_loss_t = self.size_loss(
                    size_pred[:, t][plane_mask],
                    size_gt[:, t][plane_mask]
                ).mean()
                size_losses.append(size_loss_t)
        size_loss = torch.stack(size_losses).mean() if size_losses else torch.tensor(0.0, device=self.device)
        
        velocity_losses = []
        for t in range(seq_len):
            plane_mask = has_plane_seq[:, t].bool()
            if plane_mask.sum() > 0:
                velocity_loss_t = self.velocity_loss(
                    velocity_pred[:, t][plane_mask],
                    velocity_gt[:, t][plane_mask]
                ).mean()
                velocity_losses.append(velocity_loss_t)
        velocity_loss = torch.stack(velocity_losses).mean() if velocity_losses else torch.tensor(0.0, device=self.device)
        
        iou_losses = []
        for t in range(seq_len):
            plane_mask = has_plane_seq[:, t].bool()
            if plane_mask.sum() > 0:
                pred_centers = center_pred[:, t][plane_mask]
                pred_sizes = size_pred[:, t][plane_mask]
                pred_bbox = torch.cat([
                    pred_centers - pred_sizes / 2,
                    pred_centers + pred_sizes / 2
                ], dim=-1)
                
                gt_centers = center_gt[:, t][plane_mask]
                gt_sizes = size_gt[:, t][plane_mask]
                gt_bbox = torch.cat([
                    gt_centers - gt_sizes / 2,
                    gt_centers + gt_sizes / 2
                ], dim=-1)
                
                iou_loss_t = self.iou_loss(pred_bbox, gt_bbox).mean()
                iou_losses.append(iou_loss_t)
        iou_loss = torch.stack(iou_losses).mean() if iou_losses else torch.tensor(0.0, device=self.device)
        
        total_loss = detection_loss + center_loss + size_loss + velocity_loss + iou_loss
        
        loss_dict = {
            'total_loss': total_loss,
            'detection_loss': detection_loss,
            'center_loss': center_loss,
            'size_loss': size_loss,
            'velocity_loss': velocity_loss,
            'iou_loss': iou_loss
        }
        return loss_dict
    

    # main training loop
    def train(self, num_epochs=10):
        print(f"Training for {num_epochs} epochs")
        if self.use_particle_filter:
            print("Using particle filter enhancement")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("=" * 60)
            
            train_losses = self.train_epoch(epoch+1)
            val_losses = self.validate(epoch+1)
            
            self.scheduler.step()
            
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_losses': val_losses,
                    'train_losses': train_losses
                }, 'best_model.pth')
                print(f"Best model saved with validation loss: {val_losses['total_loss']:.4f}")
            
            self.train_losses.append(train_losses['total_loss'])
            self.val_losses.append(val_losses['total_loss'])
            self.detailed_train_losses.append(train_losses)
            self.detailed_val_losses.append(val_losses)
            
            self.save_epoch_to_csv(epoch+1, train_losses, val_losses)
            
            print(f"\nEpoch {epoch+1} Results:")
            print(f"{'Metric':<20} {'Train':<12} {'Val':<12} {'Difference':<12}")
            print("-" * 60)
            print(f"{'Total Loss':<20} {train_losses['total_loss']:<12.4f} {val_losses['total_loss']:<12.4f} {abs(train_losses['total_loss'] - val_losses['total_loss']):<12.4f}")
            print(f"{'Detection Loss':<20} {train_losses['detection_loss']:<12.4f} {val_losses['detection_loss']:<12.4f} {abs(train_losses['detection_loss'] - val_losses['detection_loss']):<12.4f}")
            print(f"{'Center Loss':<20} {train_losses['center_loss']:<12.4f} {val_losses['center_loss']:<12.4f} {abs(train_losses['center_loss'] - val_losses['center_loss']):<12.4f}")
            print(f"{'Size Loss':<20} {train_losses['size_loss']:<12.4f} {val_losses['size_loss']:<12.4f} {abs(train_losses['size_loss'] - val_losses['size_loss']):<12.4f}")
            print(f"{'Velocity Loss':<20} {train_losses['velocity_loss']:<12.4f} {val_losses['velocity_loss']:<12.4f} {abs(train_losses['velocity_loss'] - val_losses['velocity_loss']):<12.4f}")
            print(f"{'IoU Loss':<20} {train_losses['iou_loss']:<12.4f} {val_losses['iou_loss']:<12.4f} {abs(train_losses['iou_loss'] - val_losses['iou_loss']):<12.4f}")
            print(f"{'Pred Error':<20} {train_losses['prediction_error']:<12.2f} {val_losses['prediction_error']:<12.2f} {abs(train_losses['prediction_error'] - val_losses['prediction_error']):<12.2f}")
            if self.use_particle_filter:
                print(f"{'Enhancement':<20} {train_losses['pf_enhancement_rate']:<12.1f}% {val_losses.get('pf_enhancement_rate', 0):<11.1f}% {'N/A':<12}")
                print(f"{'Confidence':<20} {train_losses['pf_confidence_improvement']:<12.3f} {'N/A':<12} {'N/A':<12}")
                print(f"{'Smoothness':<20} {train_losses['pf_smoothness_improvement']:<12.3f} {'N/A':<12} {'N/A':<12}")
            print(f"{'Learning Rate':<20} {self.optimizer.param_groups[0]['lr']:<12.6f}")
            print("-" * 60)
        
        print(f"\nTraining finished. Best validation loss: {self.best_val_loss:.4f}")
        
        self.save_losses_to_csv()
        
        self.plot_training_history()
        
        return self.detailed_train_losses, self.detailed_val_losses
    

    # save training history to csv files
    def save_losses_to_csv(self):
        os.makedirs('training_history', exist_ok=True)
        
        # Prepare training data
        train_data = []
        for epoch, losses in enumerate(self.detailed_train_losses, 1):
            row = {'epoch': epoch, 'split': 'train'}
            row.update(losses)
            train_data.append(row)
        
        # Prepare validation data
        val_data = []
        for epoch, losses in enumerate(self.detailed_val_losses, 1):
            row = {'epoch': epoch, 'split': 'val'}
            row.update(losses)
            val_data.append(row)
        
        # Combine and save
        all_data = train_data + val_data
        df = pd.DataFrame(all_data)
        
        # Save to CSV
        csv_path = 'training_history/losses.csv'
        df.to_csv(csv_path, index=False)
        print(f"Losses saved to {csv_path}")
        
        # Also save separate files for easy analysis
        train_df = pd.DataFrame(train_data)
        val_df = pd.DataFrame(val_data)
        
        train_df.to_csv('training_history/train_losses.csv', index=False)
        val_df.to_csv('training_history/val_losses.csv', index=False)
        return df
    

    # save single epoch losses to csv for incremental tracking
    def save_epoch_to_csv(self, epoch, train_losses, val_losses):
        os.makedirs('training_history', exist_ok=True)
        
        # Prepare data for this epoch
        train_row = {'epoch': epoch, 'split': 'train'}
        train_row.update(train_losses)
        
        val_row = {'epoch': epoch, 'split': 'val'}
        val_row.update(val_losses)
        
        epoch_data = [train_row, val_row]
        epoch_df = pd.DataFrame(epoch_data)
        
        # File paths
        csv_path = 'training_history/losses_incremental.csv'
        
        # Append to CSV (create header if first epoch)
        if epoch == 1:
            epoch_df.to_csv(csv_path, index=False, mode='w')
        else:
            epoch_df.to_csv(csv_path, index=False, mode='a', header=False)
    
    
    # calculate intersection over union loss
    def iou_loss(self, pred_bbox, gt_bbox):
        x1 = torch.max(pred_bbox[:, 0], gt_bbox[:, 0])
        y1 = torch.max(pred_bbox[:, 1], gt_bbox[:, 1])
        x2 = torch.min(pred_bbox[:, 2], gt_bbox[:, 2])
        y2 = torch.min(pred_bbox[:, 3], gt_bbox[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        pred_area = (pred_bbox[:, 2] - pred_bbox[:, 0]) * (pred_bbox[:, 3] - pred_bbox[:, 1])
        gt_area = (gt_bbox[:, 2] - gt_bbox[:, 0]) * (gt_bbox[:, 3] - gt_bbox[:, 1])
        union = pred_area + gt_area - intersection
        
        iou = intersection / (union + 1e-6)
        
        return 1.0 - iou
    

    # plot detailed training and validation losses
    def plot_training_history(self):
        epochs = range(1, len(self.detailed_train_losses) + 1)
        
        plt.figure(figsize=(16, 12))
        
        train_total = [loss['total_loss'] for loss in self.detailed_train_losses]
        train_detection = [loss['detection_loss'] for loss in self.detailed_train_losses]
        train_center = [loss['center_loss'] for loss in self.detailed_train_losses]
        train_size = [loss['size_loss'] for loss in self.detailed_train_losses]
        train_velocity = [loss['velocity_loss'] for loss in self.detailed_train_losses]
        train_pred_error = [loss['prediction_error'] for loss in self.detailed_train_losses]
        
        val_total = [loss['total_loss'] for loss in self.detailed_val_losses]
        val_detection = [loss['detection_loss'] for loss in self.detailed_val_losses]
        val_center = [loss['center_loss'] for loss in self.detailed_val_losses]
        val_size = [loss['size_loss'] for loss in self.detailed_val_losses]
        val_velocity = [loss['velocity_loss'] for loss in self.detailed_val_losses]
        val_pred_error = [loss['prediction_error'] for loss in self.detailed_val_losses]
        
        plt.subplot(2, 3, 1)
        plt.plot(epochs, train_total, 'b-', label='Train Total', linewidth=2)
        plt.plot(epochs, val_total, 'r-', label='Val Total', linewidth=2)
        plt.title('Total Loss', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 2)
        plt.plot(epochs, train_detection, 'b-', label='Train Detection', linewidth=2)
        plt.plot(epochs, val_detection, 'r-', label='Val Detection', linewidth=2)
        plt.title('Detection Loss', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 3)
        plt.plot(epochs, train_center, 'b-', label='Train Center', linewidth=2)
        plt.plot(epochs, val_center, 'r-', label='Val Center', linewidth=2)
        plt.title('Center Loss', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 4)
        plt.plot(epochs, train_size, 'b-', label='Train Size', linewidth=2)
        plt.plot(epochs, val_size, 'r-', label='Val Size', linewidth=2)
        plt.title('Size Loss', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 5)
        plt.plot(epochs, train_velocity, 'b-', label='Train Velocity', linewidth=2)
        plt.plot(epochs, val_velocity, 'r-', label='Val Velocity', linewidth=2)
        plt.title('Velocity Loss', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 6)
        plt.plot(epochs, train_pred_error, 'b-', label='Train Pred Error', linewidth=2)
        plt.plot(epochs, val_pred_error, 'r-', label='Val Pred Error', linewidth=2)
        plt.title('Prediction Error (pixels)', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Error (pixels)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Training history saved as 'training_history.png'")
        
        final_train = self.detailed_train_losses[-1]
        final_val = self.detailed_val_losses[-1]
        print(f"\nFinal Performance Summary:")
        print(f"  Total Loss:      Train={final_train['total_loss']:.4f}, Val={final_val['total_loss']:.4f}")
        print(f"  Detection Loss:  Train={final_train['detection_loss']:.4f}, Val={final_val['detection_loss']:.4f}")
        print(f"  Center Loss:     Train={final_train['center_loss']:.4f}, Val={final_val['center_loss']:.4f}")
        print(f"  Size Loss:       Train={final_train['size_loss']:.4f}, Val={final_val['size_loss']:.4f}")
        print(f"  Velocity Loss:   Train={final_train['velocity_loss']:.4f}, Val={final_val['velocity_loss']:.4f}")
        print(f"  Pred Error:      Train={final_train['prediction_error']:.2f}px, Val={final_val['prediction_error']:.2f}px")


# main training script
def main():
    train_loader = get_dataloader(split='train', batch_size=1, shuffle=True, spike_mode='binary')
    val_loader = get_dataloader(split='val', batch_size=1, shuffle=False, spike_mode='binary')
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    model = SNN_tracker_model(input_channels=2, hidden_size=512)
    
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    trainer = ParticleFilterTrainer(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        device=device,
        use_particle_filter=True
    )

    trainer.train(num_epochs=2)
    


if __name__ == "__main__":
    main()