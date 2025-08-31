"""
SNN tracker inference and evaluation
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from main import SNN_tracker_model
from dataset import get_dataloader


# inference class for trained tracker
class TrackingInference:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = SNN_tracker_model(input_channels=2, hidden_size=512)
        
        # load trained model
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(device)
        self.model.eval()
        
        self.detection_threshold = 0.5
        

    # compute intersection over union for two boxes
    def compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    

    # compute tracking metrics for predictions vs ground truth
    def compute_tracking_metrics(self, pred_centers, gt_centers, pred_bboxes, gt_bboxes, 
                                pred_detections, gt_detections, pred_velocities=None):
        metrics = {}
        
        # Detection metrics
        pred_binary = pred_detections > self.detection_threshold
        tp = np.sum(pred_binary & gt_detections)
        fp = np.sum(pred_binary & ~gt_detections)
        fn = np.sum(~pred_binary & gt_detections)
        tn = np.sum(~pred_binary & ~gt_detections)
        
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['accuracy'] = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
        metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / \
                       (metrics['precision'] + metrics['recall']) \
                       if (metrics['precision'] + metrics['recall']) > 0 else 0
        
        # position errors (only for true positives)
        valid_frames = pred_binary & gt_detections
        if np.sum(valid_frames) > 0:
            position_errors = np.linalg.norm(
                gt_centers[valid_frames] - pred_centers[valid_frames], axis=1
            )
            metrics['mean_position_error'] = np.mean(position_errors)
            metrics['median_position_error'] = np.median(position_errors)
            metrics['std_position_error'] = np.std(position_errors)
            metrics['max_position_error'] = np.max(position_errors)
            
            # Percentile errors
            metrics['position_error_25'] = np.percentile(position_errors, 25)
            metrics['position_error_75'] = np.percentile(position_errors, 75)
            metrics['position_error_95'] = np.percentile(position_errors, 95)
        else:
            metrics['mean_position_error'] = np.inf
            metrics['median_position_error'] = np.inf
            metrics['std_position_error'] = 0
            metrics['max_position_error'] = np.inf
            metrics['position_error_25'] = np.inf
            metrics['position_error_75'] = np.inf
            metrics['position_error_95'] = np.inf
        
        # IoU metrics
        ious = []
        for i in range(len(pred_bboxes)):
            if pred_binary[i] and gt_detections[i]:
                iou = self.compute_iou(pred_bboxes[i], gt_bboxes[i])
                ious.append(iou)
        
        if ious:
            metrics['mean_iou'] = np.mean(ious)
            metrics['median_iou'] = np.median(ious)
            metrics['std_iou'] = np.std(ious)
            metrics['min_iou'] = np.min(ious)
            metrics['max_iou'] = np.max(ious)
            
            # IoU thresholds
            metrics['iou_50'] = np.sum(np.array(ious) > 0.5) / len(ious)  # % with IoU > 0.5
            metrics['iou_75'] = np.sum(np.array(ious) > 0.75) / len(ious)  # % with IoU > 0.75
        else:
            metrics['mean_iou'] = 0
            metrics['median_iou'] = 0
            metrics['std_iou'] = 0
            metrics['min_iou'] = 0
            metrics['max_iou'] = 0
            metrics['iou_50'] = 0
            metrics['iou_75'] = 0
        
        # trajectory smoothness (velocity consistency)
        if np.sum(valid_frames) > 1:
            valid_indices = np.where(valid_frames)[0]
            velocities = []
            for i in range(len(valid_indices) - 1):
                idx1, idx2 = valid_indices[i], valid_indices[i+1]
                if idx2 - idx1 == 1:  
                    vel = pred_centers[idx2] - pred_centers[idx1]
                    velocities.append(vel)
            
            if len(velocities) > 1:
                velocities = np.array(velocities)
                velocity_changes = np.diff(velocities, axis=0)
                metrics['velocity_consistency'] = np.mean(np.linalg.norm(velocity_changes, axis=1))
            else:
                metrics['velocity_consistency'] = 0
        else:
            metrics['velocity_consistency'] = 0
        
        # future prediction error (one-step ahead)
        future_errors = []
        for i in range(len(pred_centers) - 1):
            if pred_binary[i] and gt_detections[i+1]:
                # Simple constant velocity prediction
                predicted_next = pred_centers[i] + (pred_centers[i] - pred_centers[max(0, i-1)])
                actual_next = gt_centers[i+1]
                error = np.linalg.norm(predicted_next - actual_next)
                future_errors.append(error)
        
        if future_errors:
            metrics['mean_future_error'] = np.mean(future_errors)
            metrics['median_future_error'] = np.median(future_errors)
            metrics['std_future_error'] = np.std(future_errors)
        else:
            metrics['mean_future_error'] = np.inf
            metrics['median_future_error'] = np.inf
            metrics['std_future_error'] = 0
        
        # track continuity (how often we maintain tracking)
        if np.sum(gt_detections) > 0:
            # count continuous tracking segments
            tracking_segments = []
            current_segment = 0
            for i in range(len(pred_binary)):
                if pred_binary[i] and gt_detections[i]:
                    current_segment += 1
                elif current_segment > 0:
                    tracking_segments.append(current_segment)
                    current_segment = 0
            if current_segment > 0:
                tracking_segments.append(current_segment)
            
            if tracking_segments:
                metrics['mean_track_length'] = np.mean(tracking_segments)
                metrics['max_track_length'] = np.max(tracking_segments)
            else:
                metrics['mean_track_length'] = 0
                metrics['max_track_length'] = 0
                
            metrics['track_fragmentation'] = len(tracking_segments) / max(1, np.sum(gt_detections))
        else:
            metrics['mean_track_length'] = 0
            metrics['max_track_length'] = 0
            metrics['track_fragmentation'] = 0
        
        # velocity prediction accuracy (if available)
        if pred_velocities is not None:
            # calculate GT velocities
            gt_velocities = []
            pred_vel_list = []
            for i in range(len(gt_centers) - 1):
                if gt_detections[i] and gt_detections[i+1]:
                    gt_vel = gt_centers[i+1] - gt_centers[i]
                    gt_velocities.append(gt_vel)
                    pred_vel_list.append(pred_velocities[i])
            
            if gt_velocities:
                gt_velocities = np.array(gt_velocities)
                pred_vel_list = np.array(pred_vel_list)
                
                # convert velocity predictions from normalized [-1,1] back to pixel space
                velocity_errors = np.linalg.norm(gt_velocities - pred_vel_list, axis=1)
                
                metrics['mean_velocity_error'] = np.mean(velocity_errors)
                metrics['median_velocity_error'] = np.median(velocity_errors)
                metrics['std_velocity_error'] = np.std(velocity_errors)
            else:
                metrics['mean_velocity_error'] = np.inf
                metrics['median_velocity_error'] = np.inf
                metrics['std_velocity_error'] = 0
        else:
            metrics['mean_velocity_error'] = np.inf
            metrics['median_velocity_error'] = np.inf
            metrics['std_velocity_error'] = 0
        
        return metrics
    

    # create visualizations comparing predictions to ground truth
    def visualize_predictions(self, spike_sequence, detection_pred, center_pred, bbox_pred, 
                            velocity_pred, has_plane_seq, center_seq, bbox_seq, 
                            save_dir, recording_name):
        recording_dir = os.path.join(save_dir, f"recording_{recording_name}")
        os.makedirs(recording_dir, exist_ok=True)
        
        seq_len = spike_sequence.shape[1]
        
        for t in range(seq_len):
            # check if we should save this frame
            has_gt = has_plane_seq[t].item() > 0.5
            has_pred = detection_pred[t] > self.detection_threshold
            
            if not (has_gt or has_pred):
                continue  
            
            spike_frame = spike_sequence[0, t].cpu().numpy()  # [2, H, W]
            
            # create event representation 
            event_img = np.zeros((spike_frame.shape[1], spike_frame.shape[2], 3))
            event_img[:, :, 0] = spike_frame[0]  # Positive events in red channel
            event_img[:, :, 2] = spike_frame[1]  # Negative events in blue channel
            
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            
            
            axes[0].imshow(event_img)
            axes[0].set_title(f'Frame {t}: Event Data\n(Red=Positive, Blue=Negative)', fontsize=16)
            axes[0].axis('off')
            
            axes[1].imshow(event_img)

            title_parts = []
            
            if has_gt:
                # GT center 
                gt_center = center_seq[t].numpy()
                gt_bbox = bbox_seq[t].numpy()
                
                rect_gt = patches.Rectangle(
                    (gt_bbox[0], gt_bbox[1]), 
                    gt_bbox[2] - gt_bbox[0], 
                    gt_bbox[3] - gt_bbox[1],
                    linewidth=4, edgecolor='lime', facecolor='none', label='GT'
                )
                axes[1].add_patch(rect_gt)
                axes[1].plot(gt_center[0], gt_center[1], 'o', color='lime', 
                           markersize=12, markeredgecolor='black', markeredgewidth=3)
                axes[1].text(10, 40, f'GT: ({gt_center[0]:.1f}, {gt_center[1]:.1f})', 
                           color='lime', fontsize=14, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.8))
                title_parts.append("GT: ✓")
            else:
                title_parts.append("GT: ✗")
            
            if has_pred:
                pred_center = center_pred[t]
                pred_bbox = bbox_pred[t]
                pred_velocity = velocity_pred[t]
            
                rect_pred = patches.Rectangle(
                    (pred_bbox[0], pred_bbox[1]), 
                    pred_bbox[2] - pred_bbox[0], 
                    pred_bbox[3] - pred_bbox[1],
                    linewidth=4, edgecolor='red', facecolor='none', label='Pred'
                )
                axes[1].add_patch(rect_pred)
                axes[1].plot(pred_center[0], pred_center[1], 'o', color='red', 
                           markersize=12, markeredgecolor='black', markeredgewidth=3)
            
                vel_scale = 15  
                axes[1].arrow(pred_center[0], pred_center[1], 
                            pred_velocity[0] * vel_scale, pred_velocity[1] * vel_scale,
                            head_width=8, head_length=5, fc='orange', ec='orange', linewidth=3)
                info_text = f'PRED: ({pred_center[0]:.1f}, {pred_center[1]:.1f})\n'
                info_text += f'Vel: ({pred_velocity[0]:.2f}, {pred_velocity[1]:.2f})\n'
                info_text += f'Conf: {detection_pred[t]:.3f}'
                axes[1].text(10, 100, info_text, color='red', fontsize=14, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.8))
                
                title_parts.append(f"PRED: ✓ ({detection_pred[t]:.3f})")
            else:
                axes[1].text(10, 100, f'NO DETECTION\nConf: {detection_pred[t]:.3f}\n(Below {self.detection_threshold})', 
                           color='gray', fontsize=14, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.8))
                title_parts.append(f"PRED: ✗ ({detection_pred[t]:.3f})")
            
            # title
            axes[1].set_title(" | ".join(title_parts), fontsize=16, fontweight='bold')
            axes[1].axis('off')
            
            if has_gt and has_pred:
                gt_center = center_seq[t].numpy()
                center_error = np.linalg.norm(pred_center - gt_center)
                
                gt_bbox = bbox_seq[t].numpy()
                iou = self.compute_iou(pred_bbox, gt_bbox)
                
                comparison_text = f'Recording {recording_name} - Frame {t:03d} | Error: {center_error:.2f}px | IoU: {iou:.3f}'
                
                if center_error < 15 and iou > 0.7:
                    title_color = 'green'
                elif center_error < 25 and iou > 0.5:
                    title_color = 'orange'
                else:
                    title_color = 'red'
                
                fig.suptitle(comparison_text, fontsize=18, fontweight='bold', color=title_color)
            else:
                fig.suptitle(f'Recording {recording_name} - Frame {t:03d}', 
                           fontsize=18, fontweight='bold')
            
            frame_filename = f'frame_{t:03d}_gt{int(has_gt)}_pred{int(has_pred)}.png'
            frame_path = os.path.join(recording_dir, frame_filename)
            plt.savefig(frame_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
        
        # summary
        self.create_recording_summary(detection_pred, center_pred, bbox_pred, velocity_pred,
                                    has_plane_seq, center_seq, bbox_seq, recording_dir, recording_name)
    

    # summary plot 
    def create_recording_summary(self, detection_pred, center_pred, bbox_pred, velocity_pred,
                               has_plane_seq, center_seq, bbox_seq, save_dir, recording_name):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # get valid frames
        gt_frames = has_plane_seq.numpy() > 0.5
        pred_frames = detection_pred > self.detection_threshold
        
        # detection confidence over time
        frames = np.arange(len(detection_pred))
        axes[0, 0].plot(frames, detection_pred, 'b-', label='Detection Confidence', linewidth=2)
        axes[0, 0].axhline(y=self.detection_threshold, color='r', linestyle='--', label='Threshold')
        axes[0, 0].fill_between(frames, 0, gt_frames.astype(float), alpha=0.3, color='green', label='GT Presence')
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].set_ylabel('Confidence')
        axes[0, 0].set_title('Detection Confidence Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        if np.any(gt_frames):
            gt_centers = center_seq.numpy()[gt_frames]
            gt_frame_indices = frames[gt_frames]
            axes[0, 1].plot(gt_centers[:, 0], gt_centers[:, 1], 'g-o', label='GT Trajectory', 
                          linewidth=3, markersize=6, markeredgecolor='black')
        
        if np.any(pred_frames):
            pred_centers = center_pred[pred_frames]
            pred_frame_indices = frames[pred_frames]
            axes[0, 1].plot(pred_centers[:, 0], pred_centers[:, 1], 'r-s', label='Pred Trajectory', 
                          linewidth=2, markersize=4, markeredgecolor='black')
        
        axes[0, 1].set_xlabel('X (pixels)')
        axes[0, 1].set_ylabel('Y (pixels)')
        axes[0, 1].set_title('Trajectory Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_aspect('equal', adjustable='box')
        
        position_errors = []
        error_frames = []
        for i in range(len(detection_pred)):
            if gt_frames[i] and pred_frames[i]:
                error = np.linalg.norm(center_pred[i] - center_seq[i].numpy())
                position_errors.append(error)
                error_frames.append(i)
        
        if position_errors:
            axes[1, 0].plot(error_frames, position_errors, 'ro-', linewidth=2, markersize=4)
            axes[1, 0].set_xlabel('Frame')
            axes[1, 0].set_ylabel('Position Error (pixels)')
            axes[1, 0].set_title(f'Position Errors (Mean: {np.mean(position_errors):.2f}px)')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Valid Comparisons', ha='center', va='center', 
                          transform=axes[1, 0].transAxes, fontsize=14)
            axes[1, 0].set_title('Position Errors')
        
        if len(velocity_pred) > 1:
            vel_magnitudes = np.linalg.norm(velocity_pred, axis=1)
            axes[1, 1].plot(frames, vel_magnitudes, 'purple', linewidth=2, label='Predicted Velocity')
            
            gt_velocities = []
            for i in range(len(center_seq) - 1):
                if gt_frames[i] and gt_frames[i+1]:
                    gt_vel = center_seq[i+1].numpy() - center_seq[i].numpy()
                    gt_vel_mag = np.linalg.norm(gt_vel)
                    gt_velocities.append((i, gt_vel_mag))
            
            if gt_velocities:
                gt_frames_vel, gt_vel_mags = zip(*gt_velocities)
                axes[1, 1].plot(gt_frames_vel, gt_vel_mags, 'go-', linewidth=2, markersize=4, label='GT Velocity')
            
            axes[1, 1].set_xlabel('Frame')
            axes[1, 1].set_ylabel('Velocity Magnitude (pixels/frame)')
            axes[1, 1].set_title('Velocity Comparison')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', 
                          transform=axes[1, 1].transAxes, fontsize=14)
            axes[1, 1].set_title('Velocity Comparison')
        
        plt.suptitle(f'Recording {recording_name} - Complete Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'recording_{recording_name}_summary.png'), 
                   dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # metrics summary
        self.save_recording_metrics(detection_pred, center_pred, bbox_pred, velocity_pred,
                                  has_plane_seq, center_seq, bbox_seq, save_dir, recording_name)
    

    def save_recording_metrics(self, detection_pred, center_pred, bbox_pred, velocity_pred,
                             has_plane_seq, center_seq, bbox_seq, save_dir, recording_name):
        """
        Save detailed metrics for this specific recording.
        """
        # Compute metrics for this recording
        metrics = self.compute_tracking_metrics(
            center_pred, center_seq.numpy(), bbox_pred, bbox_seq.numpy(),
            detection_pred, has_plane_seq.numpy(), pred_velocities=velocity_pred
        )
        
        # Save to text file
        metrics_path = os.path.join(save_dir, f'recording_{recording_name}_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write(f"DETAILED METRICS FOR RECORDING {recording_name}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("DETECTION PERFORMANCE:\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1-Score: {metrics['f1']:.4f}\n\n")
            
            f.write("POSITION TRACKING:\n")
            f.write(f"Mean Error: {metrics['mean_position_error']:.2f} pixels\n")
            f.write(f"Median Error: {metrics['median_position_error']:.2f} pixels\n")
            f.write(f"95th Percentile Error: {metrics['position_error_95']:.2f} pixels\n\n")
            
            f.write("BOUNDING BOX (IoU) PERFORMANCE:\n")
            f.write(f"Mean IoU: {metrics['mean_iou']:.4f}\n")
            f.write(f"Median IoU: {metrics['median_iou']:.4f}\n")
            f.write(f"IoU > 0.5: {metrics['iou_50']*100:.1f}%\n")
            f.write(f"IoU > 0.75: {metrics['iou_75']*100:.1f}%\n\n")
            
            f.write("VELOCITY PREDICTION:\n")
            f.write(f"Mean Error: {metrics['mean_velocity_error']:.2f} pixels/frame\n")
            f.write(f"Median Error: {metrics['median_velocity_error']:.2f} pixels/frame\n\n")
            
            f.write("TRACK CONTINUITY:\n")
            f.write(f"Mean Track Length: {metrics['mean_track_length']:.1f} frames\n")
            f.write(f"Max Track Length: {metrics['max_track_length']:.0f} frames\n")
            f.write(f"Track Fragmentation: {metrics['track_fragmentation']:.4f}\n")
            f.write(f"Velocity Consistency: {metrics['velocity_consistency']:.4f}\n")
    

    def create_overview_summary(self, all_metrics, save_dir):
        """
        Create an overview summary showing performance across all recordings.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Extract metrics for each recording
        recording_ids = list(range(len(all_metrics)))
        f1_scores = [m['f1'] for m in all_metrics]
        position_errors = [m['median_position_error'] if not np.isinf(m['median_position_error']) else 0 for m in all_metrics]
        mean_ious = [m['mean_iou'] for m in all_metrics]
        track_lengths = [m['mean_track_length'] for m in all_metrics]
        velocity_errors = [m['median_velocity_error'] if not np.isinf(m['median_velocity_error']) else 0 for m in all_metrics]
        fragmentations = [m['track_fragmentation'] for m in all_metrics]
        
        # F1 scores across recordings
        axes[0, 0].bar(recording_ids, f1_scores, color='skyblue', edgecolor='navy')
        axes[0, 0].set_xlabel('Recording ID')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].set_title('Detection F1 Score per Recording')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Position errors across recordings
        axes[0, 1].bar(recording_ids, position_errors, color='lightcoral', edgecolor='darkred')
        axes[0, 1].set_xlabel('Recording ID')
        axes[0, 1].set_ylabel('Median Position Error (px)')
        axes[0, 1].set_title('Position Error per Recording')
        axes[0, 1].grid(True, alpha=0.3)
        
        # IoU across recordings
        axes[0, 2].bar(recording_ids, mean_ious, color='lightgreen', edgecolor='darkgreen')
        axes[0, 2].set_xlabel('Recording ID')
        axes[0, 2].set_ylabel('Mean IoU')
        axes[0, 2].set_title('IoU per Recording')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Track lengths across recordings
        axes[1, 0].bar(recording_ids, track_lengths, color='gold', edgecolor='orange')
        axes[1, 0].set_xlabel('Recording ID')
        axes[1, 0].set_ylabel('Mean Track Length (frames)')
        axes[1, 0].set_title('Track Length per Recording')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Velocity errors across recordings
        axes[1, 1].bar(recording_ids, velocity_errors, color='plum', edgecolor='purple')
        axes[1, 1].set_xlabel('Recording ID')
        axes[1, 1].set_ylabel('Median Velocity Error (px/frame)')
        axes[1, 1].set_title('Velocity Error per Recording')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Performance distribution histogram
        axes[1, 2].hist(f1_scores, bins=10, alpha=0.7, color='skyblue', edgecolor='navy', label='F1 Score')
        axes[1, 2].axvline(np.mean(f1_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(f1_scores):.3f}')
        axes[1, 2].set_xlabel('F1 Score')
        axes[1, 2].set_ylabel('Number of Recordings')
        axes[1, 2].set_title('F1 Score Distribution')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('Performance Overview Across All Recordings', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'overview_summary.png'), dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # create detailed statistics table
        stats_path = os.path.join(save_dir, 'detailed_statistics.txt')
        with open(stats_path, 'w') as f:
            f.write("DETAILED STATISTICS ACROSS ALL RECORDINGS\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("RECORDING-BY-RECORDING BREAKDOWN:\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Rec':<4} {'F1':<6} {'PosErr':<8} {'IoU':<6} {'TrkLen':<8} {'VelErr':<8} {'Frag':<6}\n")
            f.write("-" * 70 + "\n")
            
            for i, metrics in enumerate(all_metrics):
                f.write(f"{i:<4} {metrics['f1']:<6.3f} {metrics['median_position_error'] if not np.isinf(metrics['median_position_error']) else 0:<8.1f} "
                       f"{metrics['mean_iou']:<6.3f} {metrics['mean_track_length']:<8.1f} "
                       f"{metrics['median_velocity_error'] if not np.isinf(metrics['median_velocity_error']) else 0:<8.1f} {metrics['track_fragmentation']:<6.3f}\n")
            
            f.write("-" * 70 + "\n\n")
            
            f.write("SUMMARY STATISTICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Recordings: {len(all_metrics)}\n")
            f.write(f"Mean F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}\n")
            f.write(f"Mean Position Error: {np.mean([e for e in position_errors if e > 0]):.2f} ± {np.std([e for e in position_errors if e > 0]):.2f} px\n")
            f.write(f"Mean IoU: {np.mean(mean_ious):.4f} ± {np.std(mean_ious):.4f}\n")
            f.write(f"Mean Track Length: {np.mean(track_lengths):.1f} ± {np.std(track_lengths):.1f} frames\n")
            f.write(f"Mean Velocity Error: {np.mean([e for e in velocity_errors if e > 0]):.2f} ± {np.std([e for e in velocity_errors if e > 0]):.2f} px/frame\n")
            f.write(f"Mean Fragmentation: {np.mean(fragmentations):.4f} ± {np.std(fragmentations):.4f}\n\n")
            
            f.write("PERFORMANCE CATEGORIES:\n")
            f.write("-" * 30 + "\n")
            excellent = sum(1 for f1 in f1_scores if f1 > 0.9)
            good = sum(1 for f1 in f1_scores if 0.7 <= f1 <= 0.9)
            fair = sum(1 for f1 in f1_scores if 0.5 <= f1 < 0.7)
            poor = sum(1 for f1 in f1_scores if f1 < 0.5)
            
            f.write(f"Excellent (F1 > 0.9): {excellent}/{len(all_metrics)} ({excellent/len(all_metrics)*100:.1f}%)\n")
            f.write(f"Good (0.7 ≤ F1 ≤ 0.9): {good}/{len(all_metrics)} ({good/len(all_metrics)*100:.1f}%)\n")
            f.write(f"Fair (0.5 ≤ F1 < 0.7): {fair}/{len(all_metrics)} ({fair/len(all_metrics)*100:.1f}%)\n")
            f.write(f"Poor (F1 < 0.5): {poor}/{len(all_metrics)} ({poor/len(all_metrics)*100:.1f}%)\n")
    
    # evaluate tracking performance on test set
    def evaluate_test_set(self, save_dir='tracking_evaluation', save_visualizations=True):
        os.makedirs(save_dir, exist_ok=True)
        
        # load test data
        test_loader = get_dataloader(split='test', batch_size=1, shuffle=False, spike_mode='binary')
        
        all_metrics = []
        
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # extract data
            spike_sequence = batch['spike_sequence'].to(self.device)
            has_plane_seq = batch['has_plane_sequence'].squeeze(0)
            center_seq = batch['center_sequence'].squeeze(0)
            bbox_seq = batch['bbox_sequence'].squeeze(0)
            
            # run inference
            with torch.no_grad():
                detection_pred, center_pred, size_pred, velocity_pred, bbox_pred = self.model(spike_sequence)
                
                # convert to numpy and denormalize
                detection_pred = detection_pred.squeeze().cpu().numpy()
                center_pred = center_pred.squeeze().cpu().numpy()
                velocity_pred = velocity_pred.squeeze().cpu().numpy()
                bbox_pred = bbox_pred.squeeze().cpu().numpy()
                
                # denormalize coordinates
                center_pred[:, 0] *= self.model.width
                center_pred[:, 1] *= self.model.height
                bbox_pred[:, 0] *= self.model.width
                bbox_pred[:, 1] *= self.model.height
                bbox_pred[:, 2] *= self.model.width
                bbox_pred[:, 3] *= self.model.height
                
                # scale velocity back to pixel space 
                velocity_pred[:, 0] *= self.model.width * 0.01  
                velocity_pred[:, 1] *= self.model.height * 0.01
            
            # create detailed visualizations for this recording
            if save_visualizations:
                self.visualize_predictions(
                    spike_sequence, detection_pred, center_pred, bbox_pred, velocity_pred,
                    has_plane_seq, center_seq, bbox_seq, save_dir, batch_idx
                )
            
            # compute metrics
            metrics = self.compute_tracking_metrics(
                center_pred,
                center_seq.numpy(),
                bbox_pred,
                bbox_seq.numpy(),
                detection_pred,
                has_plane_seq.numpy(),
                pred_velocities=velocity_pred
            )
            
            all_metrics.append(metrics)
        
        # aggregate metrics
        aggregated = self.aggregate_metrics(all_metrics)
        
        # save results
        self.save_metrics(aggregated, save_dir)
        
        # create overview summary
        if save_visualizations:
            self.create_overview_summary(all_metrics, save_dir)
        
        return aggregated
    

    # aggregate metrics across all test sequences
    def aggregate_metrics(self, metrics_list):
        if not metrics_list:
            return {}
        
        aggregated = {}
        
        # get all metric keys
        keys = metrics_list[0].keys()
        
        for key in keys:
            values = [m[key] for m in metrics_list if not np.isinf(m[key])]
            if values:
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
                aggregated[f'{key}_median'] = np.median(values)
                aggregated[f'{key}_min'] = np.min(values)
                aggregated[f'{key}_max'] = np.max(values)
            else:
                aggregated[f'{key}_mean'] = np.inf
                aggregated[f'{key}_std'] = 0
                aggregated[f'{key}_median'] = np.inf
                aggregated[f'{key}_min'] = np.inf
                aggregated[f'{key}_max'] = np.inf
        
        return aggregated
    

    # save metrics to file and create visualization
    def save_metrics(self, metrics, save_dir):
        # save to text file
        metrics_path = os.path.join(save_dir, 'tracking_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write("Tracking Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            
            # detection performance
            f.write("Detection Performance:\n")
            f.write(f"Precision: {metrics.get('precision_mean', 0):.4f} ± {metrics.get('precision_std', 0):.4f}\n")
            f.write(f"Recall: {metrics.get('recall_mean', 0):.4f} ± {metrics.get('recall_std', 0):.4f}\n")
            f.write(f"F1-Score: {metrics.get('f1_mean', 0):.4f} ± {metrics.get('f1_std', 0):.4f}\n\n")
            
            # Position Tracking
            f.write("POSITION TRACKING:\n")
            f.write(f"Mean Error: {metrics.get('mean_position_error_mean', np.inf):.2f} ± {metrics.get('mean_position_error_std', 0):.2f} pixels\n")
            f.write(f"Median Error: {metrics.get('median_position_error_mean', np.inf):.2f} pixels\n")
            f.write(f"95th Percentile Error: {metrics.get('position_error_95_mean', np.inf):.2f} pixels\n\n")
            
            # IoU Performance
            f.write("BOUNDING BOX (IoU) PERFORMANCE:\n")
            f.write(f"Mean IoU: {metrics.get('mean_iou_mean', 0):.4f} ± {metrics.get('mean_iou_std', 0):.4f}\n")
            f.write(f"Median IoU: {metrics.get('median_iou_mean', 0):.4f}\n")
            f.write(f"IoU > 0.5: {metrics.get('iou_50_mean', 0)*100:.1f}%\n")
            f.write(f"IoU > 0.75: {metrics.get('iou_75_mean', 0)*100:.1f}%\n\n")
            
            # Future Prediction
            f.write("FUTURE PREDICTION (One-step ahead):\n")
            f.write(f"Mean Error: {metrics.get('mean_future_error_mean', np.inf):.2f} ± {metrics.get('mean_future_error_std', 0):.2f} pixels\n")
            f.write(f"Median Error: {metrics.get('median_future_error_mean', np.inf):.2f} pixels\n\n")
            
            # Velocity Prediction
            f.write("VELOCITY PREDICTION:\n")
            f.write(f"Mean Error: {metrics.get('mean_velocity_error_mean', np.inf):.2f} ± {metrics.get('mean_velocity_error_std', 0):.2f} pixels/frame\n")
            f.write(f"Median Error: {metrics.get('median_velocity_error_mean', np.inf):.2f} pixels/frame\n\n")
            
            # Track Continuity
            f.write("TRACK CONTINUITY:\n")
            f.write(f"Mean Track Length: {metrics.get('mean_track_length_mean', 0):.1f} frames\n")
            f.write(f"Max Track Length: {metrics.get('max_track_length_max', 0):.0f} frames\n")
            f.write(f"Track Fragmentation: {metrics.get('track_fragmentation_mean', 0):.4f}\n")
            f.write(f"Velocity Consistency: {metrics.get('velocity_consistency_mean', 0):.4f}\n")
        
        # Create summary plot
        self.create_summary_plot(metrics, save_dir)
        
        return metrics_path
    

    def create_summary_plot(self, metrics, save_dir):
        """
        Create visualization of key metrics.
        """
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Detection metrics
        det_metrics = ['precision_mean', 'recall_mean', 'f1_mean']
        det_values = [metrics.get(m, 0) for m in det_metrics]
        det_labels = ['Precision', 'Recall', 'F1']
        axes[0, 0].bar(det_labels, det_values)
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].set_title('Detection Performance')
        axes[0, 0].set_ylabel('Score')
        
        # Position error distribution
        pos_metrics = ['mean_position_error_mean', 'median_position_error_mean', 
                      'position_error_95_mean']
        pos_values = [metrics.get(m, 0) for m in pos_metrics]
        pos_labels = ['Mean', 'Median', '95th %ile']
        axes[0, 1].bar(pos_labels, pos_values)
        axes[0, 1].set_title('Position Error (pixels)')
        axes[0, 1].set_ylabel('Error (pixels)')
        
        # IoU distribution
        iou_metrics = ['mean_iou_mean', 'iou_50_mean', 'iou_75_mean']
        iou_values = [metrics.get(m, 0) for m in iou_metrics]
        iou_values[1:] = [v*100 for v in iou_values[1:]]  # Convert to percentage
        iou_labels = ['Mean IoU', 'IoU>0.5 (%)', 'IoU>0.75 (%)']
        axes[0, 2].bar(iou_labels, iou_values)
        axes[0, 2].set_title('Bounding Box Performance')
        
        # Velocity prediction
        vel_metrics = ['mean_velocity_error_mean', 'median_velocity_error_mean']
        vel_values = [metrics.get(m, 0) for m in vel_metrics]
        vel_labels = ['Mean', 'Median']
        axes[0, 3].bar(vel_labels, vel_values)
        axes[0, 3].set_title('Velocity Error (pixels/frame)')
        axes[0, 3].set_ylabel('Error (pixels/frame)')
        
        # Future prediction
        future_metrics = ['mean_future_error_mean', 'median_future_error_mean']
        future_values = [metrics.get(m, 0) for m in future_metrics]
        future_labels = ['Mean', 'Median']
        axes[1, 0].bar(future_labels, future_values)
        axes[1, 0].set_title('Future Prediction Error (pixels)')
        axes[1, 0].set_ylabel('Error (pixels)')
        
        # Track continuity
        track_metrics = ['mean_track_length_mean', 'track_fragmentation_mean']
        track_values = [metrics.get(m, 0) for m in track_metrics]
        track_labels = ['Mean Length', 'Fragmentation']
        axes[1, 1].bar(track_labels, track_values)
        axes[1, 1].set_title('Track Continuity')
        
        # Velocity consistency
        vel_consist_metrics = ['velocity_consistency_mean']
        vel_consist_values = [metrics.get(m, 0) for m in vel_consist_metrics]
        vel_consist_labels = ['Consistency']
        axes[1, 2].bar(vel_consist_labels, vel_consist_values)
        axes[1, 2].set_title('Velocity Consistency')
        axes[1, 2].set_ylabel('Consistency Score')
        
        # Summary statistics table
        axes[1, 3].axis('off')
        summary_text = f"""
        Summary Statistics:
        
        Detection F1: {metrics.get('f1_mean', 0):.3f}
        Position Error: {metrics.get('median_position_error_mean', np.inf):.1f} px
        Mean IoU: {metrics.get('mean_iou_mean', 0):.3f}
        Velocity Error: {metrics.get('median_velocity_error_mean', np.inf):.1f} px/f
        Future Error: {metrics.get('median_future_error_mean', np.inf):.1f} px
        Track Length: {metrics.get('mean_track_length_mean', 0):.1f} frames
        """
        axes[1, 3].text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center')
        
        plt.suptitle('Enhanced SNN Tracking Performance Evaluation', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'tracking_metrics.png'), dpi=150)
        plt.close()


# main evaluation script
def main():
    model_path = '/Users/banika/Desktop/master_submission_code/best_model.pth'
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Available models:")
        for file in os.listdir('.'):
            if file.endswith('.pth'):
                print(f"  - {file}")
        return
    
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Run evaluation
    print(f"Loading model: {model_path}")
    evaluator = TrackingInference(model_path, device=device)
    
    print("Running evaluation on test set...")
    metrics = evaluator.evaluate_test_set(save_dir='inference_results', save_visualizations=True)
    
    # print key results
    print("\nTracking Results:")
    print(f"F1 Score: {metrics.get('f1_mean', 0):.4f}")
    print(f"Precision: {metrics.get('precision_mean', 0):.4f}")
    print(f"Recall: {metrics.get('recall_mean', 0):.4f}")
    print(f"Mean Position Error: {metrics.get('mean_position_error_mean', np.inf):.2f} pixels")
    print(f"Mean IoU: {metrics.get('mean_iou_mean', 0):.4f}")
    print(f"Results saved in: inference_results/")


if __name__ == "__main__":
    main()