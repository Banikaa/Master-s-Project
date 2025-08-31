# foveated SNN tracker with fixed parameters

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import pandas as pd
from tqdm import tqdm
from main import SNN_tracker_model
from dataset import get_dataloader


class FoveatedTracker:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = SNN_tracker_model(input_channels=2, hidden_size=512)
        
        # load model
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(device)
        self.model.eval()
        
        self.detection_threshold = 0.5
        self.enable_foveation = True
        
        # fixed foveation parameters 
        self.attention_falloff = 'gaussian'
        self.min_attention = 0.5
        self.falloff_sharpness = 1.2
        self.fovea_safe_zone_multiplier = 1.5  
        
        # stats tracking
        self.foveation_stats = {
            'frames_processed': 0,
            'frames_foveated': 0,
            'total_events_original': 0,
            'total_events_kept': 0,
            'total_events_removed': 0,
        }
        

    def create_foveated_input(self, spike_frame, predicted_center, predicted_size):
        if not self.enable_foveation:
            return spike_frame, None, None
            
        H, W = spike_frame.shape[1], spike_frame.shape[2]
        device = spike_frame.device
        
        # create coordinate grids
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=device),
            torch.arange(W, dtype=torch.float32, device=device),
            indexing='ij'
        )
        
        # calculate distance from predicted center
        center_x, center_y = predicted_center[0], predicted_center[1]
        dist_from_center = torch.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        # use bbox size directly for fovea radius
        bbox_width, bbox_height = predicted_size[0], predicted_size[1]
        fovea_radius = max(bbox_width, bbox_height) / 2  # Half of largest bbox dimension
        
        # create safe zone around fovea with high attention
        safe_zone_radius = fovea_radius * self.fovea_safe_zone_multiplier
        
        # create attention map 
        attention_map = torch.ones_like(dist_from_center)
        
        # apply gaussian falloff only outside the safe zone
        outside_safe_zone = dist_from_center > safe_zone_radius
        if outside_safe_zone.any():
            # gaussian falloff starting from safe zone edge
            adjusted_distance = dist_from_center - safe_zone_radius
            sigma = fovea_radius / self.falloff_sharpness
            falloff_attention = torch.exp(-(adjusted_distance**2) / (2 * sigma**2))
            
            # apply falloff only outside safe zone
            attention_map = torch.where(outside_safe_zone, falloff_attention, attention_map)
        
        # ensure minimum attention level
        attention_map = torch.clamp(attention_map, min=self.min_attention, max=1.0)
        
        # apply probabilistic sampling based on attention map
        foveated_frame = spike_frame.clone()
        
        # count events before and after foveation
        original_events = spike_frame.sum().item()
        kept_events = 0
        
        for channel in range(spike_frame.shape[0]):
            channel_data = spike_frame[channel]
            
            # create random mask for probabilistic sampling
            random_mask = torch.rand_like(channel_data, device=device)
            
            # keep events where random value < attention weight
            keep_mask = random_mask < attention_map
            
            # apply mask
            foveated_frame[channel] = channel_data * keep_mask.float()
            
            # count kept events for this channel
            kept_events += foveated_frame[channel].sum().item()
        
        # calculate event-based metrics
        removed_events = original_events - kept_events
        event_reduction = removed_events / original_events if original_events > 0 else 0
        
        # calculate effective fovea bounds for visualization
        attention_threshold = 0.5
        mask_50 = attention_map > attention_threshold
        if mask_50.any():
            y_indices, x_indices = torch.where(mask_50)
            fovea_x1, fovea_x2 = x_indices.min().item(), x_indices.max().item()
            fovea_y1, fovea_y2 = y_indices.min().item(), y_indices.max().item()
            has_effective_fovea = True
        else:
            fovea_x1 = fovea_y1 = fovea_x2 = fovea_y2 = None
            has_effective_fovea = False
        
        fovea_info = {
            'center': (center_x, center_y),
            'bounds': (fovea_x1, fovea_y1, fovea_x2, fovea_y2) if has_effective_fovea else None,
            'has_effective_fovea': has_effective_fovea,
            'fovea_radius': fovea_radius,
            'safe_zone_radius': safe_zone_radius,
            'falloff_sharpness': self.falloff_sharpness,
            'min_attention': self.min_attention,
            'bbox_width': bbox_width,
            'bbox_height': bbox_height,
            'original_events': original_events,
            'kept_events': kept_events,
            'removed_events': removed_events,
            'event_reduction': event_reduction,
            'falloff_type': self.attention_falloff,
            'fixed_params': True,
            'less_aggressive': True
        }
        
        return foveated_frame, attention_map.cpu().numpy(), fovea_info
    

    def process_sequence_fixed_foveation(self, spike_sequence, recording_id=0):
        batch_size, seq_len, channels, height, width = spike_sequence.shape
        dt = 0.005
        
        foveated_sequence = spike_sequence.clone()
        foveation_info = []
        attention_maps = []
        
        # store all predictions frame by frame
        all_detection_pred = []
        all_center_pred = []
        all_velocity_pred = []
        all_bbox_pred = []
        
        frames_foveated = 0
        
        # process frame by frame
        for t in range(seq_len):
            # get prediction for current frame
            current_input = foveated_sequence[:, :t+1]
            
            with torch.no_grad():
                detection_pred, center_pred, _, velocity_pred, bbox_pred = self.model(current_input)
                
                # extract prediction for current frame
                det_t = detection_pred[0, -1].item()
                center_t = center_pred[0, -1].cpu().numpy()
                velocity_t = velocity_pred[0, -1].cpu().numpy()
                bbox_t = bbox_pred[0, -1].cpu().numpy()
                
                # denormalize current predictions
                center_t[0] *= self.model.width
                center_t[1] *= self.model.height
                velocity_t[0] *= self.model.width * 0.01
                velocity_t[1] *= self.model.height * 0.01
                bbox_t[0] *= self.model.width
                bbox_t[1] *= self.model.height
                bbox_t[2] *= self.model.width
                bbox_t[3] *= self.model.height
                
                # store predictions
                all_detection_pred.append(det_t)
                all_center_pred.append(center_t)
                all_velocity_pred.append(velocity_t)
                all_bbox_pred.append(bbox_t)
            
            # use this prediction to foveate the next frame
            if t < seq_len - 1:
                if det_t > self.detection_threshold:
                    # predict next frame center using current velocity
                    predicted_next_center = center_t + velocity_t * dt
                    
                    # use current bbox size for next frame prediction
                    current_bbox_width = bbox_t[2] - bbox_t[0]
                    current_bbox_height = bbox_t[3] - bbox_t[1]
                    predicted_next_size = np.array([current_bbox_width, current_bbox_height])
                    
                    # apply foveation to next frame
                    next_frame_original = spike_sequence[0, t+1]
                    foveated_next_frame, attention_map, fovea_info = self.create_foveated_input(
                        next_frame_original, predicted_next_center, predicted_next_size
                    )
                    
                    if fovea_info:
                        foveated_sequence[0, t+1] = foveated_next_frame
                        frames_foveated += 1
                        
                        # update stats
                        self.foveation_stats['frames_foveated'] += 1
                        self.foveation_stats['total_events_original'] += fovea_info['original_events']
                        self.foveation_stats['total_events_kept'] += fovea_info['kept_events']
                        
                        # store foveation info for next frame
                        if len(foveation_info) <= t:
                            foveation_info.extend([None] * (t + 1 - len(foveation_info)))
                        if len(attention_maps) <= t:
                            attention_maps.extend([None] * (t + 1 - len(attention_maps)))
                        
                        # next frame gets the foveation info
                        if len(foveation_info) == t + 1:
                            foveation_info.append(fovea_info)
                            attention_maps.append(attention_map)
                        else:
                            foveation_info[t+1] = fovea_info
                            attention_maps[t+1] = attention_map

            
            # add foveation info for current frame
            while len(foveation_info) <= t:
                foveation_info.append(None)
            while len(attention_maps) <= t:
                attention_maps.append(None)
        
        print(f"recording {recording_id}: foveated {frames_foveated}/{seq_len-1} frames")
        
        # convert predictions to numpy arrays
        detection_pred_final = np.array(all_detection_pred)
        center_pred_final = np.array(all_center_pred)
        velocity_pred_final = np.array(all_velocity_pred)
        bbox_pred_final = np.array(all_bbox_pred)
        
        return detection_pred_final, center_pred_final, velocity_pred_final, bbox_pred_final, foveated_sequence, attention_maps, foveation_info
    

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
    

    def evaluate_test_set(self, save_dir='fixed_foveation_evaluation'):
        os.makedirs(save_dir, exist_ok=True)
        
        test_loader = get_dataloader(split='test', batch_size=1, shuffle=False, spike_mode='binary')
        
        # metrics tracking
        all_position_errors = []
        all_iou_scores = []
        all_future_position_errors = []
        
        # detection metrics
        true_positives = 0
        false_positives = 0 
        false_negatives = 0
        true_negatives = 0
        
        # event-based metrics
        total_original_events = 0
        total_kept_events = 0
        foveated_frame_count = 0
        total_frame_count = 0
        
        # per-recording metrics
        recording_metrics = []
        
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="processing")):
                
            spike_sequence = batch['spike_sequence'].to(self.device)
            has_plane_seq = batch['has_plane_sequence'].squeeze(0)
            center_seq = batch['center_sequence'].squeeze(0)
            bbox_seq = batch['bbox_sequence'].squeeze(0)
            
            # process with fixed parameter foveation
            results = self.process_sequence_fixed_foveation(spike_sequence, batch_idx)
            detection_pred, center_pred, velocity_pred, bbox_pred, foveated_sequence, attention_maps, foveation_info = results
            
            # calculate metrics for this recording
            gt_frames = has_plane_seq.numpy() > 0.5
            pred_frames = detection_pred > self.detection_threshold
            
            position_errors = []
            iou_scores = []
            future_position_errors = []
            
            # recording-level metrics
            recording_original_events = 0
            recording_kept_events = 0
            recording_foveated_frames = 0
            
            rec_tp = rec_fp = rec_fn = rec_tn = 0
            
            for i in range(len(detection_pred)):
                total_frame_count += 1
                
                # detection metrics
                gt_has_plane = gt_frames[i]
                pred_has_plane = pred_frames[i]
                
                if gt_has_plane and pred_has_plane:
                    true_positives += 1
                    rec_tp += 1
                elif not gt_has_plane and pred_has_plane:
                    false_positives += 1
                    rec_fp += 1
                elif gt_has_plane and not pred_has_plane:
                    false_negatives += 1
                    rec_fn += 1
                else:
                    true_negatives += 1
                    rec_tn += 1
                
                # foveation stats
                if i < len(foveation_info) and foveation_info[i] is not None:
                    fovea_info = foveation_info[i]
                    recording_original_events += fovea_info['original_events']
                    recording_kept_events += fovea_info['kept_events']
                    recording_foveated_frames += 1
                    foveated_frame_count += 1
                
                # tracking accuracy (only for true positives)
                if gt_has_plane and pred_has_plane:
                    # position error
                    error = np.linalg.norm(center_pred[i] - center_seq[i].numpy())
                    position_errors.append(float(error))
                    
                    # IoU
                    iou = self.compute_iou(bbox_pred[i], bbox_seq[i].numpy())
                    iou_scores.append(float(iou))
                    
                    # future prediction error
                    if i < len(detection_pred) - 1 and i < len(center_seq) - 1:
                        predicted_next_center = center_pred[i] + velocity_pred[i] * 0.005
                        actual_next_center = center_seq[i + 1].numpy()
                        future_error = np.linalg.norm(predicted_next_center - actual_next_center)
                        future_position_errors.append(float(future_error))
            
            # update global totals
            total_original_events += recording_original_events
            total_kept_events += recording_kept_events
            
            all_position_errors.extend(position_errors)
            all_iou_scores.extend(iou_scores)
            all_future_position_errors.extend(future_position_errors)
            
            # store recording metrics
            recording_event_reduction = 0
            if recording_original_events > 0:
                recording_event_reduction = (recording_original_events - recording_kept_events) / recording_original_events
            
            rec_precision = rec_tp / (rec_tp + rec_fp) if (rec_tp + rec_fp) > 0 else 0
            rec_recall = rec_tp / (rec_tp + rec_fn) if (rec_tp + rec_fn) > 0 else 0
            rec_f1 = 2 * (rec_precision * rec_recall) / (rec_precision + rec_recall) if (rec_precision + rec_recall) > 0 else 0
            
            recording_metrics.append({
                'recording_id': int(batch_idx),
                'frames_total': int(len(detection_pred)),
                'frames_foveated': int(recording_foveated_frames),
                'foveation_rate': float(recording_foveated_frames / len(detection_pred) if len(detection_pred) > 0 else 0),
                'original_events': int(recording_original_events),
                'kept_events': int(recording_kept_events),
                'event_reduction': float(recording_event_reduction),
                'mean_position_error': float(np.mean(position_errors) if position_errors else 0),
                'mean_iou': float(np.mean(iou_scores) if iou_scores else 0),
                'precision': float(rec_precision),
                'recall': float(rec_recall),
                'f1_score': float(rec_f1),
                'valid_predictions': int(len(iou_scores))
            })
            
            # create visualizations for this recording
            self.create_foveation_visualization(
                spike_sequence, detection_pred, center_pred, bbox_pred,
                has_plane_seq, center_seq, bbox_seq, save_dir, batch_idx, 
                foveated_sequence, attention_maps, foveation_info
            )
        
        # calculate overall metrics
        total_removed_events = total_original_events - total_kept_events
        overall_event_reduction = total_removed_events / total_original_events if total_original_events > 0 else 0
        overall_foveation_rate = foveated_frame_count / total_frame_count if total_frame_count > 0 else 0
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_positives + true_negatives) / total_frame_count if total_frame_count > 0 else 0
        
        # print results
        print(f"\nfixed parameter foveation results")
        print(f"accuracy: {accuracy:.3f}")
        print(f"mean position error: {np.mean(all_position_errors):.1f}px" if all_position_errors else "no tracking data")
        print(f"mean iou: {np.mean(all_iou_scores):.3f}" if all_iou_scores else "no iou data")
        print(f"frames foveated: {foveated_frame_count}/{total_frame_count} ({overall_foveation_rate*100:.1f}%)")
        print(f"event reduction: {overall_event_reduction*100:.1f}%")
        
        # save metrics to csv
        # save overall metrics
        overall_metrics_df = pd.DataFrame([{
            'min_attention': float(self.min_attention),
            'falloff_sharpness': float(self.falloff_sharpness),
            'safe_zone_multiplier': float(self.fovea_safe_zone_multiplier),
            'attention_falloff': str(self.attention_falloff),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'position_error_mean': float(np.mean(all_position_errors) if all_position_errors else 0),
            'position_error_std': float(np.std(all_position_errors) if all_position_errors else 0),
            'iou_mean': float(np.mean(all_iou_scores) if all_iou_scores else 0),
            'iou_std': float(np.std(all_iou_scores) if all_iou_scores else 0),
            'foveation_rate': float(overall_foveation_rate),
            'event_reduction': float(overall_event_reduction),
            'total_original_events': int(total_original_events),
            'total_kept_events': int(total_kept_events)
        }])
        
        # save per-recording metrics
        recording_metrics_df = pd.DataFrame(recording_metrics)
        
        overall_metrics_df.to_csv(os.path.join(save_dir, 'fixed_parameter_metrics.csv'), index=False)
        recording_metrics_df.to_csv(os.path.join(save_dir, 'per_recording_metrics.csv'), index=False)
        
        return all_position_errors, all_iou_scores, {'overall_metrics': overall_metrics_df, 'recording_metrics': recording_metrics_df}
    

    def create_foveation_visualization(self, spike_sequence, detection_pred, center_pred, bbox_pred,
                                     has_plane_seq, center_seq, bbox_seq, save_dir, recording_name, 
                                     foveated_sequence, attention_maps, foveation_info):
        recording_dir = os.path.join(save_dir, f"recording_{recording_name}")
        os.makedirs(recording_dir, exist_ok=True)
        
        seq_len = spike_sequence.shape[1]
        
        for t in range(seq_len):
            has_gt = has_plane_seq[t].item() > 0.5
            has_pred = detection_pred[t] > self.detection_threshold
            
            if not (has_gt or has_pred):
                continue
            
            # get frame data
            original_frame = spike_sequence[0, t].cpu().numpy()
            foveated_frame = foveated_sequence[0, t].cpu().numpy()
            attention_map = attention_maps[t] if t < len(attention_maps) and attention_maps[t] is not None else None
            fovea_info = foveation_info[t] if t < len(foveation_info) and foveation_info[t] is not None else None
            
            # create event representations
            def create_event_img(spike_frame):
                event_img = np.zeros((spike_frame.shape[1], spike_frame.shape[2], 3))
                event_img[:, :, 0] = spike_frame[0]
                event_img[:, :, 2] = spike_frame[1]
                return event_img
            
            original_img = create_event_img(original_frame)
            foveated_img = create_event_img(foveated_frame)
            
            # create figure with 3 panels
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))
            
            # panel 1: foveated frame with gt and predictions
            axes[0].imshow(foveated_img)
            axes[0].set_title(f'frame {t}: foveated events + gt vs prediction', fontsize=14)
            
            # add gt if available
            if has_gt:
                gt_center = center_seq[t].numpy()
                gt_bbox = bbox_seq[t].numpy()
                
                rect_gt = patches.Rectangle(
                    (gt_bbox[0], gt_bbox[1]), 
                    gt_bbox[2] - gt_bbox[0], 
                    gt_bbox[3] - gt_bbox[1],
                    linewidth=3, edgecolor='lime', facecolor='none'
                )
                axes[0].add_patch(rect_gt)
                axes[0].plot(gt_center[0], gt_center[1], 'o', color='lime', markersize=8)
                
                gt_text = f'gt: ({gt_center[0]:.1f}, {gt_center[1]:.1f})'
                axes[0].text(10, 30, gt_text, color='lime', fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8))
            
            # add prediction if detected
            if has_pred:
                pred_center = center_pred[t]
                pred_bbox = bbox_pred[t]
                
                rect_pred = patches.Rectangle(
                    (pred_bbox[0], pred_bbox[1]), 
                    pred_bbox[2] - pred_bbox[0], 
                    pred_bbox[3] - pred_bbox[1],
                    linewidth=3, edgecolor='red', facecolor='none'
                )
                axes[0].add_patch(rect_pred)
                axes[0].plot(pred_center[0], pred_center[1], 's', color='red', markersize=8)
                
                pred_text = f'pred: ({pred_center[0]:.1f}, {pred_center[1]:.1f})'
                if has_gt:
                    axes[0].text(10, 60, pred_text, color='red', fontsize=12, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8))
                else:
                    axes[0].text(10, 30, pred_text, color='red', fontsize=12, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8))
            axes[0].axis('off')
            
            # Panel 2: Foveated frame with attention overlay
            axes[1].imshow(foveated_img)
            
            # Add attention mask overlay if available
            if attention_map is not None:
                axes[1].imshow(attention_map, cmap='hot', alpha=0.4, vmin=0, vmax=1)
            
            # Show event reduction info if available
            if fovea_info:
                event_reduction_text = f'Event Reduction: {fovea_info["event_reduction"]*100:.1f}%'
                axes[1].text(10, 40, event_reduction_text, color='yellow', fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.8))
                axes[1].set_title(f'Frame {t}: Attention Map', fontsize=14)
            else:
                axes[1].set_title(f'Frame {t}: No Foveation', fontsize=14)
            
            axes[1].axis('off')
            
            # Panel 3: Current predictions on original frame
            axes[2].imshow(original_img)
            
            title_parts = []
            
            if has_gt:
                title_parts.append("GT: ✓")
            else:
                title_parts.append("GT: ✗")
            
            if has_pred:
                # Current prediction visualization
                pred_center = center_pred[t]
                pred_bbox = bbox_pred[t]
                
                rect_pred = patches.Rectangle(
                    (pred_bbox[0], pred_bbox[1]), 
                    pred_bbox[2] - pred_bbox[0], 
                    pred_bbox[3] - pred_bbox[1],
                    linewidth=3, edgecolor='red', facecolor='none'
                )
                axes[2].add_patch(rect_pred)
                axes[2].plot(pred_center[0], pred_center[1], 'o', color='red', markersize=8)
                
                # Prediction info
                bbox_w = pred_bbox[2] - pred_bbox[0]
                bbox_h = pred_bbox[3] - pred_bbox[1]
                
                info_text = f'prediction:\n'
                info_text += f'center: ({pred_center[0]:.1f}, {pred_center[1]:.1f})\n'
                info_text += f'bbox: {bbox_w:.1f}x{bbox_h:.1f}px\n'
                info_text += f'confidence: {detection_pred[t]:.3f}\n'
                
                # add iou if gt exists
                if has_gt:
                    gt_bbox = bbox_seq[t].numpy()
                    iou = self.compute_iou(pred_bbox, gt_bbox)
                    info_text += f'iou: {iou:.3f}\n'
                
                if fovea_info:
                    info_text += f'foveation: yes\n'
                    info_text += f'reduction: {fovea_info["event_reduction"]*100:.1f}%'
                else:
                    info_text += f'foveation: no'
                
                axes[2].text(10, 80, info_text, color='red', fontsize=11, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8))
                
                title_parts.append(f"pred: yes ({detection_pred[t]:.3f})")
            else:
                axes[2].text(10, 80, f'no detection\nconf: {detection_pred[t]:.3f}', 
                           color='gray', fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8))
                title_parts.append(f"pred: no ({detection_pred[t]:.3f})")
            
            axes[2].set_title(" | ".join(title_parts), fontsize=14, fontweight='bold')
            axes[2].axis('off')
            
            # add overall metrics if both gt and prediction exist
            if has_gt and has_pred:
                error = np.linalg.norm(pred_center - gt_center)
                iou = self.compute_iou(pred_bbox, gt_bbox)
                fig.suptitle(f'recording {recording_name} - frame {t} | error: {error:.1f}px | iou: {iou:.3f}', 
                           fontsize=16, fontweight='bold')
            else:
                fig.suptitle(f'recording {recording_name} - frame {t}', fontsize=16, fontweight='bold')
            
            # save
            plt.savefig(os.path.join(recording_dir, f'frame_{t:03d}_fixed_foveated.png'), 
                       dpi=100, bbox_inches='tight')
            plt.close()


def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"using device: {device}")
    
    model_path = "/Users/banika/Desktop/master_submission_code/best_model.pth"
    
    tracker = FoveatedTracker(model_path, device=device)
    
    print("fixed parameter foveated snn tracker")
    print(f"min attention: {tracker.min_attention:.1f}")
    print(f"falloff sharpness: {tracker.falloff_sharpness:.1f}")
    print(f"safe zone multiplier: {tracker.fovea_safe_zone_multiplier:.1f}x")
    
    results = tracker.evaluate_test_set()
    
    return results


if __name__ == "__main__":
    main()