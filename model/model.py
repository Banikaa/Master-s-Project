
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


class SNN_tracker_model(nn.Module):
    def __init__(self, input_channels=2, hidden_size=256):
        super().__init__()
        
        # image dimensions
        self.width = 304
        self.height = 240
        self.input_size = input_channels * self.width * self.height
        
        # spiking layers
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())
        
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())
        
        # output heads
        self.detection_head = nn.Linear(hidden_size, 1)
        self.center_head = nn.Linear(hidden_size, 2)
        self.size_head = nn.Linear(hidden_size, 2)
        self.velocity_head = nn.Linear(hidden_size, 2)  # For motion learning

        
    def forward(self, spike_sequence):
        batch_size, seq_len, channels, height, width = spike_sequence.shape
        
        # initialize membrane potentials once per epoch 
        mem_fc1 = self.lif1.init_leaky()
        mem_fc2 = self.lif2.init_leaky()
        mem_fc3 = self.lif3.init_leaky()
        
        # process sequence
        detection_outputs = []
        center_outputs = []
        size_outputs = []
        velocity_outputs = []
        bbox_outputs = []
        
        # load the varying size recording sequence
        for t in range(seq_len):
            spk_in = spike_sequence[:, t]
            spk_flat = spk_in.flatten(start_dim=1)
            
            # SNN processing
            cur_fc1 = self.fc1(spk_flat)
            spk_fc1, mem_fc1 = self.lif1(cur_fc1, mem_fc1)
            
            cur_fc2 = self.fc2(spk_fc1)
            spk_fc2, mem_fc2 = self.lif2(cur_fc2, mem_fc2)
            
            cur_fc3 = self.fc3(spk_fc2)
            spk_fc3, mem_fc3 = self.lif3(cur_fc3, mem_fc3)
            
            # generate outputs
            detection = torch.sigmoid(self.detection_head(spk_fc3))
            center = torch.sigmoid(self.center_head(spk_fc3))
            size = torch.sigmoid(self.size_head(spk_fc3))
            velocity = torch.tanh(self.velocity_head(spk_fc3))  # [-1, 1] range
            
            # derive bbox from center + size
            cx, cy = center[:, 0], center[:, 1]
            w, h = size[:, 0], size[:, 1]
            
            x1 = torch.clamp(cx - w/2, 0, 1)
            y1 = torch.clamp(cy - h/2, 0, 1)
            x2 = torch.clamp(cx + w/2, 0, 1)
            y2 = torch.clamp(cy + h/2, 0, 1)
            
            bbox = torch.stack([x1, y1, x2, y2], dim=1)
            
            detection_outputs.append(detection)
            center_outputs.append(center)
            size_outputs.append(size)
            velocity_outputs.append(velocity)
            bbox_outputs.append(bbox)
        
        #  outputs
        detection_outputs = torch.stack(detection_outputs, dim=1)
        center_outputs = torch.stack(center_outputs, dim=1)
        size_outputs = torch.stack(size_outputs, dim=1)
        velocity_outputs = torch.stack(velocity_outputs, dim=1)
        bbox_outputs = torch.stack(bbox_outputs, dim=1)
        
        return detection_outputs, center_outputs, size_outputs, velocity_outputs, bbox_outputs
