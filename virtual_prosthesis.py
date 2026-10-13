import torch
import mujoco
import mujoco.viewer
import time
import numpy as np
from sklearn.model_selection import train_test_split

from models.transformer_fusion import MultiModalCrossAttention
from run_transformer import load_or_preprocess_data

import sys

def run_virtual_prosthesis():
    print("Loading PyTorch Transformer Models for ALL Subjects...")
    
    # 1. Initialize lists for testing data
    X_eeg_test_all, X_emg_test_all, y_test_all, sub_ids_test_all = [], [], [], []
    
    device = torch.device("cpu")
    models = {}
    
    # We will load subjects individually to prevent RAM crashes!
    import os
    import glob
    cache_dir = "datasets/preprocessed_cache"
    
    # Automatically check all subjects 1 to 25
    for s in range(1, 26):
        eeg_cache = os.path.join(cache_dir, f"sub{s}_eeg.npy")
        emg_cache = os.path.join(cache_dir, f"sub{s}_emg.npy")
        y_cache = os.path.join(cache_dir, f"sub{s}_y.npy")
        
        if not (os.path.exists(eeg_cache) and os.path.exists(emg_cache) and os.path.exists(y_cache)):
            continue
            
        # Load individual subject data (1.9 GB max)
        X_e_s = np.load(eeg_cache)
        X_m_s = np.load(emg_cache)
        y_s = np.load(y_cache)
        
        model_path = f'best_bci_model_sub{s}.pth'
        if not os.path.exists(model_path):
            continue

        # Detect the exact classes the model was trained on
        classes_cache = f'best_bci_model_sub{s}_classes.npy'
        if os.path.exists(classes_cache):
            unique_labels = np.load(classes_cache)
        else:
            # Fallback: inspect the saved model weights directly to get num_classes
            sd = torch.load(model_path, map_location=device)
            saved_num_classes = sd['classifier.3.weight'].shape[0]
            u_classes, counts = np.unique(y_s, return_counts=True)
            if saved_num_classes < len(u_classes):
                valid_classes = u_classes[counts >= 5]
                unique_labels = valid_classes
            else:
                unique_labels = u_classes

        num_classes = len(unique_labels)
        
        # Filter data to classes recognized by this model
        v_mask = np.isin(y_s, unique_labels)
        X_e_s, X_m_s, y_s = X_e_s[v_mask], X_m_s[v_mask], y_s[v_mask]
        
        label_map = {val: idx for idx, val in enumerate(unique_labels)}
        y_s_mapped = np.array([label_map[val] for val in y_s])
        
        # Extract 15% Test Set safely
        u_filt, c_filt = np.unique(y_s_mapped, return_counts=True)
        can_stratify = (len(c_filt) > 1 and np.min(c_filt) >= 2)
        
        _, X_e_te, _, X_m_te, _, y_te = train_test_split(
            X_e_s, X_m_s, y_s_mapped, test_size=0.15, random_state=42, stratify=y_s_mapped if can_stratify else None
        )
        
        # Free memory by deleting the 85% train/val set
        del X_e_s, X_m_s, y_s, y_s_mapped
        
        try:
            m_instance = MultiModalCrossAttention(num_classes=num_classes).to(device)
            m_instance.load_state_dict(torch.load(model_path, map_location=device))
            m_instance.eval()
            models[s] = (m_instance, unique_labels)
            
            # Only add to test set if model loaded successfully
            X_eeg_test_all.append(X_e_te)
            X_emg_test_all.append(X_m_te)
            y_test_all.append(y_te)
            sub_ids_test_all.append(np.full(len(y_te), s))
            print(f"✅ Loaded Subject {s} Model ({num_classes} classes)")
        except Exception as e:
            print(f"❌ Skipping Subject {s} (Load Error: {e})")

    if not models:
        print("No models found! Run python3 run_transformer.py first.")
        return
        
    # 4. Combine and Shuffle all valid test data across all subjects
    X_eeg_te = np.concatenate(X_eeg_test_all)
    X_emg_te = np.concatenate(X_emg_test_all)
    y_te = np.concatenate(y_test_all)
    sub_ids_te = np.concatenate(sub_ids_test_all)
    
    shuffle_idx = np.random.permutation(len(y_te))
    X_eeg_te = X_eeg_te[shuffle_idx]
    X_emg_te = X_emg_te[shuffle_idx]
    y_te = y_te[shuffle_idx]
    sub_ids_te = sub_ids_te[shuffle_idx]

    # 3. Native MuJoCo XML for a Floating VR-Style Holographic Hand
    xml_string = """
    <mujoco>
      <option gravity="0 0 0" timestep="0.01"/> <!-- Zero gravity for floating VR hand -->
      <worldbody>
        <light pos="0 1 1" dir="0 -1 -1" diffuse="1 1 1" ambient="0.5 0.5 0.5"/>
        <light pos="0 -1 1" dir="0 1 -1" diffuse="0.2 0.5 1.0"/> <!-- Glowing blue backlight -->
        
        <!-- White Studio Backdrop -->
        <geom type="plane" size="5 5 0.1" pos="0 1 0" euler="90 0 0" rgba="1 1 1 1" contype="0" conaffinity="0"/>
        <geom type="plane" size="5 5 0.1" pos="0 0 -0.5" rgba="1 1 1 1" contype="0" conaffinity="0"/>
        
        <!-- Full Arm System -->
        <body name="shoulder" pos="0 -0.4 0.2">
          <!-- 3-DOF Shoulder -->
          <joint name="shoulder_pitch" type="hinge" axis="1 0 0" range="-1.5 1.5" damping="2"/>
          <joint name="shoulder_roll" type="hinge" axis="0 1 0" range="-1.5 1.5" damping="2"/>
          <joint name="shoulder_yaw" type="hinge" axis="0 0 1" range="-1.5 1.5" damping="2"/>
          
          <!-- Upper Arm (Cyberpunk Hologram) -->
          <geom type="capsule" size="0.02 0.1" pos="0 0.1 0" rgba="0.1 0.1 0.1 0.7" euler="90 0 0" contype="0" conaffinity="0"/>
          <geom type="sphere" size="0.025" pos="0 0 0" rgba="0 0.8 1 0.8" contype="0" conaffinity="0"/> <!-- Shoulder Glow -->

          <!-- Elbow -->
          <body name="elbow" pos="0 0.2 0">
            <!-- 1-DOF Elbow (Flexion/Extension) -->
            <joint name="elbow_flexion" type="hinge" axis="1 0 0" range="0 2.5" damping="2"/>
            <geom type="sphere" size="0.02" pos="0 0 0" rgba="0 0.8 1 0.8" contype="0" conaffinity="0"/> <!-- Elbow Glow -->
            
            <!-- Ulna Bone (Stationary relative to elbow flexion) -->
            <geom type="capsule" size="0.01 0.1" pos="0.015 0.1 0" rgba="0.1 0.1 0.1 0.7" euler="90 0 0" contype="0" conaffinity="0"/>
            
            <!-- Proximal Radioulnar Joint (Pronation/Supination) -->
            <body name="forearm_radius" pos="0 0 0">
              <joint name="wrist_roll" type="hinge" axis="0 1 0" range="-2 2" damping="1"/>
              
              <!-- Radius Bone (Twists around the Y-axis, crossing the Ulna) -->
              <geom type="capsule" size="0.01 0.1" pos="-0.015 0.1 0" rgba="0.1 0.1 0.1 0.7" euler="90 0 0" contype="0" conaffinity="0"/>
              
              <!-- VR Wrist Base -->
              <body name="wrist_base" pos="0 0.2 0">
                <!-- Invisible anchor -->
                <geom type="sphere" size="0.001" rgba="0 0 0 0" contype="0" conaffinity="0"/>
            
                <!-- 2-DOF Wrist (Pitch, Yaw) -->
                <joint name="wrist_pitch" type="hinge" axis="-1 0 0" range="-1.5 1.5" damping="1"/> <!-- Extension / Flexion -->
                <joint name="wrist_yaw" type="hinge" axis="0 0 1" range="-1.5 1.5" damping="1"/> <!-- Left / Right Waving -->
                
                <!-- Floating VR Palm -->
          <body name="palm" pos="0 0 0">
          <!-- Sleek futuristic palm -->
          <geom type="box" size="0.04 0.05 0.01" rgba="0.1 0.1 0.1 0.9" contype="0" conaffinity="0"/>
          <geom type="sphere" size="0.015" pos="0 -0.05 0" rgba="0 0.8 1 1" contype="0" conaffinity="0"/> <!-- Glowing Wrist core -->
          
          <!-- Thumb (2 segments) -->
          <body name="thumb_proximal" pos="-0.04 -0.01 0" euler="0 0 50">
            <joint name="thumb_j1" type="hinge" axis="-1 0 0" range="0 1.5" damping="0.5"/>
            <geom type="capsule" size="0.008 0.02" pos="0 0.02 0" rgba="0.2 0.2 0.2 0.9" euler="90 0 0" contype="0" conaffinity="0"/>
            <geom type="sphere" size="0.009" pos="0 0.04 0" rgba="0 0.8 1 0.8" contype="0" conaffinity="0"/> <!-- Joint glow -->
            
            <body name="thumb_distal" pos="0 0.04 0">
              <joint name="thumb_j2" type="hinge" axis="-1 0 0" range="0 1.5" damping="0.5"/>
              <geom type="capsule" size="0.007 0.015" pos="0 0.015 0" rgba="0.2 0.2 0.2 0.9" euler="90 0 0" contype="0" conaffinity="0"/>
            </body>
          </body>

          <!-- Index Finger (3 segments) -->
          <body name="index_proximal" pos="-0.03 0.05 0">
            <joint name="index_j1" type="hinge" axis="-1 0 0" range="0 1.5" damping="0.5"/>
            <geom type="capsule" size="0.008 0.02" pos="0 0.02 0" rgba="0.2 0.2 0.2 0.9" euler="90 0 0" contype="0" conaffinity="0"/>
            <geom type="sphere" size="0.009" pos="0 0.04 0" rgba="0 0.8 1 0.8" contype="0" conaffinity="0"/>
            
            <body name="index_middle" pos="0 0.04 0">
              <joint name="index_j2" type="hinge" axis="-1 0 0" range="0 1.5" damping="0.5"/>
              <geom type="capsule" size="0.007 0.015" pos="0 0.015 0" rgba="0.2 0.2 0.2 0.9" euler="90 0 0" contype="0" conaffinity="0"/>
              <geom type="sphere" size="0.008" pos="0 0.03 0" rgba="0 0.8 1 0.8" contype="0" conaffinity="0"/>
              
              <body name="index_distal" pos="0 0.03 0">
                <joint name="index_j3" type="hinge" axis="-1 0 0" range="0 1.5" damping="0.5"/>
                <geom type="capsule" size="0.006 0.015" pos="0 0.015 0" rgba="0.2 0.2 0.2 0.9" euler="90 0 0" contype="0" conaffinity="0"/>
              </body>
            </body>
          </body>
          
          <!-- Middle Finger -->
          <body name="middle_proximal" pos="-0.01 0.05 0">
            <joint name="middle_j1" type="hinge" axis="-1 0 0" range="0 1.5" damping="0.5"/>
            <geom type="capsule" size="0.008 0.022" pos="0 0.022 0" rgba="0.2 0.2 0.2 0.9" euler="90 0 0" contype="0" conaffinity="0"/>
            <geom type="sphere" size="0.009" pos="0 0.044 0" rgba="0 0.8 1 0.8" contype="0" conaffinity="0"/>
            
            <body name="middle_middle" pos="0 0.044 0">
              <joint name="middle_j2" type="hinge" axis="-1 0 0" range="0 1.5" damping="0.5"/>
              <geom type="capsule" size="0.007 0.018" pos="0 0.018 0" rgba="0.2 0.2 0.2 0.9" euler="90 0 0" contype="0" conaffinity="0"/>
              <geom type="sphere" size="0.008" pos="0 0.036 0" rgba="0 0.8 1 0.8" contype="0" conaffinity="0"/>
              
              <body name="middle_distal" pos="0 0.036 0">
                <joint name="middle_j3" type="hinge" axis="-1 0 0" range="0 1.5" damping="0.5"/>
                <geom type="capsule" size="0.006 0.015" pos="0 0.015 0" rgba="0.2 0.2 0.2 0.9" euler="90 0 0" contype="0" conaffinity="0"/>
              </body>
            </body>
          </body>

          <!-- Ring Finger -->
          <body name="ring_proximal" pos="0.01 0.05 0">
            <joint name="ring_j1" type="hinge" axis="-1 0 0" range="0 1.5" damping="0.5"/>
            <geom type="capsule" size="0.008 0.02" pos="0 0.02 0" rgba="0.2 0.2 0.2 0.9" euler="90 0 0" contype="0" conaffinity="0"/>
            <geom type="sphere" size="0.009" pos="0 0.04 0" rgba="0 0.8 1 0.8" contype="0" conaffinity="0"/>
            
            <body name="ring_middle" pos="0 0.04 0">
              <joint name="ring_j2" type="hinge" axis="-1 0 0" range="0 1.5" damping="0.5"/>
              <geom type="capsule" size="0.007 0.015" pos="0 0.015 0" rgba="0.2 0.2 0.2 0.9" euler="90 0 0" contype="0" conaffinity="0"/>
              <geom type="sphere" size="0.008" pos="0 0.03 0" rgba="0 0.8 1 0.8" contype="0" conaffinity="0"/>
              
              <body name="ring_distal" pos="0 0.03 0">
                <joint name="ring_j3" type="hinge" axis="-1 0 0" range="0 1.5" damping="0.5"/>
                <geom type="capsule" size="0.006 0.015" pos="0 0.015 0" rgba="0.2 0.2 0.2 0.9" euler="90 0 0" contype="0" conaffinity="0"/>
              </body>
            </body>
          </body>

          <!-- Pinky Finger -->
          <body name="pinky_proximal" pos="0.03 0.05 0">
            <joint name="pinky_j1" type="hinge" axis="-1 0 0" range="0 1.5" damping="0.5"/>
            <geom type="capsule" size="0.007 0.015" pos="0 0.015 0" rgba="0.2 0.2 0.2 0.9" euler="90 0 0" contype="0" conaffinity="0"/>
            <geom type="sphere" size="0.008" pos="0 0.03 0" rgba="0 0.8 1 0.8" contype="0" conaffinity="0"/>
            
            <body name="pinky_middle" pos="0 0.03 0">
              <joint name="pinky_j2" type="hinge" axis="-1 0 0" range="0 1.5" damping="0.5"/>
              <geom type="capsule" size="0.006 0.01" pos="0 0.01 0" rgba="0.2 0.2 0.2 0.9" euler="90 0 0" contype="0" conaffinity="0"/>
              <geom type="sphere" size="0.007" pos="0 0.02 0" rgba="0 0.8 1 0.8" contype="0" conaffinity="0"/>
              
              <body name="pinky_distal" pos="0 0.02 0">
                <joint name="pinky_j3" type="hinge" axis="-1 0 0" range="0 1.5" damping="0.5"/>
                <geom type="capsule" size="0.005 0.01" pos="0 0.01 0" rgba="0.2 0.2 0.2 0.9" euler="90 0 0" contype="0" conaffinity="0"/>
              </body>
            </body>
          </body>
          </body> <!-- closes palm -->
        </body> <!-- closes wrist_base -->
      </body> <!-- closes forearm_radius -->
    </body> <!-- closes elbow -->
  </body> <!-- closes shoulder -->
</worldbody>
      
      <equality>
        <!-- Lock joints together so fingers curl naturally via MuJoCo physics -->
        <joint joint1="thumb_j1" joint2="thumb_j2" polycoef="0 1 0 0 0"/>
        
        <joint joint1="index_j1" joint2="index_j2" polycoef="0 1 0 0 0"/>
        <joint joint1="index_j2" joint2="index_j3" polycoef="0 1 0 0 0"/>
        
        <joint joint1="middle_j1" joint2="middle_j2" polycoef="0 1 0 0 0"/>
        <joint joint1="middle_j2" joint2="middle_j3" polycoef="0 1 0 0 0"/>
        
        <joint joint1="ring_j1" joint2="ring_j2" polycoef="0 1 0 0 0"/>
        <joint joint1="ring_j2" joint2="ring_j3" polycoef="0 1 0 0 0"/>
        
        <joint joint1="pinky_j1" joint2="pinky_j2" polycoef="0 1 0 0 0"/>
        <joint joint1="pinky_j2" joint2="pinky_j3" polycoef="0 1 0 0 0"/>
      </equality>
      
      <actuator>
        <position joint="shoulder_pitch" name="shoulder_pitch_motor" kp="10"/>
        <position joint="shoulder_roll" name="shoulder_roll_motor" kp="10"/>
        <position joint="shoulder_yaw" name="shoulder_yaw_motor" kp="10"/>
        <position joint="elbow_flexion" name="elbow_motor" kp="10"/>
        
        <position joint="wrist_pitch" name="pitch_motor" kp="5"/>
        <position joint="wrist_roll" name="roll_motor" kp="5"/>
        <position joint="wrist_yaw" name="yaw_motor" kp="5"/>
        <position joint="thumb_j1" name="thumb_motor" kp="2"/>
        <position joint="index_j1" name="index_motor" kp="2"/>
        <position joint="middle_j1" name="middle_motor" kp="2"/>
        <position joint="ring_j1" name="ring_motor" kp="2"/>
        <position joint="pinky_j1" name="pinky_motor" kp="2"/>
      </actuator>
    </mujoco>
    """
    
    print("Loading Native MuJoCo Robotic Gripper...")
    try:
        m = mujoco.MjModel.from_xml_string(xml_string)
        d = mujoco.MjData(m)
    except Exception as e:
        print(f"Error compiling XML: {e}")
        return

    # 4. Launch Viewer and Run Inference
    print("\n===========================================================")
    print("  INTERACTIVE VIRTUAL PROSTHESIS (12-DOF UPPER LIMB & HAND)")
    print("===========================================================")
    print("💪 UPPER LIMB COMMANDS:")
    print("   • 'reach' / 'reach and grasp' / 'drink' / 'curl' (elbow)")
    print("   • 'rotate' (pronation/supination) / 'overhead' / 'salute'")
    print("🖐️ HAND & FINGER COMMANDS:")
    print("   • 'open', 'close', 'power', 'point', 'peace', 'gun', 'shaka'")
    print("🧠 BRAINWAVE INFERENCE:")
    print("   • Press [ENTER] to decode live Holdout EEG+EMG Brainwaves")
    print("   • Type 'quit' to exit")
    print("===========================================================\n")
    
    # Map ALL 41 Possible Classes (Jeong2020 GigaScience Dataset)
    # Target Actuation Array: [Wrist_Wave, Wrist_Roll, Thumb, Index, Middle, Ring, Pinky]
    grasp_taxonomy = {
        0: ("Rest (Open Hand)", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        1: ("Counting One (Point)", [0.0, 0.0, 1.5, 0.0, 1.5, 1.5, 1.5]),
        2: ("Counting Two (Peace) Sequence", [
            [-1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Step 1: Wrist Extension (Back)
            [-1.2, 0.0, 0.0, 1.5, 0.0, 0.0, 1.5, 1.5], # Step 2: Form Peace Sign
            [-1.2, 0.0, 0.0, 1.5, 0.0, 0.0, 1.5, 1.5], # Step 3: Hold
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # Step 4: Rest
        ]),
        3: ("Counting Three", [0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.5]),
        4: ("Counting Four", [0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0]),
        5: ("Counting Five (Spread)", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        6: ("Counting Six (ASL)", [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        7: ("Counting Seven (ASL)", [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]),
        8: ("Counting Eight (ASL)", [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]),
        9: ("Counting Nine (ASL)", [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        10: ("Counting Ten (Thumbs Up)", [-1.2, 0.0, 0.0, 1.5, 1.5, 1.5, 1.5]),
        11: ("Supination (Palm Up)", [0.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
        12: ("Pronation (Palm Down)", [0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
        13: ("Hi / Wave Sequence", [
            # [Shoulder_P, Shoulder_R, Shoulder_Y, Elbow, Wrist_P, Wrist_R, Wrist_Y, Thumb, Index, Middle, Ring, Pinky]
            [0.5, 0.0, 0.5, 1.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Lift arm, bend elbow, wrist up
            [0.5, 0.0, 0.5, 1.5, -0.5, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0], # Wave left
            [0.5, 0.0, 0.5, 1.5, -0.5, 0.0, -0.8, 0.0, 0.0, 0.0, 0.0, 0.0], # Wave right
            [0.5, 0.0, 0.5, 1.5, -0.5, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0], # Wave left
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # Rest (Lower arm)
        ]),
        14: ("Power Grip (Cylindrical)", [0.0, 0.0, 1.2, 1.5, 1.5, 1.5, 1.5]),
        15: ("Precision Pinch (Index-Thumb)", [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        16: ("Tripod Pinch", [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
        17: ("Lateral Pinch (Key Grip)", [0.0, 0.0, 1.2, 1.0, 0.0, 0.0, 0.0]),
        18: ("Hook Grip", [0.0, 0.0, 0.0, 1.5, 1.5, 1.5, 1.5]),
        19: ("Spherical Grip", [0.0, 0.0, 0.8, 1.2, 1.2, 1.2, 0.8]),
        20: ("Lumbrical Grip", [0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0]),
        21: ("OK Sign", [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        22: ("Spider / Claw", [0.0, 0.0, 0.8, 0.8, 0.8, 0.8, 0.8]),
        23: ("Finger Gun Sequence", [
            [0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Step 1: Pronation
            [0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 1.5, 1.5], # Step 2: Form Gun (Index & Middle)
            [0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 1.5, 1.5], # Step 3: Hold
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Step 4: Rest
        ]),
        24: ("Middle Finger", [0.0, 0.0, 1.5, 1.5, 0.0, 1.5, 1.5]),
        25: ("Hang Loose (Shaka)", [0.0, 0.0, 0.0, 1.5, 1.5, 1.5, 0.0]),
        26: ("Rock On (Horns)", [0.0, 0.0, 1.5, 0.0, 1.5, 1.5, 0.0]),
        27: ("Pinky Precision Grip", [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        28: ("Ring Precision Grip", [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]),
        29: ("Closed Fist", [0.0, 0.0, 1.5, 1.5, 1.5, 1.5, 1.5]),
        30: ("Adduction (Fingers Together)", [0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2]),
        31: ("Abduction (Fingers Spread)", [0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1]),
        32: ("Pinch (Thumb, Middle)", [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]),
        33: ("Half-closed Fist", [0.0, 0.0, 0.8, 0.8, 0.8, 0.8, 0.8]),
        34: ("Wrist Flexion", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        35: ("Wrist Extension", [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        36: ("Pinch (Thumb, Ring, Pinky)", [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]),
        37: ("Index and Pinky Extended", [0.0, 0.0, 1.5, 0.0, 1.5, 1.5, 0.0]),
        38: ("Thumb and Index Curled", [0.0, 0.0, 1.5, 1.5, 0.0, 0.0, 0.0]),
        39: ("Thumb, Index, Middle Curled", [0.0, 0.0, 1.5, 1.5, 1.5, 0.0, 0.0]),
        40: ("All Fingers Curled, Thumb Out", [-1.2, 0.0, 0.0, 1.5, 1.5, 1.5, 1.5]),
        41: ("Tight Fist (Max Flexion)", [0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    }

    # =================================================================
    # PRIMITIVE-FIRST DYNAMIC SEQUENCE GENERATOR (12-DOF FULL ARM)
    # Automatically convert ALL static poses into step-by-step primitive combinations
    # =================================================================
    for key, (name, act) in grasp_taxonomy.items():
        if isinstance(act[0], (int, float)):
            # Upgrade 7 or 8 element arrays to 12 elements
            if len(act) == 7:
                act.insert(2, 0.0)
            if len(act) == 8:
                act = [0.0, 0.0, 0.0, 0.0] + act
            
            # PRIMITIVE 1: Arm & Wrist Orientation (Limb positioning)
            step1 = [0.0] * 12
            step1[0:7] = act[0:7] # Apply target Arm/Wrist
            if step1[4] == 0.0:
                step1[4] = -0.4 # Organic wrist-extension wind-up if no wrist movement
                
            # PRIMITIVE 2: Thumb & Core Preparation (Forming the core shape)
            step2 = list(step1)
            step2[7] = act[7] # Move Thumb to final position
            step2[8] = act[8] * 0.5 # Curl Index halfway
            step2[9] = act[9] * 0.5 # Curl Middle halfway
            
            # PRIMITIVE 3: Full Hand Actuation (Fingers snap into final position)
            step3 = list(act)
            
            # Create the 5-step primitive combination sequence
            seq = [
                step1,                                    # Orient arm/wrist
                step2,                                    # Position thumb
                step3,                                    # Close fingers
                step3,                                    # Hold
                [0.0] * 12                                # Return to rest
            ]
            grasp_taxonomy[key] = (f"{name} (Primitive Combo)", seq)
        else:
            # For pre-made sequences like Hi/Wave, just upgrade them if they are old format
            new_seq = []
            for step in act:
                if len(step) == 7:
                    step.insert(2, 0.0)
                if len(step) == 8:
                    step = [0.0, 0.0, 0.0, 0.0] + step
                new_seq.append(step)
            grasp_taxonomy[key] = (name, new_seq)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        i = 0
        while viewer.is_running():
            cmd = input("\nEnter command (or press Enter for Brainwave): ").strip().lower()
            
            if cmd in ['quit', 'exit', 'q']:
                break
                
            # TEXT COMMAND MODE
            if cmd:
                found = False
                # Search through all 41 taxonomy classes by name or number
                for class_id, (g_name, g_act) in grasp_taxonomy.items():
                    if cmd in g_name.lower() or cmd == str(class_id):
                        target_actuation = g_act
                        grip_name = f"Text Command -> {g_name}"
                        found = True
                        break
                
                # Keep special functional fallbacks (Full 12-DOF Upper Limb & Hand Kinematics)
                if not found:
                    if 'drink' in cmd or 'cup' in cmd or 'water' in cmd:
                        target_actuation = [
                            [0.6, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # 1. Reach for cup
                            [0.6, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 1.2, 1.5, 1.5, 1.5, 1.5],    # 2. Grasp cup firmly
                            [0.1, 0.0, 0.2, 1.8, -0.3, 0.0, 0.0, 1.2, 1.5, 1.5, 1.5, 1.5],   # 3. Lift to mouth (Elbow flexion)
                            [0.1, 0.0, 0.2, 1.8, -0.3, 1.0, 0.0, 1.2, 1.5, 1.5, 1.5, 1.5],   # 4. Tilt cup to drink
                            [0.1, 0.0, 0.2, 1.8, -0.3, 0.0, 0.0, 1.2, 1.5, 1.5, 1.5, 1.5],   # 5. Untilt cup
                            [0.6, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 1.2, 1.5, 1.5, 1.5, 1.5],    # 6. Lower cup to table
                            [0.6, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # 7. Release cup
                            [0.0] * 12                                                         # 8. Return to rest
                        ]
                        grip_name = "Upper Limb Sequence -> Reach, Grasp & Drink Water"

                    elif 'pick' in cmd or 'grab' in cmd or ('reach' in cmd and 'grasp' in cmd):
                        target_actuation = [
                            [0.6, 0.0, 0.0, 0.3, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # 1. Reach forward, hand open
                            [0.85, 0.0, 0.0, 0.1, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 2. Target acquisition
                            [0.85, 0.0, 0.0, 0.1, -0.2, 0.0, 0.0, 1.2, 1.5, 1.5, 1.5, 1.5],  # 3. Power Grip closes on object
                            [0.3, 0.0, 0.0, 1.4, -0.2, 0.0, 0.0, 1.2, 1.5, 1.5, 1.5, 1.5],   # 4. Retract arm, holding object
                            [0.3, 0.0, 0.0, 1.4, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # 5. Release object
                            [0.0] * 12                                                         # 6. Return to rest
                        ]
                        grip_name = "Upper Limb Sequence -> Reach, Grasp Object & Retract"

                    elif 'up' in cmd or 'overhead' in cmd or 'raise' in cmd:
                        target_actuation = [
                            [0.7, 0.3, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.3, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # Overhead reach
                            [1.3, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # Hold
                            [0.0] * 12
                        ]
                        grip_name = "Upper Limb Sequence -> Overhead Arm Elevation"

                    elif 'reach' in cmd or 'forward' in cmd:
                        target_actuation = [
                            [0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.85, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # Forward reach
                            [0.85, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # Hold
                            [0.0] * 12
                        ]
                        grip_name = "Upper Limb Sequence -> Forward Arm Reach"

                    elif 'curl' in cmd or 'bicep' in cmd or 'elbow' in cmd:
                        target_actuation = [
                            [0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # Full elbow flexion (105 deg)
                            [0.0, 0.0, 0.0, 1.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0] * 12
                        ]
                        grip_name = "Upper Limb Sequence -> Elbow Flexion (Bicep Curl)"

                    elif 'rot' in cmd or 'twist' in cmd or 'roat' in cmd:
                        target_actuation = [
                            [0.0, 0.0, 0.0, 0.6, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # Full Pronation
                            [0.0, 0.0, 0.0, 0.6, 0.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # Full Supination
                            [0.0, 0.0, 0.0, 0.6, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # Pronation
                            [0.0] * 12
                        ]
                        grip_name = "Upper Limb Sequence -> Forearm Rotation & Twist (Pronation/Supination)"

                    elif 'salute' in cmd:
                        target_actuation = [
                            [0.5, 0.4, 0.0, 1.9, -0.2, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.5, 0.4, 0.0, 1.9, -0.2, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0] * 12
                        ]
                        grip_name = "Upper Limb Sequence -> Military Salute"

                    elif 'open' in cmd or 'rest' in cmd:
                        target_actuation = grasp_taxonomy[0][1]
                        grip_name = "Text Command -> Open Hand Combo"
                    elif 'close' in cmd:
                        target_actuation = grasp_taxonomy[14][1]
                        grip_name = "Text Command -> Power Grip Combo"
                    elif 'supination' in cmd:
                        target_actuation = grasp_taxonomy[11][1]
                        grip_name = "Text Command -> Supination Combo"
                    elif 'pronation' in cmd:
                        target_actuation = grasp_taxonomy[12][1]
                        grip_name = "Text Command -> Pronation Combo"
                    elif 'wave' in cmd or 'hi' in cmd or 'hello' in cmd:
                        target_actuation = grasp_taxonomy[13][1]
                        grip_name = "Upper Limb Sequence -> Dynamic Wave (Shoulder + Elbow + Wrist)"
                    else:
                        target_actuation = grasp_taxonomy[0][1]
                        grip_name = f"Text Command -> Default Open (Unrecognized: {cmd})"
            
            # BRAINWAVE INFERENCE MODE
            else:
                if i >= len(X_eeg_te):
                    print("Out of test samples! Restart the script.")
                    break
                    
                s_id = sub_ids_te[i]
                eeg_tensor = torch.tensor(X_eeg_te[i:i+1], dtype=torch.float32).to(device)
                emg_tensor = torch.tensor(X_emg_te[i:i+1], dtype=torch.float32).to(device)
                
                # Dynamically route to the correct Personalized AI Model
                model_s, unique_labels_s = models[s_id]
                
                with torch.no_grad():
                    output = model_s(eeg_tensor, emg_tensor)
                    predicted_class = torch.argmax(output, dim=1).item()
                    predicted_original_label = unique_labels_s[predicted_class]
                
                if predicted_original_label in grasp_taxonomy:
                    grip_name, target_actuation = grasp_taxonomy[predicted_original_label]
                    grip_name = f"[Subject {s_id}] Brainwave Intent -> {grip_name}"
                else:
                    grip_name = f"[Subject {s_id}] Brainwave Intent -> Unknown Grip {predicted_original_label}"
                    target_actuation = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                
                i += 1
                
            print(f"Executing: {grip_name}")
            
            # Normalize target_actuation to always be a sequence of steps
            if isinstance(target_actuation[0], (int, float)):
                sequence = [target_actuation]
                animation_steps = 150 # 1.5s per step for single grips
            else:
                sequence = target_actuation
                animation_steps = 60 # 0.6s per step for fast fluid sequences
            
            # 5. Slow & Smooth Linear Interpolation Animation
            for seq_target in sequence:
                current_actuation = np.array([d.ctrl[idx] for idx in range(m.nu)])
                target_act = np.array(seq_target)
                
                for step in range(animation_steps):
                    if not viewer.is_running():
                        break
                    
                    # Calculate interpolated position for this step
                    progress = (step + 1) / float(animation_steps)
                    interp = current_actuation + (target_act - current_actuation) * progress
                    
                    for act_idx in range(m.nu):
                        d.ctrl[act_idx] = interp[act_idx]
                        
                    mujoco.mj_step(m, d)
                    viewer.sync()
                    time.sleep(0.01) # 10ms per frame

if __name__ == "__main__":
    run_virtual_prosthesis()
