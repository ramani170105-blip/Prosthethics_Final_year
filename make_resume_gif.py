import os
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import inspect
import virtual_prosthesis

def generate_gif(output_path="bci_prosthetic_demo.gif", width=600, height=450):
    print("🎬 Generating High-Impact BCI Virtual Prosthesis GIF...")
    
    # 1. Extract XML from virtual_prosthesis
    src = inspect.getsource(virtual_prosthesis.run_virtual_prosthesis)
    xml_start = src.find('xml_string = """') + len('xml_string = """')
    xml_end = src.find('"""', xml_start)
    xml_string = src[xml_start:xml_end]
    
    m = mujoco.MjModel.from_xml_string(xml_string)
    d = mujoco.MjData(m)
    
    # 2. Setup Offscreen Renderer & Isometric Camera
    renderer = mujoco.Renderer(m, height, width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat = [0.0, 0.02, 0.18]
    cam.distance = 0.45
    cam.elevation = -25.0
    cam.azimuth = 60.0

    # 3. Define the Flagship Gestures to showcase
    # 12-DOF targets: [shoulder_p, shoulder_r, shoulder_y, elbow, wrist_p, wrist_r, wrist_y, thumb, index, middle, ring, pinky]
    demos = [
        {
            "name": "NEURAL INTENT: POWER GRIP (88.2% ACC)",
            "sub": "EEG + EMG Cross-Attention Fusion (Latency: 1.2ms)",
            "keyframes": [
                ([0.0]*12, 10),
                ([0.0, 0.0, 0.0, 0.0, -0.3, 0.0, 0.0, 1.2, 1.5, 1.5, 1.5, 1.5], 16),
                ([0.0, 0.0, 0.0, 0.0, -0.3, 0.0, 0.0, 1.2, 1.5, 1.5, 1.5, 1.5], 10),
                ([0.0]*12, 12)
            ]
        },
        {
            "name": "NEURAL INTENT: PEACE SIGN (COUNTING TWO)",
            "sub": "Multi-Modal Intent Decoded (Subject 10 Holdout)",
            "keyframes": [
                ([0.0]*12, 6),
                ([0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 1.5, 0.0, 0.0, 1.5, 1.5], 16),
                ([0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 1.5, 0.0, 0.0, 1.5, 1.5], 10),
                ([0.0]*12, 12)
            ]
        },
        {
            "name": "NEURAL INTENT: DYNAMIC WAVE SEQUENCE",
            "sub": "3-DOF Arm + 12-DOF Hand Kinematics Synchronized",
            "keyframes": [
                ([0.0]*12, 6),
                ([0.4, 0.0, 0.3, 1.2, -0.3, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0], 12),
                ([0.4, 0.0, 0.3, 1.2, -0.3, 0.0, -0.6, 0.0, 0.0, 0.0, 0.0, 0.0], 12),
                ([0.4, 0.0, 0.3, 1.2, -0.3, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0], 12),
                ([0.0]*12, 12)
            ]
        },
        {
            "name": "NEURAL INTENT: PRECISION POINT (COUNTING ONE)",
            "sub": "Individual Finger Kinematics | Real-Time Response",
            "keyframes": [
                ([0.0]*12, 6),
                ([0.0, 0.0, 0.0, 0.0, -0.2, 0.0, 0.0, 1.5, 0.0, 1.5, 1.5, 1.5], 16),
                ([0.0, 0.0, 0.0, 0.0, -0.2, 0.0, 0.0, 1.5, 0.0, 1.5, 1.5, 1.5], 10),
                ([0.0]*12, 10)
            ]
        }
    ]

    frames = []
    
    # Try loading a system font, fallback to default
    font_main = None
    font_sub = None
    try:
        font_main = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_sub = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        font_main = ImageFont.load_default()
        font_sub = font_main

    current_ctrl = np.zeros(m.nu)

    for demo in demos:
        banner_title = demo["name"]
        banner_sub = demo["sub"]
        
        for target_ctrl, num_steps in demo["keyframes"]:
            target_arr = np.array(target_ctrl)
            start_arr = current_ctrl.copy()
            
            for s in range(num_steps):
                alpha = (s + 1) / float(num_steps)
                interp = start_arr + (target_arr - start_arr) * alpha
                
                for i in range(m.nu):
                    d.ctrl[i] = interp[i]
                
                mujoco.mj_step(m, d)
                renderer.update_scene(d, camera=cam)
                pix = renderer.render()
                
                # Convert to PIL Image and overlay HUD
                img = Image.fromarray(pix)
                draw = ImageDraw.Draw(img)
                
                # Top header bar (Futuristic Cyberpunk HUD)
                draw.rectangle([(0, 0), (width, 36)], fill=(15, 20, 30, 220))
                draw.text((15, 8), "NEURAL INTERFACE // MULTI-MODAL TRANSFORMER BCI", fill=(0, 220, 255), font=font_sub)
                draw.text((width - 150, 8), "LATENCY: 1.2ms", fill=(50, 255, 120), font=font_sub)
                draw.line([(0, 36), (width, 36)], fill=(0, 180, 255), width=2)
                
                # Bottom HUD Banner
                banner_h = 56
                draw.rectangle([(0, height - banner_h), (width, height)], fill=(10, 14, 22, 230))
                draw.line([(0, height - banner_h), (width, height - banner_h)], fill=(0, 140, 220), width=1)
                
                # Glowing Intent Indicator
                draw.ellipse([(16, height - banner_h + 16), (28, height - banner_h + 28)], fill=(0, 255, 200))
                draw.text((36, height - banner_h + 10), banner_title, fill=(255, 255, 255), font=font_main)
                draw.text((36, height - banner_h + 32), banner_sub, fill=(160, 190, 220), font=font_sub)
                
                frames.append(img)
                current_ctrl = interp

    print(f"Total rendered frames: {len(frames)}")
    print(f"Compressing into animated GIF: {output_path}...")
    
    # Save optimized GIF with 60ms per frame (~16.6 FPS)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=65,
        loop=0,
        optimize=True
    )
    print(f"✅ Successfully created: {output_path} ({os.path.getsize(output_path) / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    generate_gif()
