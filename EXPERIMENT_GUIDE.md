# Fairino FR10 JEPA & DINO-WM Policy Execution Guide

## 1. Summary of Changes & Engineering Improvements

1. **Meta FAIR DINO-WM Integration (`jepa_control/pipeline/dino_wm_runner.py`)**:
   - Official DROID pretrained weights (`dino_wm_droid.pth.tar`) with CEM model-predictive control.
   - 5.5x faster evaluation step latency (~3.8s/step vs 22s/step) via batch size and CEM horizon tuning.

2. **Visual Object Grounding (`jepa_control/perception/screwdriver_detector.py`)**:
   - Live camera segmentation for tabletop target objects.
   - Dynamic affine transformation: maps detected pixel centroids $(u, v)$ into robot Cartesian space $(X, Y, R_z)$ so the arm automatically centers directly over the object wherever it is placed on the table.

3. **Physical Gripper Calibration (`jepa_control/robot/fairino_driver.py`)**:
   - Calibrated JODELL RG gripper:
     - `pos = 0`: Fully closed firm grasp.
     - `pos = 100`: Wide open release.
   - Expanded Cartesian reachable envelope: `min_y_mm = -1020.0` to permit full tabletop reach without artificial clamping.

4. **Staged 5-Phase Milestone Architecture (`scripts/run_offline_rollout.py`)**:
   - **Phase 1 (Descent)**: Centers $(X, Y)$ above target and descends steadily.
   - **Phase 2 (Grasp)**: Clamps gripper firmly (`pos = 0`) with a 1.2s physical dwell to ensure secure grasp.
   - **Phase 3 (Lift)**: Lifts cleanly to safe elevation ($Z \ge 120\text{ mm}$).
   - **Phase 4 (Transport)**: Dynamically carries across to place coordinates.
   - **Phase 5 (Place & Release)**: Lowers into destination, releases gripper (`pos = 100`), and retracts.

---

## 2. Setup Evaluation: Black Cloth & Bottle

> [!NOTE]
> **Is the bottle and black cloth setup good for experiments?**
>
> **YES, it is excellent!**
> 1. **Clean Visual Features**: The matte black cloth eliminates aluminum surface reflections and glare from overhead lighting. This gives the DINO-WM patch tokens and feature extractors extremely sharp representations of the robot arm and the object.
> 2. **High Visual Contrast**: The silver/light bottle stands out distinctly against the black background, allowing reliable tracking.
> 3. **Demonstration Alignment**: Your recorded demonstration (`fr10_demo.mp4` / `fr10_pick_and_place.h5`, 361 frames) executed a complete pick-and-place cycle on this exact setup.
>
> **Things to keep in mind**:
> - Ensure the bottle is light enough or gripped at a stable height so it does not slip or tip during high-speed transport.

---

## 3. How to Run

### Step 1: Activate Environment
```bash
cd /home/fr10/fr10_ws/src/jepa-world-model-control
source .venv/bin/activate
```

### Step 2: Reset Robot Pose
- **Return to Tabletop Home Pose**:
  ```bash
  python3 scripts/reset_to_home.py
  ```
- **Reset Directly Above Target Object (Pre-Approach)**:
  ```bash
  python3 scripts/reset_to_home.py --pick
  ```

### Step 3: Run Autonomous JEPA Policy Rollout
- **Live Run with DINO-WM and HUD Visualizer**:
  ```bash
  python3 scripts/run_offline_rollout.py --model dino_wm --live --camera zed --visualize --max_steps 30 --speed 45
  ```
- **Run without Visual Centroid Grounding (Pure Reference Demonstration)**:
  ```bash
  python3 scripts/run_offline_rollout.py --model dino_wm --live --camera zed --visualize --max_steps 30 --speed 45 --no_grounding
  ```
