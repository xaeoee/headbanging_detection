# headbanging_detection



# Clap Detection Using CNN
## **I.** Description
This project processes video files to detect headbanging using **normal vectors**. MediaPipe is utilized to derive the coordinates of key facial landmarks. Specifically, two vectors are calculated: one extending from the midpoint between the eyes to the left lip and another to the right lip. By computing the cross product of these two vectors, a vector perpendicular to the facial plane at the midpoint between the eyes is obtained. 

Using this perpendicular vector, the program tracks:
- The movement direction,
- The number of movements,
- The time duration of the movements, and
- The angle of movement.

Thresholds are then set based on these metrics. If these thresholds are exceeded, the behavior is classified as headbanging. 

This program facilitates the identification of clapping, which can be indicative of abnormal behavior in children with ASD.

## II. Environment Set-up

### Step1. Git Clone
Use `git clone` or directly download.
.
Then, you should have the following directory structure:
```
HEADBANGING_DETECTION/
├── function.py
├── headbanging.py
├── main.py
├── README.md
├── requirement.txt
├── version.py
```

Here's the updated README instructions with Python version `3.9.21` specified:

---

### Step 2. Create a Conda Virtual Environment Using `requirements.txt`

1. First, create a new Conda environment with Python 3.9.21:
   ```bash
   conda create -n <environment_name> python=3.9.21
   ```
   Replace `<environment_name>` with the desired name for your environment.

2. Activate the newly created environment:
   ```bash
   conda activate <environment_name>
   ```

3. Install the Python packages listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

### Step3. Run & test
```
conda activate env  # Activate the environment
python3 main.py --video_path yourvideo.mp4
```
