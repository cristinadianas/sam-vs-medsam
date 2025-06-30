import os
import subprocess

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
venv_python = os.path.join(BASE_DIR, "venv39", "Scripts", "python.exe")
script_path = os.path.join(BASE_DIR, "benchmark.py")

tasks = [
    "Task01_BrainTumour",
    "Task02_Heart",
    "Task03_Liver"
    "Task04_Hippocampus",
    "Task05_Prostate",
    "Task06_Lung",
    "Task07_Pancreas",
    "Task08_HepaticVessel",
    "Task09_Spleen",
    "Task10_Colon"
]

bbox_values = [3, 5, 10, 15, 20]
limit = "25"

for task in tasks:
    for bbox in bbox_values:
        subprocess.run([
            venv_python,
            script_path,
            "--task", task,
            "--bbox", str(bbox),
            "--limit", limit
        ])
