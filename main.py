import os
import sys
import matplotlib as mpl
import subprocess

mpl.use("Qt5Cairo")

python_command = "python" if sys.platform[0:3] == "win" else "python3"
search_path = "input"
output_path = "output"
ending = [".png", ".PNG", ".jpg", ".JPG", ".tif", ".TIF"]  # Allowed endings

# Make a list of all valid image files in search_path.
files = []
if os.path.exists(search_path):
    files = [f for f in os.listdir(search_path) if
             os.path.isfile(os.path.join(search_path, f)) and any([f.endswith(x) for x in ending])]
else:
    raise FileNotFoundError("Input path could not be found. Put files in input folder.")

# Assert output directory.
os.makedirs(output_path, exist_ok=True)

# Loop over all valid input files.
for file in files:
    # File naming.
    filepath = os.path.join(search_path, file)
    file_base = os.path.splitext(file)[0]
    os.makedirs(os.path.join(output_path, file_base), exist_ok=True)

    # Sequence of modules to process image file.
    print("Running: ", filepath)
    subprocess.run([python_command, "ImageSelection.py", "--file ", filepath], shell=True, check=True)
    # subprocess.run(python_command + " GridSegmentation.py --file " + file, shell=True, check=True)
    # subprocess.run(python_command+" DropletSeparationWithGrid.py --file "+file, shell=True, check=True)
    # subprocess.run(python_command + " DropletSeparation3.py --file " + file, shell=True, check=True)

