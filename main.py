import os
import sys
import matplotlib as mpl
import subprocess
import shutil

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
    # File properties.
    file_path = os.path.join(search_path, file)
    file_base = os.path.splitext(file)[0]
    file_extension = os.path.splitext(file)[1]
    result_path = os.path.join(output_path, file_base)

    # Result directory.
    os.makedirs(result_path, exist_ok=True)
    shutil.copyfile(file_path, os.path.join(result_path, file))

    # Sequence of modules to process image file.
    print("Running: ", file_path)
    subprocess.run(python_command + " FindScaleBar.py --file %s" % os.path.join(result_path, file))
    subprocess.run(python_command + " ImageSelection.py --file %s" % os.path.join(result_path, file))
    subprocess.run(python_command + " GridSegmentation.py --file %s" % os.path.join(
        result_path, file_base + "_select" + file_extension))
    subprocess.run(python_command + " DropletSeparation.py --file %s" % os.path.join(
        result_path, file_base + "_select" + file_extension))

