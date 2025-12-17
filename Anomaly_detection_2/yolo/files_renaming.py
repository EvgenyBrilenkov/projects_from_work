import os
import glob

directory = "/wrk/main/yolo/yolo_data/labels/train"

for filepath in glob.glob(os.path.join(directory, "*")):
    dirname = os.path.dirname(filepath)
    basename = os.path.basename(filepath)
    name, ext = os.path.splitext(basename)

    new_name = name[:3]+ext
    new_path = os.path.join(dirname, new_name)
    os.rename(filepath, new_path)