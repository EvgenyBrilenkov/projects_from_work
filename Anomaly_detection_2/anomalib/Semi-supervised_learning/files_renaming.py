import os

def rename_files(directory_path):
    files = os.listdir(directory_path)
    for index, filename in enumerate(files):
        current_path = os.path.join(directory_path, filename)
        name, ext = os.path.splitext(filename)
        new_name = f"{index:03d}_mask{ext}"
        new_path = os.path.join(directory_path, new_name)
        try:
            os.rename(current_path, new_path)
            print(f'{current_path} -> {new_path}')
        except Exception as e:
            print(e)
            
rename_files('/wrk/main/data/masks')