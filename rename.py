import os

root = "./image enhancement/img"

for file in os.listdir(root):
    if "guilded_" in file:
        rename = file.replace("guilded_", "guided_")
        print(file)
        os.rename(os.path.join(root, file), os.path.join(root, rename))

