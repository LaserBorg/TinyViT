classnames_path = "checkpoints/22k/imagenet22k_classnames.txt"

imagenet22k_classnames = []

with open(classnames_path, "r") as f:
    for line in f:
        first_entry = line.strip().split(",")[0]
        imagenet22k_classnames.append(first_entry)
