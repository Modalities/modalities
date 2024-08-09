import ast
import csv
import os


def get_num_functions_from_file(file_path) -> int:
    with open(file_path, "r") as file:
        file_content = file.read()

    tree = ast.parse(file_content)

    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

    return len(functions)


def count_def_lines(file_path):
    count = 0
    with open(file_path, "r") as file:
        for line in file:
            if line.strip().startswith("def"):
                count += 1
    return count


src_directory = "/raid/s3/opengptx/mehdi/git_repos/modalities/src"
result = []

for root, _, files in os.walk(src_directory):
    for file in files:
        if file.endswith(".py"):
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, src_directory)
            def_count = get_num_functions_from_file(file_path)
            result.append((relative_path, def_count))

# ... existing code ...

# Write the result to a CSV file
csv_file = "/raid/s3/opengptx/mehdi/git_repos/modalities/method_counts.csv"

with open(csv_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["File", "Method Count"])
    for relative_path, def_count in result:
        writer.writerow([f"src/{relative_path}", def_count])

print(f"Result written to {csv_file}")
