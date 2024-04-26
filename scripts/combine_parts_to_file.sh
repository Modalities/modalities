#!/bin/sh

# Usage: ./combine_files.sh <directory_with_parts> <output_filename>
# Example: ./combine_files.sh /path/to/parts combined_file.dat

dir_with_parts="$1"
output_filename="$2"

# Check if parameters are provided
if [ -z "$dir_with_parts" ] || [ -z "$output_filename" ]; then
    echo "Usage: ./combine_files.sh <directory_with_parts> <output_filename>"
    exit 1
fi

# Check if the directory exists
if [ ! -d "$dir_with_parts" ]; then
    echo "Directory does not exist: $dir_with_parts"
    exit 1
fi

# Combining files
echo "Combining files from $dir_with_parts to create $output_filename..."
cat "$dir_with_parts"/* > "$output_filename"

echo "Combination complete."