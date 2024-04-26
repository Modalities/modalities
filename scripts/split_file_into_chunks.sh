#!/bin/sh

# Usage: ./split_file.sh <inputfile>
# Example: ./split_file.sh largefile.dat

inputfile="$1"
blocksize=1073741824  # 1GB block size in bytes
count=40              # Each chunk will have 40 blocks of 1GB (40GB per chunk)
filesize=$(stat -c %s "$inputfile")  # Get the size of the file in bytes
chunksize=$((blocksize * count))     # Size of each chunk in bytes

num_chunks=$(( (filesize + chunksize - 1) / chunksize ))  # Calculate the total number of chunks

echo "Total size of file: $filesize bytes"
echo "Size of each chunk: $((blocksize * count)) bytes"
echo "Number of chunks required: $num_chunks"

i=0
while [ $i -lt $num_chunks ]
do
    skip=$((i * count))
    outputfile="${inputfile}_part${i}"
    echo "Creating chunk $outputfile..."
    dd if="$inputfile" of="$outputfile" bs=$blocksize count=$count skip=$skip
    i=$((i + 1))
done

echo "File splitting complete."