#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <directory> <topic_id> <featurization>"
    exit 1
fi

dir=$1
topic_id=$2
featurization=$3

# Create a directory for CRF temporary files
mkdir -p crf-tmp-${topic_id}
pushd crf-tmp-${topic_id}

# Generate folds
for i in 0 1 2 3 4
do
    # Ensure the fold file is empty
    > fold.$i
    
    # Find all cache files and read document IDs from them
    cache_files=$(ls $dir/qrels/$topic_id/*-$i.cache 2> /dev/null)
    if [ -z "$cache_files" ]; then
        echo "No cache files found for fold $i in $dir/qrels/$topic_id/"
        continue
    fi

    for cache_file in $cache_files
    do
        while read docid
        do
            qrels_file="$dir/qrels/$topic_id/$docid.qrels"
            doc_file="$dir/docs/$docid.$featurization"
            # Check if the required files exist before pasting
            if [ -f "$qrels_file" ] && [ -f "$doc_file" ]; then
                paste $qrels_file $doc_file >> fold.$i
                echo >> fold.$i # Line break for sequence learners/crfsuite
            else
                echo "Missing file for $docid in fold $i"
            fi
        done < $cache_file
    done
done

popd
echo "Folds generated in directory crf-tmp-${topic_id}"
