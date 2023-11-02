#!/bin/bash

commit_msg_file="$1"
default_revert_msg=$(cat "$commit_msg_file")
convco_aligned_revert_msg=$(echo "$default_revert_msg" | sed '1s/^Revert /revert: /')
echo "$convco_aligned_revert_msg" > "$commit_msg_file"
