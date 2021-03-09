#!/usr/bin/env bash
for EXT in data ops tokens left right
do
	tail -1406 data_precompute_sorted/fwd_sorted_valid.${EXT} > data_precompute_sorted/fwd_9to15_valid.${EXT}
done
