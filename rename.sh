#!/usr/bin/env bash
for TYPE in prim_fwd
do
for EXT in data ops tokens left right
do
	mv ${TYPE}.${EXT} fwd_valid.${EXT}
done
done
