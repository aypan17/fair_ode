#!/usr/bin/env bash
for EXT in data ops tokens left right
do
	cat a/prim_fwd.${EXT} b/prim_fwd.${EXT} c/prim_fwd.${EXT} d/prim_fwd.${EXT} e/prim_fwd.${EXT} f/prim_fwd.${EXT} g/prim_fwd.${EXT} h/prim_fwd.${EXT} i/prim_fwd.${EXT} > fwd_train.${EXT}
done
