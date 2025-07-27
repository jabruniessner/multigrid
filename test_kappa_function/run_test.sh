#!/bin/bash


kappa_map_path="/home/jakob/projects/multigird/Vincent_Meeting_Series_discontinuous_coeff/test_kappa_function"

origin=$(awk '$1=="origin"{print $2" "$3" "$4}' "$kappa_map_path/single_atom-PE0.dx")
echo $origin
