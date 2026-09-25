#!/bin/bash







#echo "The number of input arguments is: $#"

if [ $# -ne 1 ]; then
  echo "script requires exactly one input parameter, omega"
  exit 1
fi


program_name="PBE_pmgc_class_test"

project_path="$(pwd)/.."
kappa_map_path="$project_path/test_kappa_function"
pqr_file="$kappa_map_path/1PIT.pqr"
origin_file="$kappa_map_path/1PIT-PE0.dx"
program="$project_path/build/$program_name"
outfile="output.txt"

make $program_name

#echo "Using the origin: " $origin

if [ -f $pqr_file ] && [ -f $origin_file ] && [ -f $program ]; then
 # echo "using pqr file $pqr_file"
 # echo "using origin file $origin_file"
  
  origin=$(awk '$1=="origin"{print $2" "$3" "$4}' $origin_file)

  $program $pqr_file $outfile $origin $1

fi

