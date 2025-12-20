#!/bin/bash


echo "The number of input arguments is: $#"

if [ $# -ne 1 ]; then
  echo "script requires exactly one input parameter, omega"
  exit 1
fi


project_path="$(pwd)/.."
kappa_map_path="$project_path/test_kappa_function"
pqr_file="$kappa_map_path/single_atom.pqr"
origin_file="$kappa_map_path/single_atom-PE0.dx"
program="$project_path/build/Laplacian3D_epsilon" 

  folder="outputfiles"

  if [[ ! -d $folder ]]; then
    mkdir $folder
  fi




for i in {2..5}; do
  

  program="Laplacian_epsilon_3D_${i}l"

  make $program

  echo "Using the origin: " $origin

  if [ -f $pqr_file ] && [ -f $origin_file ] && [ -f $program ]; then
    echo "using pqr file $pqr_file"
    echo "using origin file $origin_file"
  
    origin=$(awk '$1=="origin"{print $2" "$3" "$4}' $origin_file)

    echo "The origin is: " $origin
    ./$program $pqr_file $origin $1 10 > "$folder/outfile_${i}_levels.txt"
 
  else
    echo "One of the files is was not found, check that they are all there!"
    echo "pqr file " $pqr_file
    echo "origin file " $origin_file
    echo "program file " $program
  fi

done

