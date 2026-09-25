#!/bin/bash







#echo "The number of input arguments is: $#"

if [ $# -ne 1 ]; then
  echo "script requires exactly one input parameter, omega"
  exit 1
fi


project_path="$(pwd)/.."
kappa_map_path="$project_path/test_kappa_function"
pqr_file="$kappa_map_path/single_atom.pqr"
origin_file="$kappa_map_path/single_atom-PE0.dx"
program="$project_path/build/Laplacian3D_epsilon" 

function finding_convergence_rate {

 line_array=()
  
  #reading the output of the executed program into an array
  while read -r line; do 
    vals=$(echo $line | awk '{print $4" "$7}')
    if [ $(echo $vals | wc --words) -eq 2 ]; then
      line_array+=( "$vals" )
    fi
  done < <($program $pqr_file $origin $1 10)
  
  #finding the convergence factor
  last_elem=$(echo ${line_array[-1]} } | awk '{print $2}')
  first=$(echo ${line_array[0]} | awk '{print $2}')
  convergence_factor=$(echo $last_elem $first | awk '{print ($1/$2)^(1./9)}')
  echo $convergence_factor

}


#make Laplacian3D_epsilon


#echo "Using the origin: " $origin

if [ -f $pqr_file ] && [ -f $origin_file ] && [ -f $program ]; then
 # echo "using pqr file $pqr_file"
 # echo "using origin file $origin_file"
  
  origin=$(awk '$1=="origin"{print $2" "$3" "$4}' $origin_file)

 # echo "The origin is: " $origin

  convergence_factor_now=0;
  convergence_factor_previous=1;
  convergence_factor_previous_previous=1;

  omega_now=$1
  omega_previous=0
  omega_previous_previous=0


  convergence_factor_now=$(finding_convergence_rate $omega_now)

  #finding_convergence_rate $omega_now

  echo $1 " " $convergence_factor_now


 
else
  echo "One of the files is was not found, check that they are all there!"
  echo "pqr file " $pqr_file
  echo "origin file " $origin_file
  echo "program file " $program
fi

