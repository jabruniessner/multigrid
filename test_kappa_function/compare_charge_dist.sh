#!/bin/bash


apbs apbs.in

a=$(awk 'NR==6{print $2, $3, $4}' kappa_apbs-PE0.dx)

../create_charge_dist 1PIT.pqr charge_own.dx $a 1.5

python3 compare_kappa_maps.py charge_apbs-PE0.dx charge_own.dx $1


