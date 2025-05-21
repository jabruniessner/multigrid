#!/bin/awk -f
# measure_distances.awk

# This script takes as input the start line and end line of the points in a ply file and the center point of a sphere 
# and computes the distance between the point in the line and the center of the sphere
#
# Usage: awk -f measure_distances.awk <start_line> <end_line> <center_x> <center_y> <center_z> input.ply
#

BEGIN {
  num_points = ARGV[1]
  center_x = ARGV[2]
  center_y = ARGV[3]
  center_z = ARGV[4]

  ARGV[1] = ""
  ARGV[2] = ""
  ARGV[3] = ""
  ARGV[4] = ""
  }

NR==10, NR==10+num_points-1 {
  distance = sqrt((center_x - $1)^2 + (center_y - $2)^2 + (center_z - $3)^2)
  print "Distance " NR ": " distance
  }
