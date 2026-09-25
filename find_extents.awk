#!/usr/bin/awk -f

NR==1{
  max[0]=min[0]=$6
  max[1]=min[1]=$7
  max[2]=min[2]=$8
  }

/TER/{next}
/END/{next}

NR>1{
  max[0]=($6>max[0])?$6:max[0]
  min[0]=($6<min[0])?$6:min[0]
  max[1]=($7>max[1])?$7:max[1]
  min[1]=($7<min[1])?$7:min[1]
  max[2]=($8>max[2])?$8:max[2]
  min[2]=($8<min[2])?$8:min[2]
  }

END{
  printf("X: %f %f\n", min[0], max[0])
  printf("Y: %f %f\n", min[1], max[1])
  printf("Z: %f %f\n", min[2], max[2])
  }
