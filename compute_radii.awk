

{
  x=50
  y=50
  z=50
  r=45

  i = $1+.5-x
  j = $2+.5-y
  k = $3+.5-z
  a = sqrt(i*i + j*j + k*k)  
  if( a > r+sqrt(3.)/2){
    print $1" "$2" "$3" "a
    }
}
