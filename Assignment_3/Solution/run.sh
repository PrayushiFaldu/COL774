#!/bin/bash

if [ "$1" == "1" ];
then
	if [ "$5" == "a" ];
	then
	   python3 1_a.py $2 $3 $4
	elif [ "$5" == "b" ];
	then
	   python3 1_b.py $2 $3 $4
	elif [ "$5" == "c" ];
	then
	   python3 1_c.py $2 $3 $4
	elif [ "$5" == "d" ];
	then
	   python3 1_d.py $2 $3 $4
	fi
elif [ "$1" == "2" ];
then
  if [ "$4" == "a" ];
  then
   echo "output included in part c."
  elif [ "$4" == "b" ];
  then
   echo "output included in part c."
  elif [ "$4" == "c" ];
  then
   python3 2_b_a.py $2 $3
  elif [ "$4" == "d" ];
  then
   python3 2_b_a.py $2 $3
  elif [ "$4" == "e" ];
  then
   python3 2_e.py $2 $3
  elif [ "$4" == "f" ];
  then
   python3 2_f.py $2 $3
  fi
fi