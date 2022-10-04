#!/bin/bash

if [ "$1" == "1" ];
then
	if [ "$4" == "a" ];
	then
	   python3 A_a_b_c.py $2 $3
	elif [ "$4" == "b" ];
	then
	   python3 A_a_b_c.py $2 $3
	elif [ "$4" == "c" ];
	then
	   python3 A_a_b_c.py $2 $3
	elif [ "$4" == "d" ];
	then
	   python3 A_d.py $2 $3
	elif [ "$4" == "e" ];
	then
	   python3 A_e.py $2 $3
	elif [ "$4" == "g" ];
	then
	   python3 A_g.py $2 $3
	fi
elif [ "$1" == "2" ];
then
  if [ "$4" == "0" ];
	then
	  if [ "$5" == "a" ];
	  then
	   python3 2_a_a.py $2 $3
	  elif [ "$5" == "b" ];
	  then
	   python3 2_a_b.py $2 $3
	  elif [ "$5" == "c" ];
	  then
	   python3 2_a_c.py $2 $3
	  fi
	elif [ "$4" == "1" ];
	then
	  if [ "$5" == "a" ];
	  then
	   python3 2_b_a.py $2 $3
	  elif [ "$5" == "b" ];
	  then
	   python3 2_b_b.py $2 $3
	  elif [ "$5" == "c" ];
	  then
	   echo "Confusion metrics printed in respective parts a and b and also included in write up file"
	  elif [ "$5" == "d" ];
	  then
	   python3 2_b_d.py $2 $3
	  fi
	fi
fi