#!/bin/bash
#---run
#############################################################################
# ./MAIN.sh 0 test 200 0.18

#
#
#---compile
#############################################################################
nvcc -O3 -DSFMT_MEXP=19937 src/mcpf_2d_usc.cu src/SFMT.c -o run_simulation -std=c++11 -lcufft


#
#
#---func
#############################################################################

fmkdir () {
    if [ ! -d $1 ]; then
	mkdir $1
    fi
}

frm(){
    if [ -d $1 ]; then
	rm -r $1
    fi
}


#
#---main
#############################################################################

fmkdir "DATA"

Dir="DATA/"$1
fmkdir $Dir
rm -fr $Dir/*

GPUnumlist=()
GPUnumuuidlist=()
st=`expr $1 \* 20`
echo $st
sleep $st
GPUmem=`nvidia-smi --query-gpu="memory.used" --format=csv,noheader,nounits`
GPUuuid=`nvidia-smi --query-gpu="uuid" --format=csv,noheader,nounits`
GPUmemlist=(${GPUmem// / })
GPUuuidlist=(${GPUuuid// / })

j=0
for i in "${!GPUmemlist[@]}"
do
  mem=`echo ${GPUmemlist[$i]}`
  if [ $mem -lt 1000 -a $i -ne 2 ]; then
      GPUnumlist+=($j)
      GPUnumuuidlist+=(${GPUuuidlist[$i]})
  fi
  if [ $i -ne 2 ]; then
      j=`expr $j + 1`
  fi
done

echo "${GPUnumlist[@]}"

echo "${GPUnumuuidlist[@]}"
echo "${GPUnumuuidlist[0]}"

CUDA_VISIBLE_DEVICES=${GPUnumuuidlist[0]} ./run_simulation $1 $2 $3 $4
