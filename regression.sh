# !/bin/bash

set -e

GPUIDX=$1
NUMEXP=$2
RESPATH=$3

if [ ! -d "data" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir data
fi

echo "gpuidx is ${GPUIDX}"

wget 'http://joneswzshare.oss-cn-hangzhou-zmf.aliyuncs.com/download/public-data.zip'
yes A | unzip public-data.zip -d data/
rm public-data.zip

if docker images | grep nehzux/kddcup2020; then
  echo "the docker image has been pulled"
else
  echo "no docker image"
  docker pull nehzux/kddcup2020:v2
fi

if [ -z $3 ]
then
  echo "you do not specify the result saving path. Use default path: current dir"
  RESPATH="."
else
  if [ ! -d ${RESPATH} ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir ${RESPATH}
  fi
fi

for i in $(seq 1 1 $NUMEXP)
do
  echo "exp${i}"
  echo "results save to ${RESPATH}/"
  rm -rf ${RESPATH}/exp${i}.out
  rm -rf ${RESPATH}/exp${i}.err
  docker run --gpus device=${GPUIDX} --cpus=4 --memory=30g -it --rm -v "$(pwd):/app/autograph" -w /app/autograph nehzux/kddcup2020:v2 python run_local_test.py --dataset_dir=./data/public/a --code_dir=./code_submission 2>>${RESPATH}/exp${i}.err | tee -a ${RESPATH}/exp${i}.out
  docker run --gpus device=${GPUIDX} --cpus=4 --memory=30g -it --rm -v "$(pwd):/app/autograph" -w /app/autograph nehzux/kddcup2020:v2 python run_local_test.py --dataset_dir=./data/public/b --code_dir=./code_submission 2>>${RESPATH}/exp${i}.err | tee -a ${RESPATH}/exp${i}.out
  docker run --gpus device=${GPUIDX} --cpus=4 --memory=30g -it --rm -v "$(pwd):/app/autograph" -w /app/autograph nehzux/kddcup2020:v2 python run_local_test.py --dataset_dir=./data/public/c --code_dir=./code_submission 2>>${RESPATH}/exp${i}.err | tee -a ${RESPATH}/exp${i}.out
  docker run --gpus device=${GPUIDX} --cpus=4 --memory=30g -it --rm -v "$(pwd):/app/autograph" -w /app/autograph nehzux/kddcup2020:v2 python run_local_test.py --dataset_dir=./data/public/d --code_dir=./code_submission 2>>${RESPATH}/exp${i}.err | tee -a ${RESPATH}/exp${i}.out
  docker run --gpus device=${GPUIDX} --cpus=4 --memory=30g -it --rm -v "$(pwd):/app/autograph" -w /app/autograph nehzux/kddcup2020:v2 python run_local_test.py --dataset_dir=./data/public/e --code_dir=./code_submission 2>>${RESPATH}/exp${i}.err | tee -a ${RESPATH}/exp${i}.out
done

docker run --cpus=4 --memory=30g -it --rm -v "$(pwd):/app/autograph" -w /app/autograph nehzux/kddcup2020:v2 python code_submission/show_perf.py --num_exp=${NUMEXP} --res_path=${RESPATH}/exp_summary.txt | tee ${RESPATH}/exp_summary.txt

echo "finished the regression"

