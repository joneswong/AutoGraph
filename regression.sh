# !/bin/bash

set -e

if [ ! -d "data" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir data
fi

wget 'http://joneswzshare.oss-cn-hangzhou-zmf.aliyuncs.com/download/public-data.zip'
yes A | unzip public-data.zip -d data/
rm public-data.zip

if docker images | grep nehzux/kddcup2020; then
  echo "the docker image has been pulled"
else
  echo "no docker image"
  docker pull nehzux/kddcup2020:v2
fi

# if you are using the original docker: --gpus=0
# if you are using ali docker (e.g., map 7-th gpu to the 0-th gpu): --device /dev/nvidia7:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm
docker run --device /dev/nvidia7:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm -it --rm -v "$(pwd):/app/autograph" -w /app/autograph nehzux/kddcup2020:v2 python run_local_test.py --dataset_dir=./data/public/a --code_dir=./code_submission
docker run --device /dev/nvidia7:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm -it --rm -v "$(pwd):/app/autograph" -w /app/autograph nehzux/kddcup2020:v2 python run_local_test.py --dataset_dir=./data/public/b --code_dir=./code_submission
docker run --device /dev/nvidia7:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm -it --rm -v "$(pwd):/app/autograph" -w /app/autograph nehzux/kddcup2020:v2 python run_local_test.py --dataset_dir=./data/public/c --code_dir=./code_submission
docker run --device /dev/nvidia7:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm -it --rm -v "$(pwd):/app/autograph" -w /app/autograph nehzux/kddcup2020:v2 python run_local_test.py --dataset_dir=./data/public/d --code_dir=./code_submission
docker run --device /dev/nvidia7:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm -it --rm -v "$(pwd):/app/autograph" -w /app/autograph nehzux/kddcup2020:v2 python run_local_test.py --dataset_dir=./data/public/e --code_dir=./code_submission

echo "finished the regression"
