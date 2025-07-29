docker run --name maniskill -itd --entrypoint bash --shm-size=64g --gpus all --network host \
  -v ${HOME}/workspace:/root/workspace:rw,z \
  -v /mnt/nfs_68/caozhe/miniconda3:/mnt/nfs_68/caozhe/miniconda3:rw,z \
  -v ${HOME}/mnt/nfs_68/caozhe/conda:/root/.cache/conda:rw,z \
  -v ${HOME}/mnt/nfs_68/caozhe/pip:/root/.cache/pip:rw,z \
  -v ${HOME}/mnt/nfs_68/caozhe/huggingface:/root/.cache/huggingface:rw,z \
  maniskill:latest