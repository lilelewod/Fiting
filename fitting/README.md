# Induction

Probabilistic program induction for robust machine learning.

## Installation

[HARL]: https://github.com/PKU-MARL/HARL
[LightZero]: https://github.com/opendilab/LightZero
[Dreamer V3]:https://github.com/danijar/dreamerv3
[DI-engine]: https://github.com/opendilab/DI-engine
[Drm]: https://arxiv.org/abs/2310.19668
[TorchRL]: https://github.com/pytorch/rl
[EvoX]: https://github.com/EMI-Group/evox

You can use one of the following reinforcement learning algorithm libraries: [HARL], [DI-engine] (i.e., ding), [LightZero], [Dreamer V3], [DrM], or [TorchRL]. To use the libraries, you can either build docker images use the `Dockerfile` in the `docker` folder, or download docker images from the following links.


#### [DI-engine] (ding) 

（1）使用cupy <br>
- docker image (with cupy): <br>
    - 通过网盘分享的文件：agent-v4-cupy14.0.0a1-torch2.5.1-cu124.tar <br>
    - 链接: https://pan.baidu.com/s/1thcvLJa_C1_NoQbbVygcGA?pwd=sins <br>
    - 注：本镜像也可以运行HARL库（configurations/surface/harl/happo/config_step_based.py），但因open3d故需将numpy降级到1.26版本

- docker container: <br>
  - python interpreter: `/usr/bin/python3` <br>
  - `root` 密码: `123456` <br>
  - 普通用户`coder` 密码: `123456` <br>

- 运行容器请指定用户为`coder`：<br>
  - 容器不存在时创建并运行名为`agent`的容器：<br>
`docker run -it --gpus all --ipc=host --net=host -v /home/zzl/code:/mnt -e DISPLAY=$DISPLAY --name=agent -u coder zhangzongliang/agent:v4-cupy14.0.0a1-torch2.5.1-cu124` <br>
  - 容器已存在时先启动容器：`dokcer start agent` <br>
  - 然后进入容器：`docker exec -u coder -it agent /bin/bash` <br>
  - 用Python运行文件：`configurations/road_curve/ding/ppo/config_cupy.py` <br>
  - 修改和提交代码时记得修改git账户名和邮箱


（2）不使用cupy <br>
- docker image (without cupy): https://pan.baidu.com/s/1vquPxn5Cv34jUKR9gNzzKQ?pwd=427n  <br>


- Install DI-engine (i.e., ding) 0.5.1 using one of the commands below: <br>
`pip install -i DI-engine==0.5.1` <br>
`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple DI-engine==0.5.1`

- The base image can be found at: https://github.com/anibali/docker-pytorch <br>
- Python interpreter: `/home/user/micromamba/bin/python` <br>
- How to root in the container: `sudo su`, https://github.com/anibali/docker-pytorch/issues/9 

#### [HARL]

- 使用podman运行容器（推荐）：
  - podman镜像agent-v6-torch2.6.0-cu12.4.tar：
  - 下载链接: https://pan.baidu.com/s/1kIgOTNZ7txWf_Vyzh-bdEQ?pwd=jwjx 提取码: jwjx
  - podman用法与docker基本一样，将docker命令替换为podman即可
  - 要使用cuda需安装CDI：
    - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
    - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html
  - 并使用如下命令创建容器： <br>
  `podman run -it --device nvidia.com/gpu=all --ipc=host --net=host -v /home/m25lll/code:/mnt -e DISPLAY=$DISPLAY --name=agent localhost/zhangzongliang/agent:v6-torch2.6.0-cu12.4` <br>
  - 注意：podman在容器里直接使用root账户，而不使用其他普通账户（例如上文中的`coder`账户）
  - python interpreter: `/opt/conda/bin/python` <br>
  - example: `configurations/surface/harl/happo/config_step_based.py` <br>


#### [TorchRL]

- 使用podman运行容器（推荐）：
  - podman镜像agent-v5-torchrl0.7.2-torch2.6.0-cu12.4.tar：
  - 下载链接: https://pan.baidu.com/s/1dqA-Apz8HnrP6aHmC6J1yQ?pwd=7ri2 提取码: 7ri2 
  - podman用法与docker基本一样，将docker命令替换为podman即可
  - 要使用cuda需安装CDI：
    - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
    - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html
  - 并使用如下命令创建容器： <br>
  `podman run -it --device nvidia.com/gpu=all --ipc=host --net=host -v /home/zzl/code:/mnt -e DISPLAY=$DISPLAY --name=agent localhost/zhangzongliang/agent:v5-torchrl0.7.2-torch2.6.0-cu12.4` <br>
  - 注意：podman在容器里直接使用root账户，而不使用其他普通账户（例如上文中的`coder`账户）
  - python interpreter: `/opt/conda/bin/python` <br>


#### [LightZero]

docker image: https://pan.baidu.com/s/1AgrgyBH8MXZ4bo0NawzSaQ?pwd=dy8x <br>

In Project Structure Add Content Root: `/opendilab/LightZero`
![lightzero-content-root](documents/images/lightzero-content-root.png)

#### [Dreamer V3]

docker image: https://pan.baidu.com/s/18cPqLqCkTj2vVmZvuSOtzw?pwd=sx7c <br>

#### [DrM]

docker image: https://pan.baidu.com/s/1dXPP1In37cDEDWI9jPfsCw?pwd=rtyq <br>
Python interpreter in the docker image: `/home/user/micromamba/bin/python` <br>


## Execution
Run a file in the `configurations` folder


## Contribution

1.  Fork the repository
2.  Create Feat_xxx branch
3.  Commit your code
4.  Create Pull Request

## Tips
Word version of the README file with image description can be found in: `documents/README.Docx` <br>

If using PyCharm, you can add the folder `utils/lib` as Source Folders: `Settings -> Project -> Project Structure -> Mark as Sources` <br>

#### cupy
docker image: `comming soon` <br>
docker container: python interpreter: `/usr/bin/python3`; root password: `123456`; coder password: `123456` <br>
kdtree is not in the latest release of cupy and only in the developing main branch, therefore should be installed locally (inside the official cupy docker container) as follows: <br>
`pip uninstall cupy-cuda12x` <br>
`git clone --recursive https://github.com/cupy/cupy.git` <br>
`cd cupy` <br>
`pip install -e .` <br>

#### podman
更新后显卡驱动，需重新生成显卡配置文件，否则可能无法启动容器： <br>
`sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` <br>
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html <br>
https://github.com/NVIDIA/nvidia-container-toolkit/issues/450

## Useful commands
`watch -n 2 nvidia-smi` <br>

`git config --global --add safe.directory /mnt/robustlearning/agent` <br>

Create and run docker container using one of the following commands: <br>

`docker run -it --gpus all --ipc=host --net=host -v /home/zzl/code:/mnt -e DISPLAY=$DISPLAY --name=agent zhangzongliang/modeling:v3-pytorch1.13.1` <br>

`docker run -it --gpus all --ipc=host --net=host -v /home/zzl/code:/mnt -v /tmp/.X11-unix:/tmp/.X11-unix -e GDK_SCALE -e GDK_DPI_SCALE -e DISPLAY=$DISPLAY --name=modeling zhangzongliang/modeling:v3-pytorch1.13.1` <br>

Start docker container: <br>
`dokcer start modeling`

Run docker container as root: <br>
`docker exec -it modeling /bin/bash`

Run docker container `agent` as `user`: <br>
`docker exec -u user -it agent /bin/bash`

Run VSCode inside container: `code --no-sandbox --user-data-dir="./.vscodedata/"` <br>
VSCode 可能需安装依赖： https://code.visualstudio.com/docs/setup/linux <br>
sudo apt install ./<file>.deb <br>
// If you're on an older Linux distribution, you will need to run this instead: <br>
// sudo dpkg -i <file>.deb <br>
// sudo apt-get install -f # Install dependencies <br>





