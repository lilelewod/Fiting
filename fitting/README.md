# Fitting

Robust geometric model fitting.

## Installation

[HARL]: https://github.com/PKU-MARL/HARL
[TorchRL]: https://github.com/pytorch/rl

You can use one of the following reinforcement learning algorithm libraries: [HARL], or [TorchRL]. To use the libraries, you can either build docker/podman images use the `Dockerfile` in the `docker` folder, or download docker/podman images from the following links.


#### [HARL]

- 使用podman运行容器（推荐）：
  - podman镜像agent-v6-torch2.6.0-cu12.4.tar：
  - 下载链接: https://pan.baidu.com/s/1kIgOTNZ7txWf_Vyzh-bdEQ?pwd=jwjx 提取码: jwjx
  - podman用法与docker基本一样，将docker命令替换为podman即可
  - 要使用cuda需安装CDI：
    - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
    - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html
  - 并使用如下命令创建容器： <br>
  `podman run -it --device nvidia.com/gpu=all --ipc=host --net=host -v /home/zzl/code:/mnt -e DISPLAY=$DISPLAY --name=agent localhost/zhangzongliang/agent:v6-torch2.6.0-cu12.4` <br>
  - 注意：podman在容器里直接使用root账户，而不使用其他普通账户
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
  - 注意：podman在容器里直接使用root账户，而不使用其他普通账户
  - python interpreter: `/opt/conda/bin/python` <br>



## Execution
Run a file in the `examples` folder


## Contribution

1.  Fork the repository
2.  Create Feat_xxx branch
3.  Commit your code
4.  Create Pull Request

## Tips

#### podman
更新后显卡驱动，需重新生成显卡配置文件，否则可能无法启动容器： <br>
`sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` <br>
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html <br>
https://github.com/NVIDIA/nvidia-container-toolkit/issues/450

用podman从dockerhub下载镜像只需在下载地址前加docker.io，例如： <br>
`podman pull docker.io/pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime`

#### windows vscode 远程图形连接 linux 服务器里的podman容器，步骤如下：
vscode 安装 remote ssh 插件 <br>
windows 安装 VcXsrv： https://sourceforge.net/projects/vcxsrv/ <br>
启动VcXsrv时设置Display number（例如设为10），勾选 Disable access control <br>
在环境变量里添加DISPLAY变量，例如值为localhost:10.0 <br>
在VcXsrv安装目录的X0.hosts文件里添加Linux服务器IP <br>
通过vscode ssh到linux服务器，配置x11转发，例如：<br>
```
Host 192.168.101.202
  HostName 192.168.101.202
  Port 22
  User zzl
  ForwardX11 yes
  ForwardX11Trusted yes
  ForwardAgent yes

Host 192.168.188.6
  HostName 192.168.188.6
  Port 22
  User zzl
  ForwardX11 yes
  ForwardX11Trusted yes
  ForwardAgent yes
```
并重新创建podman容器，例如: <br>
`podman run -it --device nvidia.com/gpu=all --ipc=host --net=host --privileged --pids-limit -1 --security-opt seccomp=unconfined -v /home/zzl/code:/mnt -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v $HOME/.Xauthority:/root/.Xauthority:rw -e DISPLAY=$DISPLAY -e XAUTHORITY=/root/.Xauthority --name=fitting localhost/zhangzongliang/induction:v7-torch2.6.0-cu12.4` <br>
可在容器里运行xclock，验证图形连接是否成功。运行xclock需先安装x11-apps：`sudo apt install x11-apps` <br>
若windows换了IP连接linux 服务器，需重启容器 <br>
若不成功，可在linux服务器终端运行`xhost +`试试 <br>
参考：<br>
https://www.cnblogs.com/Bubgit/p/18829192 <br>
https://blog.csdn.net/luokang21/article/details/144370634 <br>
https://zhuanlan.zhihu.com/p/238751551 


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

ray tune 若出现 resource unavilable 可尝试在创建podman容器时添加`--pids-limit -1`，例如: <br>
`podman run -it --device nvidia.com/gpu=all --ipc=host --net=host --privileged --pids-limit -1 --security-opt seccomp=unconfined -v /home/zzl/code:/mnt -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v $HOME/.Xauthority:/root/.Xauthority:rw -e DISPLAY=$DISPLAY -e XAUTHORITY=/root/.Xauthority --name=fitting localhost/zhangzongliang/induction:v7-torch2.6.0-cu12.4` <br>





