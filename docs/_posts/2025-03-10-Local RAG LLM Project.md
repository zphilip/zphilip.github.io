---
layout: post
title:  "Local RAG LLM Project"
date:   2025-03-10
categories: Project
tags: AI RAG LLM
---
# Project MythingLLM in Box which is pull from Anythingllm 
# Part 1, software prepare/dev
## 1. adding local images into rag database by smbshare client connection [done]

## 2. mount the smb share and adding file into Mythingllm rag database [done]

   ![image-20250319220048156](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/image-20250319220048156.png)

## 3. embedding those image to vector database [done], adding asImage.js to handle those 

4. add search bar to search the vector database cross the worksapce 

   [image-20250319215722554]<img src="/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/image-20250319215722554.png" alt="image-20250319215722554" />
# Part2, hardware setup for software test
## 1. Test in the embedding/small board , I get the Orange Pi5 AIPro 24G and raspberry pi 5 16G   [Done]
![]()
![](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/20250319232559.jpg)
![](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/20250319232613.jpg)

test result : 
### 1, orange pi 5 AIPro (ubuntu OS) somehow have problem with the NodeJS , it continues report JavaScript heap OOM, No matter how I use the NODE_OPTIONS , 
 - optimize the software to using stream and small  chunk to read file  ...etc
 - using yarn install/add/build will cause javascript OOM...randomly ... no clue why .. something using yarn install/add --no-parallel will success , but no guarantee.. 
 - for example export NODE_OPTIONS=" --max-semi-space-size=4096/8192/16000. etc"  
 - also adjust the sudo sysctl -w vm.max_map_count=262144
 - sudo echo 1/2/3 | sudo tee /proc/sys/vm/drop_caches.. 
 - using the node --expose-gc --optimize_for_size --max-old-space-size=8192 --gc_interval=100 index.js
 - starting to trace the log to debug ... for example node --trace-gc --inspect-brk --expose-gc --optimize_for_size --max-old-space-size=8192 index.js... using chrome://inspect to get the trace/log.. actually the log show there no much heap memory is used... no clue. 
 - finally I give up run the nodejs frontend/server/collector in the orangePi.
 I don't test the Openeuler OS + NodeJS yet.. 
 
### 2, on another hand, while using raspberry pi 5 16G , everything good, no yarn install/build OOM, no running time OOM... so I wonder the it might OS/Orange Pi5 AIPro probably have some unknown limitation. 

### 3, aspberry pi 5 16G have it's own issue..  I using PCIE M.2 extension board to add NVMe SSD, but it seems will have IO problem while the disk access too much and it will cause whole system crash or it cause the wifi/eth connection failure...
 ![](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/20250319235528.png)
 
 4, using M.2 to Qculink eGPU docker connect to external GPU, I try to the old GForce 960 , refer to https://alican-kiraz1.medium.com/run-llm-on-pi5-connecting-an-nvidia-gpu-to-raspberry-pi-5-via-pcie-x4-a6d52c3efd2a  , but it don't work for me.  it can show the hardware information , but  NVIDIA driver  don't work. 
 ![2020250320000509.png](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/Pasted%20image%2020250320000509.png)
![](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/Pasted%20image%2020250320000851.png)
![](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/Pasted%20image%2020250320000646.png)
### 4, Next step I will try the AMD RX580...(RUNNING)
![](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/20250320202621.jpg)
![](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/20250320202644.jpg)
![](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/20250320202702.jpg)

![](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/Pasted%20image%2020250320181652.png)
Finally RUNING!!!
![](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/20250320235130.jpg)
Following the instruction to do the kernal rebuild， refer  to following:
https://gigazine.net/gsc_news/en/20240226-raspberry-pi-5-gpu-graphics-card/ 
https://www.jeffgeerling.com/blog/2024/use-external-gpu-on-raspberry-pi-5-4k-gaming
https://www.jeffgeerling.com/comment/reply/node/3420/comment_node_blog_post
```
git clone --depth=1 --branch rpi-6.6.y-gpu https://github.com/Coreforge/linux.git
sudo su
apt install git bc bison flex libssl-dev make libncurses5-dev

# Download Coreforge's modified memcpy library.
wget https://gist.githubusercontent.com/Coreforge/91da3d410ec7eb0ef5bc8dee24b91359/raw/b4848d1da9fff0cfcf7b601713efac1909e408e8/memcpy_unaligned.c
gcc -shared -fPIC -o memcpy.so memcpy_unaligned.c
sudo mv memcpy.so /usr/local/lib/memcpy.so
sudo vi /etc/ld.so.preload
		
		# Put the following line inside ld.so.preload:
		/usr/local/lib/memcpy.so

KERNEL=kernel_2712
make bcm2712_defconfig
make menuconfig .... 
make -j6 Image.gz modules dtbs

make -j6 modules_install

Next, run the following commands in order to copy the necessary files.  
cp arch/arm64/boot/dts/broadcom/*.dtb /boot/firmware/ 
cp arch/arm64/boot/dts/overlays/*.dtb* /boot/firmware/overlays/  
cp arch/arm64/boot/dts/overlays/README /boot/firmware/overlays/
cp arch/arm64/boot/Image.gz /boot/firmware/$KERNEL.img

apt install -y firmware-amd-graphics

nano /boot/firmware/config.txt
		dtparam=pciex1  
		dtparam=pciex1_gen=3

nano /etc/modprobe.d/blacklist-amdgpu.conf
	blacklist amdgpu

sudo modprobe amdgpu
- rx580 fan control is lost 

ls /sys/class/hwmon/hwmon5/                                                       

manually contro it 
echo 1 | sudo tee /sys/class/hwmon/hwmon5/pwm1_enable                             
cat /sys/class/hwmon/hwmon5/pwm1_max                                             
255                                                                               zphilip@raspberrypi:~ $ cat /sys/class/hwmon/hwmon5/fan1_input                    
148                                                                               zphilip@raspberrypi:~ $ echo 128 | sudo tee /sys/class/hwmon/hwmon5/pwm1          128                                                                               
zphilip@raspberrypi:~ $ cat /sys/class/hwmon/hwmon5/fan1_input                    
1939                                                                              
zphilip@raspberrypi:~ $ cat /sys/class/hwmon/hwmon5/pwm1_enable                   1                                                                                 zphilip@raspberrypi:~ $ echo 200 | sudo tee /sys/class/hwmon/hwmon5/pwm1

sudo apt-get install neofetch
```

---
2025-03-21
## 2, Make llama.cpp ()
