---
layout: post
title: Local RAG LLM Project
date: 2025-03-10
categories: Project
tags:
  - AI
  - RAG
  - LLM
---
# Project MythingLLM in Box which is pull from Anythingllm 
# Part 1, software prepare/dev
## 1. adding local images into rag database by smbshare client connection [done]

## 2. mount the smb share and adding file into Mythingllm rag database [done]

   ![image-20250319220048156](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/image-20250319220048156.png)

## 3. embedding those image to vector database [done], adding asImage.js to handle those 

4. add search bar to search the vector database cross the worksapce 

   [image-20250319215722554]
   <img src="/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/image-20250319215722554.png" alt="image-20250319215722554" />
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
- eGPU work 
![[Pasted image 20250325111315.png]]
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
https://www.youtube.com/watch?v=J0z09Ddr58w
```
git clone --depth=1 --branch rpi-6.6.y-gpu https://github.com/Coreforge/linux.git
or git clone --branch rpi-6.6.y --depth 1  https://github.com/raspberrypi/linux.git
```
```
sudo su
sudo apt install git bc bison flex libssl-dev make libncurses5-dev

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
1. Before compiling the kernel, run `make menuconfig` and select the options:  
    1. Kernel Features > Page Size > 4 KB (for Box86 compatibility)  
    2. Kernel Features > Kernel support for 32-bit EL0 > Fix up misaligned multi-word loads and stores in user space  
    3. Kernel Features > Fix up misaligned loads and stores from userspace for 64bit code  
    4. Device Drivers > Graphics support > AMD GPU (optionally SI/CIK support too)  
    5. Device Drivers > Graphics support > Direct Rendering Manager (XFree86 4.1.0 and higher DRI support) > Force Architecture can write-combine memory
nano .config
  
Rewrite the part marked 'CONFIG_LOCALVERSION='○○○○' with an easy-to-understand name. This time, it was written as 'CONFIG_LOCALVERSION='-v8_16k', so I changed it to 'CONFIG_LOCALVERSION='-v8_16kradeon' and saved it.


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
255          
zphilip@raspberrypi:~ $ cat /sys/class/hwmon/hwmon5/fan1_input  
148                                                                               
zphilip@raspberrypi:~ $ echo 128 | sudo tee /sys/class/hwmon/hwmon5/pwm1  
128                                                                               
zphilip@raspberrypi:~ $ cat /sys/class/hwmon/hwmon5/fan1_input                    
1939                                                                              
zphilip@raspberrypi:~ $ cat /sys/class/hwmon/hwmon5/pwm1_enable                   
1                                                                                 
zphilip@raspberrypi:~ $ echo 200 | sudo tee /sys/class/hwmon/hwmon5/pwm1

sudo apt-get install neofetch
## sudo nano /etc/X11/xorg.conf.d/20-amdgpu.conf
## cat /sys/class/hwmon/hwmon4/power1_average

make kernelversion
 echo 1 | sudo tee /sys/class/hwmon/hwmon4/fan1_enable
 echo 128 | sudo tee /sys/class/hwmon/hwmon4/pwm1 128
```

---
2025-03-21
## 2, Make llama.cpp ()
make the llama.cpp.. 
![](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/Pasted%20image%2020250321002459.png)

```
# Install dependencies: Vulkan SDK, glslc, and cmake
sudo apt install -y libvulkan-dev glslc cmake

# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build with Vulkan support
cmake -B build -DGGML_VULKAN=1
cmake --build build --config Release

# Download llama3.2:3b
cd models && wget https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf

# Run it.
cd ../
./build/bin/llama-cli -m "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf" -p "Why is the blue sky blue?" -e -ngl 100 -t 4

# You should see in the output, ggml_vulkan detected your GPU. For example:
# ggml_vulkan: Found 1 Vulkan devices:
# ggml_vulkan: 0 = AMD Radeon RX 6700 XT (RADV NAVI22) (radv) | uma: 0 | fp16: 1 | warp size: 64

```
```
./build/bin/llama-cli -m "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf" -p "Why is the blue sky blue?" -e -ngl 100 -t 4
./build/bin/llama-cli -m "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf" -p "What is quantum computing and why is it important?" -e -ngl 100 -t 4

```

it is failure due to the "buss error" , it might because the extension board only support PCIE2.. I try several different smaller gguf model but it didn't work out.  finally the PCIE will downgrade PICE gen1.. so I decide to switch the extension board firstly.

## 3, I buy another extension board with only one SSD slot and support gen3

![](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/Snipaste_2025-03-23_08-37-06.png)
![](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/Snipaste_2025-03-23_08-40-23.png)

![](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/Pasted%20image%2020250325112700.png)

### Somehow the RX580 don't work anymore... probably accidently I broke it :( , Orz..
Finally get one AMD RX 6700 xt 
![](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/Pasted%20image%2020250327165309.png)
Run on PCIe Gen3, 
![](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/Pasted%20image%2020250327165648.png)
Token speed have 27tokens/seconds
![](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/Pasted%20image%2020250327164948.png)
./llama-bench -p 0 -n 512 -m ./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf -m ./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf.2 -m ./models/llava-llama-3-8b-v1_1-int4.gguf
# Part 3, setup llam.cpp server with GPU support
## 1, llama.cpp in docker + vulkan + Amdgpu + arm64..
- ollama have no arm version and also have vulkan support version ...看来要自己造轮子？
- there is vulkan version post , [https://github.com/whyvl/ollama-vulkan/issues/7#issuecomment-2660836871](https://github.com/whyvl/ollama-vulkan/issues/7#issuecomment-2660836871)...不知道行不行
- there are also llama.cpp .devops/vulkan.Dockerfile, 
	- create docker using llama.cpp vulkan.dockerfile :  docker build -t llama-cpp-vulkan -f .devops/vulkan.Dockerfile .
- Above docker file all stuck on vulkan sdk installation, it seems no one try it in arm64 yet.
- there are no offical vulkan sdk version for arm64, luckly someone do it , https://github.com/jakoch/vulkan-sdk-arm , download it and install it locally , something like below :
-  Install Vulkan SDK from local file (ARM version)
```
		ENV VULKAN_SDK_VERSION=1.4.309.0
		ENV VULKAN_SDK_PATH=/opt/vulkan-sdk
		
		#install vulkan sdk from local file, arm64 version cannot install from apt
		ENV VULKAN_SDK_VERSION=1.4.309.0
		ENV VULKAN_SDK_PATH=/opt/vulkan-sdk
		RUN wget https://github.com/jakoch/vulkan-sdk-arm/releases/download/1.4.309.0/vulkansdk-ubuntu-22.04-arm-1.4.309.0.tar.xz -O /tmp/vulkan-sdk.tar.xz \
		    && mkdir -p /opt/vulkan-sdk \
		    && tar -xJf /tmp/vulkan-sdk.tar.xz -C /opt/vulkan-sdk --strip-components=1 \
		    && rm /tmp/vulkan-sdk.tar.xz
		
		#Set environment variables for Vulkan SDK
		ENV VULKAN_SDK=/opt/vulkan-sdk
		ENV PATH="VULKAN_SDK/bin:PATH"
		ENV LD_LIBRARY_PATH="VULKAN_SDK/lib:LD_LIBRARY_PATH"
		ENV VK_ICD_FILENAMES="$VULKAN_SDK/etc/vulkan/icd.d"
		ENV VK_LAYER_PATH="$VULKAN_SDK/etc/vulkan/layer.d"
```
- then problem is there are no glslc...it can be installed in debian bookworm version, but not in ubuntu and debian blueye version.
	- build glslc for arm64+ubuntue   --- this is not glslc..it is glslang it already in the vulkansdk..
			git clone https://github.com/KhronosGroup/glslang.git
			cd glslang
			./update_glslang_sources.py
			cmake -B build -DCMAKE_BUILD_TYPE=Release
			cmake --build build
			sudo cmake --install build
	- https://github.com/google/shaderc/tree/v2023.8
	- https://github.com/KhronosGroup/SPIRV-Tools
	- build glslc (https://github.com/google/shaderc/)...
		```
		RUN apt-get update && apt-get install -y \
		    spirv-tools \
		    spirv-headers \
		    glslang-tools \
		    glslang-dev
		
		RUN git clone --branch vulkan-sdk-1.4.309 https://github.com/KhronosGroup/SPIRV-Tools.git \
		    && cd SPIRV-Tools \
		    && git clone --branch vulkan-sdk-1.4.309 https://github.com/KhronosGroup/SPIRV-Headers.git external/spirv-headers \
		    && cmake -GNinja -B build -DCMAKE_BUILD_TYPE=Release \
		    && cmake --build build --target install
		
		# Create the build directory inside the shaderc folder
		RUN git clone --branch v2023.8 --single-branch https://github.com/google/shaderc.git && \
		    cd shaderc && \
		    ./utils/git-sync-deps
		
		# Install Python symlink first
		RUN apt-get update && apt-get install -y python-is-python3
		
		WORKDIR /app/shaderc/build
		RUN cmake -GNinja \
		    -DCMAKE_BUILD_TYPE=Release \
		    -DCMAKE_INSTALL_PREFIX=/usr/local \
		    .. \
		    && ninja install
		```
- using debian:bookworm , vulkaninfo show something and llama.cpp can sucessfully compiled . but run will failed with "Bus error (core dumped) " unsure why-
```
		FROM debian:bookworm 
		# Install system dependencies
		RUN apt-get update && \
		    apt-get install -y --no-install-recommends \
		    python3-pip \
		    python3-dev \
		    vulkan-tools \
		    libvulkan1 \
		    libvulkan-dev \
		    mesa-vulkan-drivers \
		    mesa-common-dev \
		    vulkan-validationlayers \
		    libvulkan-dev \
		    glslang-tools \
		    glslc \
		    cmake \
		    git \
		    wget \
		    ninja-build \      
		    build-essential \  
		    && rm -rf /var/lib/apt/lists/*
		
		RUN apt-get update && \
		    apt-get install -y --no-install-recommends \
		    gnupg \                
		    ca-certificates \    
		    && rm -rf /var/lib/apt/lists/*
		
		# Verify installation
		#RUN vulkaninfo
		WORKDIR /app/llama.cpp
		
		# Build with Vulkan support
		RUN cmake -B build -DGGML_VULKAN=1
		RUN cmake --build build --config Release
		
		# Install llama-cpp-python with Vulkan support
		RUN CMAKE_ARGS="-DGGML_VULKAN=ON" pip3 install --no-cache-dir llama-cpp-python	
```
	source /path/to/your/venv/bin/activate

in the docker container, running : /llama.cpp/build/bin/llama-cli -m "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf" -p "Why is the blue sky blue?" -e  -ngl 100 -t 4 , it failed like below
```
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon RX 6700 XT (RADV NAVI22) (radv) | uma: 0 | fp16: 1 | warp size: 32 | shared memory: 65536 | matrix cores: none
build: 4984 (5d016702) with cc (Debian 12.2.0-14) 12.2.0 for aarch64-linux-gnu
main: llama backend init
main: load the model and apply lora adapter, if any
llama_model_load_from_file_impl: using device Vulkan0 (AMD Radeon RX 6700 XT (RADV NAVI22)) - 12032 MiB free
llama_model_loader: loaded meta data with 35 key-value pairs and 255 tensors from models/Llama-3.2-3B-Instruct-Q4_K_M.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
...........
load_tensors: loading model tensors, this can take a while... (mmap = true)
make_cpu_buft_list: disabling extra buffer types (i.e. repacking) since a GPU device is available
load_tensors: offloading 28 repeating layers to GPU
load_tensors: offloading output layer to GPU
load_tensors: offloaded 29/29 layers to GPU
load_tensors:      Vulkan0 model buffer size =  1918.35 MiB
load_tensors:   CPU_Mapped model buffer size =   308.23 MiB
...........................................................................
llama_context: constructing llama_context
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 4096
llama_context: n_ctx_per_seq = 4096
llama_context: n_batch       = 2048
llama_context: n_ubatch      = 512
llama_context: causal_attn   = 1
llama_context: flash_attn    = 0
llama_context: freq_base     = 500000.0
llama_context: freq_scale    = 1
llama_context: n_ctx_per_seq (4096) < n_ctx_train (131072) -- the full capacity of the model will not be utilized
Bus error (core dumped)
```

---
2025-03-30
==Done... finally the llama.cpp work in the docker container!!!  performance is good==
![](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/Pasted%20image%2020250330014803.png)
Llama.cpp server up..
![](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/Pasted%20image%2020250330102505.png)
## 2, llava supporting
![](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/DSC_0195.jpg )
[llava-1.6-mistral-7b-gguf](https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf), llava-1.6-mistral-7b/llava-v1.6-mistral-7b.Q4_K_M.gguf, 4.37GB

![](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/Pasted%20image%2020250331001638.png)

ggml_llava-v1.5-7b, llava-v1.5-7b/ggml-model-q4_k.gguf -- 4GB
![](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/Pasted%20image%2020250331001833.png)

llava-llama-3-8b-v1_1-int4.gguf
![](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/Pasted%20image%2020250331003144.png)
## 3, using llama.cpp-python for llava and embedding 

python3 -m llama_cpp.server --config_file config.json
it seems llamacpp-python have much worser performance than pure llama.cpp 
![](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/Pasted%20image%2020250331153457.png)
the token measurement might not correct ... I test it with same picture 

![](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/Pasted%20image%2020250331233126.png)
time is about 2 times, but the tokensper second  is about 19.42/7.02=2.77
![](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/Pasted%20image%2020250331233150.png)
![](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/Pasted%20image%2020250331233215.png)

**using llama.cpp native server** (https://github.com/ggml-org/llama.cpp/blob/master/examples/server/README.md)
 - nohup /llama.cpp/build/bin/llama-server -m models/Llama-3.2-3B-Instruct-Q4_K_M.gguf -c 4096 --host 0.0.0.0 --port 8000 --n-gpu-layers 99 >chat.log 2>&1 &
 - nohup /llama.cpp/build/bin/llama-server --model models/nomic-embed-text-v1.Q8_0.gguf -c 8192 -b 8192 --host 0.0.0.0 --port 8001  --n-gpu-layers 99 --embedding >embed.log 2>&1 &
 - since current llama.cpp don't support multimodal , I switch to llama.cpp python to startup the multimodal : python3 -m llama_cpp.server --config_file config.json  >llava.log 2>&1 &   
![](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/Pasted%20image%2020250401162556.png)
#### the environment value used for mythingllm ：
export EMBEDDING_BASE_PATH=http://llamacpp.server:8001                                             
export LLAMACPP_MODEL_PREF="llama-3-2b"                                                             
export IMAGE2TEXT_MODEL_PREF="llava-v1.5-7b"                                           
export EMBEDDING_MODEL_DIM=768                                                                      
export IMAGE2TEXT_BASE_PATH="http://llamacpp.server:8002"

For now I will using above setting to make the server (llama-server + llama_cpp.server) run for embedding, llm and image2text....
still not working for ollama-vulkan-arm building: 
+ https://github.com/whyvl/ollama-vulkan/issues/7#issuecomment-2660836871
+ https://github.com/nasrally/ollama-vulkan/commit/ccd533a2df4a333a0da5f95ac8f5cddda071c5a7.. fix and remove the libcap.so needs from libvulkan.so...
![](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/Pasted%20image%2020250408110109.png)

## 3 running mythingllm +  rewriteQuestionChainT5 
 - Due to using the quantized model , so the l2 distance from the search result will be much smaller than before 
	 This behavior actually makes sense once you understand how **quantized models** (like `nomic-embed-text-v1.Q8_0.gguf`) work and how **distance metrics** relate to vector **scaling and normalization**.	
	### 🔍 Why are the L2 distances so much smaller with `.Q8_0.gguf`?
	
	When you're using:
		- `nomic-embed-text-v1.Q8_0.gguf`: This is a **quantized model** (int8), which means the weights are approximated to 8-bit values to reduce model size and speed up inference. However, this **impacts the output vector values** — they may be:    
	    - Smaller in magnitude        
	    - Not normalized the same way as the full-precision version        
	    - Less precise in fine detail        
	Versus:
	- `nomic-embed-text-v1` full model (e.g., via HuggingFace or unquantized GGUF): Higher-precision floating point vectors are output — often normalized or scaled consistently.
	  
	### 📐 Why does L2 distance behave differently?
	
	- L2 (Euclidean) distance is **sensitive to vector magnitude**. So if your embedding vectors are small (say each component is ~0.01), the L2 distance between two vectors will also be small (0.1 to 1.5 range is normal in that case).    
	- If vectors are larger (each component ~10–30), then distances can go from 100 to 500+.  
	### ✅ What you can do
	
	#### 1. **Normalize the vectors manually**
	
	If the quantized model doesn't normalize embeddings, you can do it yourself:
	
	```python
	import numpy as np
	
	def normalize(vec):
	    return vec / np.linalg.norm(vec)
	```
	
	Normalize both the stored embeddings and the query before computing L2 or cosine distance.
	
	#### 2. **Switch to cosine similarity (recommended for text embeddings)**
	Cosine similarity is **scale-invariant**, so it works better when embedding vectors have inconsistent magnitudes (as is the case with quantized models).
	- Use **cosine similarity or cosine distance** instead of L2:    
	```python
	from sklearn.metrics.pairwise import cosine_similarity
	
	similarity = cosine_similarity([vec1], [vec2])  # result between -1 and 1
	```
	#### 3. **Be consistent**
	Make sure:
	- You **store** all embeddings from the **same model**, and   
	- You use the **same distance metric** when storing and querying.   

| Model Type             | Vector Magnitude | L2 Distance Range | Notes         |
| ---------------------- | ---------------- | ----------------- | ------------- |
| Full precision         | Large (~10–30)   | ~100–500          | More accurate |
| Quantized `.Q8_0.gguf` | Small (~0.01–1)  | ~0.1–2.0          | Less precise  |
 - rewriteQuestionChainT5