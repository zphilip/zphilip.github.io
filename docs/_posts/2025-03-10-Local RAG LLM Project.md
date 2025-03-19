---
layout: post
title:  "Local RAG LLM Project"
date:   2025-03-10
categories: Project
tags: AI RAG LLM
---
# Project MythingLLM in Box which is pull from Anythingllm 
1. adding local images into rag database by smbshare client connection [done]

2. mount the smb share and adding file into Mythingllm rag database [done]

   ![image-20250319220048156](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/image-20250319220048156.png)

3. embedding those image to vector database [done], adding asImage.js to handle those 

4. add search bar to search the vector database cross the worksapce 

   [image-20250319215722554]<img src="/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/image-20250319215722554.png" alt="image-20250319215722554" />
5. Test in the embedding/small board , I get the Orange Pi5 AIPro 24G and raspberry pi 5 16G   [Done]
![]()
![](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/20250319232559.jpg)
![](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/20250319232613.jpg)

test result : 
1, orange pi 5 AIPro (ubuntu OS) somehow have problem with the NodeJS , it continues report JavaScript heap OOM, No matter how I use the NODE_OPTIONS , 
 - optimize the software to using stream and small  chunk to read file  ...etc
 - using yarn install/add/build will cause javascript OOM...randomly ... no clue why .. something using yarn install/add --no-parallel will success , but no guarantee.. 
 - for example export NODE_OPTIONS=" --max-semi-space-size=4096/8192/16000. etc"  
 - also adjust the sudo sysctl -w vm.max_map_count=262144
 - sudo echo 1/2/3 | sudo tee /proc/sys/vm/drop_caches.. 
 - using the node --expose-gc --optimize_for_size --max-old-space-size=8192 --gc_interval=100 index.js
 - starting to trace the log to debug ... for example node --trace-gc --inspect-brk --expose-gc --optimize_for_size --max-old-space-size=8192 index.js... using chrome://inspect to get the trace/log.. actually the log show there no much heap memory is used... no clue. 
 - finally I give up run the nodejs frontend/server/collector in the orangePi.
 I don't test the Openeuler OS + NodeJS yet.. 
 
 2, on another hand, while using raspberry pi 5 16G , everything good, no yarn install/build OOM, no running time OOM... so I wonder the it might OS/Orange Pi5 AIPro probably have some unknown limitation. 

3, aspberry pi 5 16G have it's own issue..  I using PCIE M.2 extension board to add NVMe SSD, but it seems will have IO problem while the disk access too much and it will cause whole system crash or it cause the wifi/eth connection failure...
 ![assets/2025-03-10 Local RAG LLM Project.assets/20250319235528.png]
 
 4, using M.2 to Qculink eGPU docker connect to external GPU, I try to the old GForce 960 , refer to https://alican-kiraz1.medium.com/run-llm-on-pi5-connecting-an-nvidia-gpu-to-raspberry-pi-5-via-pcie-x4-a6d52c3efd2a , but it don't work for me.  it can show the hardware information , but  NVIDIA driver  don't work. 
 ![2020250320000509.png](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/Pasted%20image%2020250320000509.png)
![](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/Pasted%20image%2020250320000851.png)
![](/assets/2025-03-10%20Local%20RAG%20LLM%20Project.assets/Pasted%20image%2020250320000646.png)
5, Next step I will try the AMD RX580...