---
layout: page
title: working todo
date: 2024-03-25
categories: working
tags: AI
---

- [x] #task compare the nougat small model and code with the nougat big model what's the different between them.  🔼 🛫 2024-01-09 : ✅ 2024-01-21
- transformer version change to  4.34.1, then it work (static saving safe is in newer version)
- [x] #task based work output of above task let's partially loading the Chinese Bart pre-trained model data 🔼 🛫 2024-01-09 ✅ 2024-02-29
- [x] prepare one small set dataset include Chinese data for training 🔼 🛫 2024-01-09 ✅ 2024-01-21
- small set dataset --- nougat-dataset-test include - ocrpadded, ocrset
- Done. the original bart model ("facebook/mbart-large-50") already has multiple language support include chinese Bart pre-trained ,
-  languages covered
Arabic (ar_AR), Czech (cs_CZ), German (de_DE), English (en_XX), Spanish (es_XX), Estonian (et_EE), Finnish (fi_FI), French (fr_XX), Gujarati (gu_IN), Hindi (hi_IN), Italian (it_IT), Japanese (ja_XX), Kazakh (kk_KZ), Korean (ko_KR), Lithuanian (lt_LT), Latvian (lv_LV), Burmese (my_MM), Nepali (ne_NP), Dutch (nl_XX), Romanian (ro_RO), Russian (ru_RU), Sinhala (si_LK), Turkish (tr_TR), Vietnamese (vi_VN), Chinese (zh_CN), Afrikaans (af_ZA), Azerbaijani (az_AZ), Bengali (bn_IN), Persian (fa_IR), Hebrew (he_IL), Croatian (hr_HR), Indonesian (id_ID), Georgian (ka_GE), Khmer (km_KH), Macedonian (mk_MK), Malayalam (ml_IN), Mongolian (mn_MN), Marathi (mr_IN), Polish (pl_PL), Pashto (ps_AF), Portuguese (pt_XX), Swedish (sv_SE), Swahili (sw_KE), Tamil (ta_IN), Telugu (te_IN), Thai (th_TH), Tagalog (tl_XX), Ukrainian (uk_UA), Urdu (ur_PK), Xhosa (xh_ZA), Galician (gl_ES), Slovene (sl_SI)

The loading such as :  ==MBartForCausalLM.from_pretrained( "facebook/mbart-large-50" ).state_dict()==
- [ ] adding authentication function to android and pc side client. let's start this after Chinese supporting in server side.
- [ ] adding authentication and security function to server side (multiple user supporting). let's start this after Chinese supporting in server side. 
- [x] today's work 📅 2024-01-11 , ⏫  try to modify the ocr padding method to scale the ocr png to size fit  swin-transformer , then padding.  I suspect the training error is caused by too small png 🛫 padding.  current size is 672(w)x896(h)x8.. arxiv size is 816x1056x24.. something wrong. ✅ 2024-02-29
- notice that the original code has the resize function to make the small picture to fit the 886*672 size , so using the original un-padding figure (latex ocr dataset image) to feed the current training. but still have the "repetition error"
- ==(DONE) put the original un-padding figure  and the padded figure together into the training== , it seems that the training have no "repetition error" still unclear why it is so????
	- this have the repetition error... while using bigger dataset from arXiv orignal data
	- I am think the VIT how to train the all blank image... it is lead to some error , or how it is treated , get some test ???
	- how about using all back or white image as training data 
	- how the voice to text treat the white noise? how the blank image is treated 
	- in normal ViT , the fixed size image is always required...,   how about reserve the SWIN transformer architecture, change from low level resolution to high resolution 
	- #todo Need to check the result after above change....
- [x] bleu score ✅ 2024-01-12, the bleu score is noted at [[Transformer_learning#3.2 Bleu Score]]
- [x] Pytorch-view的用法 ✅ 2024-02-29
- 在pytorch中view函数的作用为重构张量的维度，相当于numpy中resize（）的功能，但是用法可能不太一样。如下例所示

```text
>>> import torch
>>> tt1=torch.tensor([-0.3623, -0.6115,  0.7283,  0.4699,  2.3261,  0.1599])
>>> result=tt1.view(3,2)
>>> result
tensor([[-0.3623, -0.6115],
        [ 0.7283,  0.4699],
        [ 2.3261,  0.1599]])
```

1. torch.view(参数a，参数b，...)

在上面例子中参数a=3和参数b=2决定了将一维的tt1重构成3x2维的张量。

2. 有的时候会出现torch.view(-1)或者torch.view(参数a，-1)这种情况。
```text
>>> import torch
>>> tt2=torch.tensor([[-0.3623, -0.6115],
...         [ 0.7283,  0.4699],
...         [ 2.3261,  0.1599]])
>>> result=tt2.view(-1)
>>> result
tensor([-0.3623, -0.6115,  0.7283,  0.4699,  2.3261,  0.1599])
```
由上面的案例可以看到，如果是torch.view(-1)，则原张量会变成一维的结构。
```text
>>> import torch
>>> tt3=torch.tensor([[-0.3623, -0.6115],
...         [ 0.7283,  0.4699],
...         [ 2.3261,  0.1599]])
>>> result=tt3.view(2,-1)
>>> result
tensor([[-0.3623, -0.6115,  0.7283],
        [ 0.4699,  2.3261,  0.1599]])
```

由上面的案例可以看到，如果是torch.view(参数a，-1)，则表示在参数b未知，参数a已知的情况下自动补齐列向量长度，在这个例子中a=2，tt3总共由6个元素，则b=6/2=3。


**Albumentations数据增强方法 **, practice in notebook "/labs/nougat_unet/predict.ipynb", refer page [Albumentations数据增强方法_shiftscalerotate-CSDN博客](https://blog.csdn.net/qq_27039891/article/details/100795846)

- [x] Long Table splitting into small one ✅ 2024-03-07
- almost done with , extract_table/saveHtmlFile/split_row function in crawler__dataset_generate.py .  this function split the html table into different small table.   
- the problem is if the table have multiple figure embedded,  the size of the html is not controlled well
- table size is not good way to estimate.  current only calculate the text/word count 
- [ ] Text emotion /sentiment split, text to voice model study...etc
	- [x] checked the Text emotion and text voice internet resources ✅ 2024-03-07
	- [ ] next to setup one local text to voice model to test 

# Daily Work Notes 2024-03-07
1, in last few days , I try to setup [fofr/cog-face-to-sticker: face-to-sticker (github.com)](https://github.com/fofr/cog-face-to-sticker) in my local machine. it is one model/workflow with ComfyUI --- No successe yet.  It is maybe the memeory issue. I will continue to do this and the final goal is integarated with my website and give some trail.

2, in last few days, I try to combine the FastAPI and gradio together for my AIWorm.cn/Nougat+ website.  some problem goging. 

3, learning content of today :  
- [x] #task PCA [Principal Component Analysis (PCA) (youtube.com)](https://www.youtube.com/watch?v=fkf4IBRSeEc&list=WL&index=111)
- [ ] #task Setup 🛫 2024-03-07 face-to-sticker

# Daily Work Notes 2024-03-08
1, continue doing docker maker for nougat webserver which starting from yesterday
2, continue learning PCA 
3, Start the nougat learning note to record the all result, following is the nougat middle model using partially arxiv data and some Chinese internet data crawlered 
![](/W%26B%20Chart%202024_3_8%2009_20_34.png)
![](/W%26B%20Chart%202024_3_8%2009_20_54.png)
This is the previous small model training loss figure. every Time stop training and restart from checkpoint, you will see the loss is high.. unsure why?

![](/W%26B%20Chart%202024_3_4%2016_21_15%201.png)
Previous Vit+unet transformer test result
![](/W%26B%20Chart%202024_3_8%2009_32_58.png)

## nested regex expression note: 
example：[regex101: build, test, and debug regex](https://regex101.com/r/NsVPFp/3) 
`\(+\*+  (?:[^*(]|(?:\*+[^)*])|(?:\(+[^*(]))*     \*+\)+`
	`/\(+\*+ (?:(?:`
	        `\(+\*+    (?:(?:`
	        
	                   `\(+\*+       (?:[ ^*(  ]  |  (?  :\*+[^)*]  )  |  (?:\(+[^*(])   )*`
	
	                   `\*+\)+        )|[^*(]|(?:\*+[^)*])|(?:\(+[^*(]))*`
	
	        `\*+\)+    )|[^*(]|(?:\*+[^)*])|(?:\(+[^*(]))*`
	 
	`\*+\)+`
[regex - Regular expressions for pattern matching time format in python - Stack Overflow](https://stackoverflow.com/questions/62584721/regular-expressions-for-pattern-matching-time-format-in-python)
### nested regex expression :
`p{1} = \{  (?:[^{}])*   \}`
`p{2} = \{  (?:   (?:   p{1}   ) |  (?:[^{}])  )*   \}`
[regex101: build, test, and debug regex](https://regex101.com/r/JewQGp/1/) https://regex101.com/r/4liwhI/1

`\{  (?:(?:`   
     `\{    (?:(?:`   
           `\{    (?:[^{}])*`    
               `\}   )|(?:[^{}]))*`   
                    `\}    )|(?:[^{}]))*`   
                          `\}`
- replace with (\\begin{([a-z]+?\*?)}) to the \{  so we have following
	`p{1} = (?<!\\)(\\begin{([a-z]+?\*?)})(?:(?!\\begin{\2}|\\end{\2}).)*(\\end{\2})`
	
	`(?:[^{}])* ---- (?:(?!\\begin{\2}|\\end{\2}).)*`  exclude structure 排除结构 , for example `(?<!\\|\$)(\$)([^\$]+)(\$)(?:(?![$|\\]).)`
	https://regex101.com/r/yngAW3/1
	`p{2} = (?<!\\)(\\begin{([a-z]+?\*?)})(?:(?:(?<!\\)(\\begin{([a-z]+?\*?)})(?:(?!\\begin{\2}|\\end{\2}).)*(\\end{\2}))|(?:(?!\\begin{\2}|\\end{\2}).)*)*(\\end{\2})`
	
	`p{3} = (?<!\\)(\\begin{([a-z]+?\*?)})(?:(?:(?<!\\)(\\begin{([a-z]+?\*?)})(?:(?:(?<!\\)(\\begin{([a-z]+?\*?)})(?:(?!\\begin{\2}|\\end{\2}).)*(\\end{\2}))|(?:(?!\\begin{\2}|\\end{\2}).)*)*(\\end{\2}))|(?:(?!\\begin{\2}|\\end{\2}).)*)*(\\end{\2})`
- final version 
	`"(?<!\\)(\\begin{([a-z]+?\*?)})(?:(?:(?<!\\)(\\begin{([a-z]+?\*?)})(?:(?:(?<!\\)(\\begin{([a-z]+?\*?)})(?:(?!\\begin{\6}|\\end{\6}).)*(\\end{\6}))|(?:(?!\\begin{\4}|\\end{\4}).))*(\\end{\4}))|(?:(?!\\begin{\2}|\\end{\2}).))*(\\end{\2})"gm`
	https://regex101.com/r/7zJ0g1/1

	`(?:[ ^*(  ]  |  (?  :\*+[^)*]  )  |  (?:\(+[^*(])   )*`
	`(?:[ ^{}                                        ]   )*`
	`(?:[^(\\begin{([a-z]+?\*?)})+(\\end{\2})+])*`
	
	`)|    [^*(] | (?:\*+[^)*])   |  (?:\(+[^*(])         )*`
	`)|    (?: [^{}])                                   )*`
	`)|    (?: [^ (\\begin{([a-z]+?\*?)})+(\\end{\2})+]) )*`
	
	`(\\begin{([a-z]+?\*?)})+   (?:(?:`
	    `(\\begin{([a-z]+?\*?)})+     (?:(?:`
	            `(\\begin{([a-z]+?\*?)})+     (?:[^*(]|(?:\*+[^)*])|(?:\(+[^*(]))*`
	                     `(\\end{\2})+            )|[^*(]|(?:\*+[^)*])|(?:\(+[^*(]))*`
	                              `(\\end{\2})+       )|[^*(]|(?:\*+[^)*])|(?:\(+[^*(]))*`
	                                      `(\\end{\2})+`
	
	
	                                      `(?:[^{\1}]|(?:\*+[^(\\end{\2})+])|(?:\(+[^{\1}]))*`
	
	                                      `(\\begin{([a-z]+?\*?)})[^(\\end)](.+?)(\\end{\2})(.+?)(\\end{\2})+`
	
	
	
	                                      `(\\begin{([a-z]+?\*?)})+(?:(?:(\\begin{([a-z]+?\*?)})+(?:(?:(\\begin{([a-z]+?\*?)})+(?:[^(\\begin{\2})(\\end{\2})]|(?:\*+[^(\\end{\2})])|(?:\(+[^(\\begin{\2})]))*(\\end{\2})+)[^(\\begin{\2})(\\end{\2})]|(?:\*+[^(\\end{\2})])|(?:\(+[^(\\begin{\2})]))*(\\end{\2})+)[^(\\begin{\2})^(\\end{\2})]|(?:\*+[^(\\end{\2})])|(?:\(+[^(\\begin{\2})]))*(\\end{\2})+`
	
	`(\\begin{([a-z]+?\*?)})(?:[^(\\begin{\2})]|(?:\*+[^(\\end{\2})])|(?:\(+[^(\\begin{\2})]))*(\\end{\2})`                                    
	  `^(?!(red|green|blue)$).*$`
  
  ## hard-disk mount by uuid
	`/dev/sdd        /data/data1     ext4    defaults        0       0`
	`/dev/sdc        /data/data2     ext4    defaults        0       0`
	`/dev/sdb        /data/data3     ext4    defaults        0       0`
		
	`lrwxrwxrwx 1 root root 10 Jan 10 06:24 39e64f6f-4e8d-4f78-93bc-59179f553043 -> ../../sda2`
	`lrwxrwxrwx 1 root root 10 Jan 10 06:24 3a6a811a-10b0-4e56-8675-f6266f6db518 -> ../../dm-0`
	`lrwxrwxrwx 1 root root  9 Jan 10 06:24 799f8d5f-e372-465c-b5a9-aa42db0cee36 -> ../../sdb`
	`lrwxrwxrwx 1 root root  9 Jan 10 06:24 7b800e22-10f5-4b8a-96ca-856b8a77fdb6 -> ../../sdd`
	`lrwxrwxrwx 1 root root  9 Jan 10 06:24 d0e23ef0-ae13-4277-8671-340b4a0fb934 -> ../../sdc`
	
	`scsi-1ATA_INTEL_SSDSC2BB800G6R_BTWA641100TT800HGN -> ../../sda`
	
	`ata-INTEL_SSDSC2BB800G6R_BTWA641100TT800HGN -> ../../sda`
	`ata-TOSHIBA_HDWE160_971UK1WTF56D -> ../../sdd`
	`ata-TOSHIBA_MD04ACA500_55E8K1SFFS9A -> ../../sdc`
	`ata-WDC_WUS721010ALE6L4_VCK8940P -> ../../sdb`
	
	`(base) userroot@userroot:~$ blkid`
	`/dev/sda2: UUID="39e64f6f-4e8d-4f78-93bc-59179f553043" TYPE="ext4" PARTUUID="28c6cfbd-5edc-4e53-8342-b76c1f91dd50"`
	`/dev/sda3: UUID="7dNYf8-rzin-MHJB-895r-HQWt-pL5c-A3tedU" TYPE="LVM2_member" PARTUUID="257300e4-a1fa-4512-bf9b-963d5e4292bc"`
	`/dev/sdc: UUID="d0e23ef0-ae13-4277-8671-340b4a0fb934" TYPE="ext4"`
	`/dev/sdd: UUID="7b800e22-10f5-4b8a-96ca-856b8a77fdb6" TYPE="ext4"`
	`/dev/sdb: UUID="799f8d5f-e372-465c-b5a9-aa42db0cee36" TYPE="ext4"`
	`/dev/mapper/ubuntu--vg-ubuntu--lv: UUID="3a6a811a-10b0-4e56-8675-f6266f6db518" TYPE="ext4"`
	- fstab:
	`UUID=7b800e22-10f5-4b8a-96ca-856b8a77fdb6        /data/data1     ext4    defaults        0       0`
	`UUID=d0e23ef0-ae13-4277-8671-340b4a0fb934        /data/data2     ext4    defaults        0       0`
	`UUID=799f8d5f-e372-465c-b5a9-aa42db0cee36        /data/data3     ext4    defaults        0       0`

# Daily Work Notes 2024-03-11
1. try to rebuild the comfyui's environment/and try cog docker build , the problem might be the environment issue
2. continue the work task "aiworm.cn websit setup" 
3. continue the learning task "pca"

# Daily Work Notes 2024-03-13
1. still failed to setup the face to sticker comfyui environment... this time using the windows comfyui 
2. aiworm.cn websit work onging , almost done.... the problem now for those model accuracy is not enough.   more bigger model? beside this, while using the Chinese and English mixed text , it will translate part of it, it is not what wanted....  it need to dig out
- [ ] mBart or Bert model stopping translate but just inference in nougat task
3. PCA/SVD continue.  
4. stop training in the AutoDL in Epoch 17 ... I will add new code to separate the Chinese and English text in tokenizer... it might be some mistaken I don't found out before :
 BERT多语言版本预训练模型上线前需要对句子进行人工分字
![](/assets/obsidian_assets/Pasted image 20240313095711.png)

```
def trans(sen):
    """
    Translates a sentence containing both Chinese and English characters.
    If a word is in Chinese, it adds the word and its translation.
    Otherwise, it keeps the word unchanged.
    """
    buf = ""
    for word in sen:
        if "\u4e00" <= word <= "\u9fff":
            buf += f'{word} ' + "translation of Chinese character"
        else:
            buf += word
    return buf.replace(' ', ',')

if __name__ == "__main__":
    chline = "包含中文字符的句子。"
    enline = 'The use of citations in evaluative scientometrics as a proxy for quality'

    chenline = "包含英文The，包含中文 use of包含英文citations in evaluative scientometrics as a proxy for quality"
    print(trans(chline))
    print(trans(enline))
    print(trans(chenline))
```

# Daily Work Notes 2024-03-17
![Pasted image 20240318093438.png](/assets/obsidian_assets/Pasted image 20240318093438.png)
# Daily Work Notes 2024-03-25
<img src="/assets/obsidian_assets/Pasted image 20240325101352.png" alt="img" style="zoom: 100%;" />

1. task PCA  learning is finished. the note should be published to blog
2. [ ] publish PCA learning note to blog
3. [ ] start to prepare the article about the nougat/OCR structure improvement
4. finish the improvements about upload file and at same time give parameters to fastapi. 
	1. [python - How to POST both Files and a List of JSON data in FastAPI? - Stack Overflow](https://stackoverflow.com/questions/77254763/how-to-post-both-files-and-a-list-of-json-data-in-fastapi/77279612#77279612)
	2. [python - How to add both file and JSON body in a FastAPI POST request? - Stack Overflow](https://stackoverflow.com/questions/65504438/how-to-add-both-file-and-json-body-in-a-fastapi-post-request)
	3. current using simple way as following: 

```
## Method 1

As described [here](https://fastapi.tiangolo.com/tutorial/request-forms-and-files/), one can define files and form fileds at the same time using `File` and `Form`. Below is a working example. In case you had a large number of parameters and would like to **define them separately from the endpoint**, please have a look at [this answer](https://stackoverflow.com/a/71650857/17865804) on how to create a custom dependency class instead.

**app.py**
	from fastapi import Form, File, UploadFile, Request, FastAPI
	from typing import List
	from fastapi.responses import HTMLResponse
	from fastapi.templating import Jinja2Templates
	
	app = FastAPI()
	templates = Jinja2Templates(directory="templates")
	
	
	@app.post("/submit")
	def submit(
	    name: str = Form(...),
	    point: float = Form(...),
	    is_accepted: bool = Form(...),
	    files: List[UploadFile] = File(...),
	):
	    return {
	        "JSON Payload": {"name": name, "point": point, "is_accepted": is_accepted},
	        "Filenames": [file.filename for file in files],
	    }
	
	
	@app.get("/", response_class=HTMLResponse)
	def main(request: Request):
	    return templates.TemplateResponse("index.html", {"request": request})

test.py:
	import requests
	url = 'http://127.0.0.1:8000/submit'
	files = [('files', open('test_files/a.txt', 'rb')), ('files', open('test_files/b.txt', 'rb'))]
	data = {"name": "foo", "point": 0.13, "is_accepted": False}
	resp = requests.post(url=url, data=data, files=files) 
	print(resp.json())
```

1. start to fastAPI authentication/user data to database???? 
	- save user data
	- security