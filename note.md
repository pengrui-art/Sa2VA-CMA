### 下面是转化为hf格式的命令
```python
python projects/llava_sam2/hf/convert_to_hf.py work_dirs/sa2va_1b/sa2va_1b.py work_dirs/sa2va_1b/iter_196000.pth --save-path work_dirs/sa2va_1b_hf_196000
```
### 下面是传输文件的命令
```bash
rsync -av --progress /data1/pengrui/Project/Sa2VA/work_dirs/sa2va_1b_hf_375568 pengr@10.1.10.191:/data1/pengrui/project/Sa2VA/work_dirs
```
### 运行代码的命令
```python
python projects/llava_sam2/gradio/app_unified.py --share
```
### 多卡评测命令
#### refcoco_eval bash
```bash
projects/llava_sam2/evaluation/dist_test.sh projects/llava_sam2/evaluation/refcoco_eval.py /data1/pengrui/Model/ByteDance/Sa2VA-1B 4 --dataset refcoco --split val --image-folder /data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/ref_seg/refcoco/coco2014/train2014 --data-path /data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/ref_seg --work-dir /data1/pengrui/Project/MySa2VA/work_dirs
  ```
#### refcoco_eval scores（80.2678）
============================================ current  
CIoU: 0.8026784658432007, GIoU: 0.8085440993309021 current  
============================================ current  
RES_refcoco_val successfully finished evaluating current  
{'Acc': np.float32(0.80267847)}  
[rank0]:[W928 21:09:42.729597486 ProcessGroupNCCL.cpp:1538] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())  
#### refcoco+_eval bash
```bash
projects/llava_sam2/evaluation/dist_test.sh \
projects/llava_sam2/evaluation/refcoco_eval.py \
/data1/pengrui/Project/MySa2VA/work_dirs/sa2va_1b/sa2va_1b_hf_375568 \
4 \
--dataset refcoco_plus \
--split val \
--image-folder /data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/ref_seg/refcoco+/coco2014/train2014 \
--data-path /data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/ref_seg \
--work-dir /data1/pengrui/Project/MySa2VA/work_dirs
```
#### refcoco+ score （74.9022）
============================================ current  
CIoU: 0.7490223050117493, GIoU: 0.7602055072784424 current  
============================================ current  
RES_refcoco+_val successfully finished evaluating current  
{'Acc': np.float32(0.7490223)}  
[rank0]:[W928 21:14:24.074726617 ProcessGroupNCCL.cpp:1538] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())  

#### refcocog_eval bash
```python
projects/llava_sam2/evaluation/dist_test.sh \
projects/llava_sam2/evaluation/refcoco_eval.py \
/data1/pengrui/Project/MySa2VA/work_dirs/sa2va_1b/sa2va_1b_hf_375568 \
4 \
--dataset refcocog \
--split val \
--image-folder /data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/ref_seg/refcocog/coco2014/train2014 \
--data-path /data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/ref_seg \
--work-dir /data1/pengrui/Project/MySa2VA/work_dirs
```
 #### refcoco+ score (76.9569)
============================================ current  
CIoU: 0.7695698738098145, GIoU: 0.7741730213165283 current  
============================================ current  
RES_refcocog_val successfully finished evaluating current  
{'Acc': np.float32(0.7695699)}  
[rank0]:[W928 21:27:39.913804889 ProcessGroupNCCL.cpp:1538] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())  
 #### Mevis_U bash
```python
projects/llava_sam2/evaluation/dist_test.sh /data1/pengrui/Project/MySa2VA/projects/llava_sam2/evaluation/ref_vos_eval.py /data1/pengrui/Project/MySa2VA/work_dirs/sa2va_1b/sa2va_1b_hf_375568 4 --dataset MEVIS_U --work_dir work_dirs/refvos_mevis_u_run1
```

```python
python tools/eval/eval_mevis.py work_dirs/refvos_mevis_u_run1/results.json --mevis_exp_path /data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/video_datas/mevis/valid_u/meta_expressions.json --mevis_mask_path /data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/video_datas/mevis/valid_u/mask_dict.json --save_name mevis_valu.json
```
#### Mevis_U score
{  
    "J": 47.32,  
    "F": 55.7,  
    "J&F": 51.51  
}  

#### Ref-DAVIS17 bash
```
projects/llava_sam2/evaluation/dist_test.sh \
/data1/pengrui/Project/MySa2VA/projects/llava_sam2/evaluation/ref_vos_eval.py\
 /data1/pengrui/Project/MySa2VA/work_dirs/sa2va_1b/sa2va_1b_hf_375568 \
  4 \
  --dataset DAVIS \
  --work_dir work_dirs/refvos_DAVIS_run1
```

```
python tools/eval/eval_davis.py work_dirs/refvos_DAVIS_run1/results.json 
--mevis_exp_path /data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/video_datas/davis17/meta_expressions/valid/meta_expressions.json
--mevis_mask_path /data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/video_datas/davis17/valid/mask_dict.pkl
--save_name davis_valu.json
```
#### Ref-DAVIS17 SCORE
{  
    "J": 64.55,  
    "F": 72.74,  
    "J&F": 68.65  
}  


#### ReVOS bash
```
projects/llava_sam2/evaluation/dist_test.sh /data1/pengrui/Project/MySa2VA/projects/llava_sam2/evaluation/ref_vos_eval.py /data1/pengrui/Project/MySa2VA/work_dirs/sa2va_1b/sa2va_1b_hf_375568 4 --dataset REVOS --work_dir work_dirs/refvos_revos_run1
```

```
python /data1/pengrui/Project/MySa2VA/tools/eval/eval_revos.py work_dirs/refvos_revos_run1/results.json 
--exp_path /data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/video_datas/revos/meta_expressions_valid_.json
--mask_path /data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/video_datas/revos/mask_dict.json
--pred_path revos_valu.json
```
#### ReVOS score
{'referring': {'J': np.float64(36.80424600638978), 'F': np.float64(39.55156549520767), 'A': np.float64(94.25567731629394), 'R': np.float64(90.33970607028755), 'JF': np.float64(38.17790575079873)}, 'reason': {'J': np.float64(31.971979797979795), 'F': np.float64(35.65684848484849), 'A': np.float64(93.23529696969696), 'R': np.float64(89.34119191919193), 'JF': np.float64(33.81441414141414)}, 'overall': {'J': np.float64(34.38811290218479), 'F': np.float64(37.604206990028075), 'A': np.float64(93.74548714299544), 'R': np.float64(89.84044899473975), 'JF': np.float64(35.99615994610643)}}
Results saved to /data1/pengrui/Project/MySa2VA/work_dirs/refvos_revos_run1/revos_valid.json
Results saved to /data1/pengrui/Project/MySa2VA/work_dirs/refvos_revos_run1/revos_valid.csv
time: 136.8206 s
