import json
import numpy as np
import torch
import cv2
import sys
sys.path.append("..")
from tqdm import tqdm
from segment_anything import sam_model_registry

with open('/hdd1/tb/vg/output.json', 'r') as file:
    data2 = json.load(file)

sam_checkpoint = "/hdd1/tb/sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

device_ids = [ 0,2,3]  
sam = torch.nn.DataParallel(sam, device_ids=device_ids).cuda(device_ids[0])  # 主设备是 device_ids[0]

count = -1
batched_input = []

for item in tqdm(data2):
    length = len(item['regions'])
    for i in range(0, length):
        region = item['regions'][i]
        image_path = region['url']
        x, y, w, h = region['x'], region['y'], region['width'], region['height']

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将 BGR 转换为 RGB
        image_tensor = torch.tensor(image).permute(2, 0, 1).cuda(device_ids[0])  # 将 image 转换为 tensor 并移到主 GPU

        input_box = torch.tensor([[x, y, x+w, y+h]], device=device_ids[0])  # 将 box 转换为 tensor 并移到主 GPU

        batched_input.append({
            'image': image_tensor,
            'original_size': image.shape[:2],
            'boxes': input_box
        })

        if len(batched_input) == 8:
            outputs = sam.module.forward(batched_input, multimask_output=False) 
            for j, output in enumerate(outputs):
                masks = output['masks'].cpu().numpy() 
                for k, mask in enumerate(masks):
                    count = count + 1
                    mask_to_save = (mask.squeeze() * 255).astype(np.uint8)
                    cv2.imwrite(f'./mask/mask{count}.png', mask_to_save)

            batched_input.clear()

if batched_input:
    outputs = sam.module.forward(batched_input, multimask_output=False) 
    for j, output in enumerate(outputs):
        masks = output['masks'].cpu().numpy()  
        for k, mask in enumerate(masks):
            count = count + 1
            mask_to_save = (mask.squeeze() * 255).astype(np.uint8)
            cv2.imwrite(f'./mask/mask{count}.png', mask_to_save)
