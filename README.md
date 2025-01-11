# Image inpainting
Project for training and evaluating PartialConv and other model of image inpainting task.
Original:
- https://github.com/fenglinglwb/MAT
- https://github.com/naoto0804/pytorch-inpainting-with-partial-conv

---
## **1. Prerequisites**
Project work with Python 3.12.

Install all requirement depencencies:
```bash
pip install -r requirements.txt
```
## **2. Mask creation**
Create mask for image. 
```bash
cd libs/mask_generate
python mask_generate.py --type=1 --image_dir='' --mask_dir='' --masked_dir='' 
```

## **3. Demo**
```bash
cd libs/MAT
python generate_image_test.py
```
```bash
cd libs/PartialConv
python test.py
```

## **4. Training**
There are notebooks for training PartialConv and MAT in /notebooks









