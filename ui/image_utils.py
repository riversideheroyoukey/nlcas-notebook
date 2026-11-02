# ui/image_utils.py
# Version: v3.0.0 (Recovery Build)

import base64
from io import BytesIO
from PIL import Image, ImageChops

def preprocess_image(image_data_b64):
    img = Image.open(BytesIO(base64.b64decode(image_data_b64))).convert("RGB")
    
    # 白背景で塗りつぶした画像と比較して境界を特定
    bg = Image.new("RGB", img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg)
    bbox = diff.convert("L").point(lambda x: 0 if x < 20 else 255).getbbox()
    
    if bbox:
        padding = 50
        img = img.crop((
            max(0, bbox[0] - padding), 
            max(0, bbox[1] - padding), 
            min(img.width, bbox[2] + padding), 
            min(img.height, bbox[3] + padding)
        ))

    if img.width < 1000:
        ratio = 1000 / img.width
        img = img.resize((1000, int(img.height * ratio)), Image.Resampling.LANCZOS)
    
    img.save("debug_preprocessed.png")
    
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')