import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# モデルの定義
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

# モデルパスの指定（手動でダウンロードしたファイルのパスを使用）
model_path = 'path/to/downloaded/RealESRGAN_x4plus.pth'

# Real-ESRGANerの初期化
upscaler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=True
)

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = np.array(img)
    return img

# 画像の読み込み
input_image_path = '/Users/matsuokoujirou/Documents/IMG_5303'
img = load_image(input_image_path)

# 画像の超解像
output_img, _ = upscaler.enhance(img, outscale=4)

# 結果の保存
output_image_path = 'output_image.jpg'
output_img = Image.fromarray(output_img)
output_img.save(output_image_path)

input_image_path = '/Users/matsuokoujirou/Documents/IMG_5303'

