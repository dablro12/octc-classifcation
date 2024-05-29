import torch 
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import os

# Grad-CAM 함수 정의
def apply_gradcam(model, image_path, target_layer, device, type = 'Ours'):
    # 원본 이미지를 엽니다.
    original_img = Image.open(image_path).convert('RGB')
    
    # 원본 이미지의 크기를 얻습니다.
    original_size = original_img.size


    # 이미지 전처리 파이프라인을 설정합니다.
    if type == 'Ours':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])

    # 전처리된 이미지 텐서를 생성합니다.
    img_tensor = transform(original_img).unsqueeze(0).to(device)

    # Grad-CAM
    gradients, activations = [], []
    
    # Forward hook 함수 정의
    def save_activation(module, input, output):
        activations.append(output)

    # Backward hook 함수 정의
    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Forward와 Backward hook 등록
    forward_hook = target_layer.register_forward_hook(save_activation)
    backward_hook = target_layer.register_backward_hook(save_gradient)

    # 모델을 통한 예측 실행
    output = model(img_tensor)

    # Binary classification (sigmoid activation)
    one_hot_output = torch.sigmoid(output)
    one_hot_output.backward()
    model.zero_grad()

    # Hook 제거
    forward_hook.remove()
    backward_hook.remove()

    # Activation과 Gradient 가져오기
    activation = activations[0].cpu().data.numpy()[0]
    gradient = gradients[0].cpu().data.numpy()[0]

    # Gradient 가중치 계산
    pooled_gradients = np.mean(gradient, axis=(1, 2))

    # Activation과 Gradient를 사용하여 CAM 계산
    cam = np.zeros(activation.shape[1:], dtype=np.float32)
    for i, w in enumerate(pooled_gradients):
        cam += w * activation[i, :, :]

    # Normalize CAM
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)

    # Resize CAM to the original image size
    cam = cv2.resize(cam, original_size, interpolation=cv2.INTER_LINEAR)
    cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB) 
    
    # Overlay CAM on the original image
    original_img_np = np.array(original_img)  # Convert PIL image to numpy array
    overlayed_img = cv2.addWeighted(original_img_np, 1.0, cam, 0.5, 0)
    
    return original_img_np, overlayed_img