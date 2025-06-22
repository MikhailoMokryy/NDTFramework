model="resnet18"
model_path="./checkpoints/experiment1/resnet18/2025-03-27T11:37.pth"
augmentation="gaussian_blur"

# training model
python main.py --arch=${model} --model-path=${model_path} --eval --aug=${augmentation}
