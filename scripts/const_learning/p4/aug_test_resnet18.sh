model="resnet18"
model_path="./checkpoints/const_learning/resnet18/2025-04-11T01:04.pth"
augmentation="gaussian_blur"

python main.py --arch=${model} --hierarchy=induced-resnet18 --loss=SoftTreeSupLoss --model-path=${model_path} --eval --aug=${augmentation}
