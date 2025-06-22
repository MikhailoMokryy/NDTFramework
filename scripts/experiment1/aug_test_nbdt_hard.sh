model="nbdt"
model_path="./checkpoints/experiment1/nbdt/2025-04-01T14:11.pth"
augmentation="gaussian_blur"

python main.py --arch=${model} --hierarchy=induced-resnet18 --loss=HardTreeSupLoss --model-path=${model_path} --eval --aug=${augmentation}
