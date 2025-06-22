model="nbdt"
model_path="./checkpoints/experiment1/nbdt/2025-03-31T21:55.pth"
experiment="experiment1"
augmentation="gaussian_blur"

# training model
python main.py --arch=${model} --hierarchy=induced-resnet18 --loss=SoftTreeSupLoss --experiment=${experiment} --model-path=${model_path} --eval --aug=${augmentation}
