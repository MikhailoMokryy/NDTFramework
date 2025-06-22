model="ndt"
model_path="./checkpoints/experiment1/ndt/2025-03-28T16:48.pth"
tree_depth=8
augmentation="gaussian_blur"

# training model
python main.py --arch=${model} --tree-depth=${tree_depth} --model-path=${model_path} --eval --aug=${augmentation}
