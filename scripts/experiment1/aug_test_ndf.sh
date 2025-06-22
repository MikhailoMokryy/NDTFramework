model="ndf"
model_path="./checkpoints/experiment1/ndf/2025-03-28T20:56.pth"
tree_depth=8
n_tree=20
augmentation="gaussian_blur"

# training model
python main.py --arch=${model} --tree-depth=${tree_depth} --n-tree=${n_tree} --model-path=${model_path} --eval --aug=${augmentation}
