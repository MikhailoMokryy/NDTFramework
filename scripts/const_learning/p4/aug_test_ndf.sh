model="ndf"
model_path="./checkpoints/const_learning/ndf/2025-04-10T21:00.pth"
tree_depth=8
n_tree=20
augmentation="gaussian_blur"

# training model
python main.py --arch=${model} --model-path=${model_path} --tree-depth=${tree_depth} --n-tree=${n_tree} --eval --aug=${augmentation}
