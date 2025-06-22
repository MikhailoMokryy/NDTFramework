model="ndf"
model_path="./checkpoints/const_learning_p05/ndf/model-2025-05-22.pth"
tree_depth=8
n_tree=20
augmentation="gaussian_blur"

echo "Pure:"
python main.py --arch=${model} --model-path=${model_path} --tree-depth=${tree_depth} --n-tree=${n_tree} --eval

echo
echo "Aug:"
python main.py --arch=${model} --model-path=${model_path} --tree-depth=${tree_depth} --n-tree=${n_tree} --eval --aug=${augmentation}
