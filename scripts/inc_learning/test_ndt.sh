model="ndt"
model_path="./checkpoints/inc_learning/ndt/model-2025-05-18.pth"
augmentation="gaussian_blur"
tree_depth=8

echo "Pure:"
python main.py --arch=${model} --model-path=${model_path} --tree-depth=${tree_depth} --eval

echo
echo "Aug:"
python main.py --arch=${model} --model-path=${model_path} --tree-depth=${tree_depth} --eval --aug=${augmentation}
