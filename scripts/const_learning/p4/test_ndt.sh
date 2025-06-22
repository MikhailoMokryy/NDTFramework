model="ndt"
model_path="./checkpoints/const_learning/ndt/2025-04-10T13:13.pth"
tree_depth=8

# training model
python main.py --arch=${model} --model-path=${model_path} --tree-depth=${tree_depth} --eval
