model="ndt"
model_path="./checkpoints/experiment1/ndt/2025-03-28T16:48.pth"
lr=1e-4
tree_depth=8
epochs=40

# training model
python main.py --arch=${model} --tree-depth=${tree_depth} --lr=${lr} --epochs=${epochs} --model-path=${model_path} --eval
