model="ndf"
model_path="./checkpoints/experiment1/ndf/2025-03-28T20:56.pth"
epochs=40
tree_depth=8
n_tree=20
lr=1e-4

# training model
python main.py --arch=${model} --tree-depth=${tree_depth} --n-tree=${n_tree} --lr=${lr} --epochs=${epochs} --model-path=${model_path} --eval
