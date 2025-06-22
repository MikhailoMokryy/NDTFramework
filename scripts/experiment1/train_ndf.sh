model="ndf"
batch_size=64
tree_depth=8
n_tree=20
lr=1e-4
epochs=40
experiment="experiment1"

# training model
python main.py --arch=${model} --epochs=${epochs} --batch-size=${batch_size} --tree-depth=${tree_depth} --n-tree=${n_tree} --lr=${lr} --experiment=${experiment}
