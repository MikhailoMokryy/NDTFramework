model="ndt"
tree_depth=8
lr=1e-4
batch_size=64
epochs=40
experiment="experiment1"

# train model
python main.py --arch=${model} --epochs=${epochs} --tree-depth=${tree_depth} --batch-size=${batch_size} --lr=${lr} --experiment=${experiment}
