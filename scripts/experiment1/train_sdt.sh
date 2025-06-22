model="sdt"
epochs=40
max_depth=5
batch_size=64
lr=1e-3
experiment="experiment1"

# training model
python main.py --arch=${model} --epochs=${epochs} --batch-size=${batch_size} --max-depth=${max_depth} --lr=${lr} --experiment=${experiment}
