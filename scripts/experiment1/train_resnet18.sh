model="resnet18"
lr=1e-1
epochs=40
experiment="experiment1"

# training model
python main.py --arch=${model} --lr=${lr} --epochs=${epochs} experiment=${experiment}
