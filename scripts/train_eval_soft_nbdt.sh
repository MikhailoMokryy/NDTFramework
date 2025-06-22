model="nbdt"
epochs=50

# training model
python main.py --arch=${model} --epochs=${epochs} --hierarchy=induced-resnet18 --loss=SoftTreeSupLoss
