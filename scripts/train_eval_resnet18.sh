model="resnet18"
epochs=50

# training model
python main.py --arch=${model} --epochs=${epochs}
