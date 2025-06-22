model="resnet18"
model_path="./models/resnet18.pth"

# test model
python main.py --arch=${model} --model-path=${model_path} --eval
