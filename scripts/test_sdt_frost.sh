model="sdt_frost"
model_path="./models/sdt_frost.pth"

# test model
python main.py --arch=${model} --model-path=${model_path} --eval
