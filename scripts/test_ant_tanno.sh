model="ant_tanno"
model_path="./checkpoints/ant_tanno.pth"

# test model
python main.py --arch=${model} --model-path=${model_path} --eval
