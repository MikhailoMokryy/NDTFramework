model="ndf_kontschieder"
model_path="./checkpoints/ckpt-ndf_kontschieder-20250202142532.pth"

# test model
python main.py --arch=${model} --model-path=${model_path} --eval
