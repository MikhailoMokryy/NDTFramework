model="resnet18"
model_path="./checkpoints/const_learning_p2/resnet18/model-2025-05-03.pth"
augmentation="gaussian_blur"

echo "Pure:"
python main.py --arch=${model} --hierarchy=induced-resnet18 --model-path=${model_path} --eval

echo
echo "Aug:"
python main.py --arch=${model} --hierarchy=induced-resnet18 --model-path=${model_path} --eval --aug=${augmentation}
