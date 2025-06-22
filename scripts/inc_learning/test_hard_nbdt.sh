model="hard_nbdt"
model_path="./checkpoints/inc_learning/hard_nbdt/model-2025-05-18.pth"
augmentation="gaussian_blur"

echo "Pure:"
python main.py --arch=${model} --hierarchy=induced-resnet18 --loss=HardTreeSupLoss --model-path=${model_path} --eval

echo
echo "Aug:"
python main.py --arch=${model} --hierarchy=induced-resnet18 --loss=HardTreeSupLoss --model-path=${model_path} --eval --aug=${augmentation}
