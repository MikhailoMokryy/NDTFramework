model="ant_tanno"
batch_size=256

# training model
python main.py --arch=${model} --batch-size=${batch_size}
