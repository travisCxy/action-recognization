import os
from tensorflow.python import pywrap_tensorflow

checkpoint_path = os.path.join('tf_rgb_imagenet',"model.ckpt")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
	print("tensor name:",key)
	#print(reader.get_tensor(key))
