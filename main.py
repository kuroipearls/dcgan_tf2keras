import argparse
from GAN import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def parse_args():
	desc = "DCGAN Implementation using TF2 & Keras"
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--epoch', type=int, default=30)
	parser.add_argument('--batch_size', type=int, default=64)

	parser.add_argument('--g_lr', type=float, default=0.0003)
	parser.add_argument('--d_lr', type=float, default=0.0003)

	parser.add_argument('--z_dim', type=int, default=128)
	# parser.add_argument('--img_size', type=int, default=128)

	return parser.parse_args()

def main():
	# parse the arguments. 
	args = parse_args()

	# use mnist data.
	batch_size = args.batch_size
	(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
	all_digits = np.concatenate([x_train, x_test])
	all_digits = all_digits.astype("float32") / 255
	all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
	dataset = tf.data.Dataset.from_tensor_slices(all_digits)
	dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(32)

	# initialize the networks.
	gan = GAN(args)
	# compile the networks. 
	gan.compile(
	    d_optimizer=keras.optimizers.Adam(learning_rate=args.d_lr),
	    g_optimizer=keras.optimizers.Adam(learning_rate=args.g_lr),
	    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
	)
	# start training. 
	gan.fit(
	    dataset, epochs=args.epoch, callbacks=[GANMonitor(num_img=3, z_dim=args.z_dim)]
	)

if __name__ == '__main__':
	main()