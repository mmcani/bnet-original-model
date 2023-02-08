import os
import glob
import tensorflow as tf
from tensorflow import keras as k
from keras import layers as l
from keras.models import load_model
import torch
import torchaudio

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Mute Tensorflow

# Model name/path
MODEL_PATH = 'BirdNET_2022_3K_V2.2.h5'


# Define custom Layer
class LinearSpecLayer(l.Layer):
    def __init__(self, sample_rate=48000, spec_shape=(64, 384), frame_step=374, frame_length=512, fmin=250,
                 fmax=15000, data_format='channels_last', **kwargs):
        super(LinearSpecLayer, self).__init__(**kwargs)
        self.sample_rate = sample_rate
        self.spec_shape = spec_shape
        self.data_format = data_format
        self.frame_step = frame_step
        self.frame_length = frame_length
        self.fmin = fmin
        self.fmax = fmax

    def build(self, input_shape):
        self.mag_scale = self.add_weight(name='magnitude_scaling',
                                         initializer=k.initializers.Constant(value=1.23),
                                         trainable=True)
        super(LinearSpecLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            return tf.TensorShape((None, self.spec_shape[0], self.spec_shape[1], 1))
        else:
            return tf.TensorShape((None, 1, self.spec_shape[0], self.spec_shape[1]))

    def call(self, inputs, training=None):

        # Normalize values between 0 and 1
        inputs = tf.math.subtract(inputs, tf.math.reduce_min(inputs, axis=1, keepdims=True))
        inputs = tf.math.divide(inputs, tf.math.reduce_max(inputs, axis=1, keepdims=True) + 0.000001)
        spec = tf.signal.stft(inputs,
                              self.frame_length,
                              self.frame_step,
                              window_fn=tf.signal.hann_window,
                              pad_end=False,
                              name='stft')

        # Cast from complex to float
        spec = tf.dtypes.cast(spec, tf.float32)

        # Only keep bottom half of spectrum
        spec = spec[:, :, :self.frame_length // 4]

        # Convert to power spectrogram
        spec = tf.math.pow(spec, 2.0)

        # Convert magnitudes using nonlinearity
        spec = tf.math.pow(spec, 1.0 / (1.0 + tf.math.exp(self.mag_scale)))

        # Swap axes to fit input shape
        spec = tf.transpose(spec, [0, 2, 1])

        # Add channel axis
        if self.data_format == 'channels_last':
            spec = tf.expand_dims(spec, -1)
        else:
            spec = tf.expand_dims(spec, 1)

        print(f"final spec shape:{spec}")

        return spec

    def get_config(self):
        config = {'data_format': self.data_format,
                  'sample_rate': self.sample_rate,
                  'spec_shape': self.spec_shape,
                  'frame_step': self.frame_step,
                  'fmin': self.fmin,
                  'fmax': self.fmax,
                  'frame_length': self.frame_length}
        print(config)
        base_config = super(LinearSpecLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def loop_prediction(spectra_dir):
    for filename in glob.iglob(spectra_dir + '/' + '**/*.txt', recursive=True):
        print(filename)


if __name__ == '__main__':
    # Load Keras model
    model = load_model(MODEL_PATH,
                       custom_objects={'LinearSpecLayer': LinearSpecLayer})

    image_only_model = tf.keras.Model(
        inputs=[model.layers[2].input],
        outputs=model.outputs
    )

    # image_only_model.summary()

    # # Create dummy data with input shape (1,144000)
    # dummy_data = tf.random.uniform(shape=(1, 144000))
    #
    # # Run inference
    # p = model.predict(dummy_data)

    # for root, dirs, files in os.walk("/home/mi/Data/BirdNET-Tiny-50/spectra/Apus apus_Common Swift"):
    #     for file in files:
    #         if file.endswith(".pt"):
    #             scientific_name, bird_name = root.split('.')[-1].split('_')
    #             file_path = os.path.join(root, file)
    #             spectrum = tf.convert_to_tensor(torch.load(file_path).numpy())
    #             pp = image_only_model.predict(spectrum)
    #             detection = tf.argmax(pp, 1, name=None)
    #             print(f"detected bird: {detection}")

    for root, dirs, files in os.walk("/home/mi/Data/BirdNET-Tiny-50/audio/Parus major_Great Tit"):
        for file in files:
            if file.endswith(".flac"):

                # load file
                file_path = os.path.join(root, file)
                scientific_name, bird_name = root.split('.')[-1].split('_')
                print(f"bird: {bird_name}, file: {file_path}")
                # data, sample_rate = sf.read(file_path)
                waveform, sample_rate = torchaudio.load(file_path, normalize=False)
                waveform_tf = tf.convert_to_tensor(waveform.numpy())
                pp = model.predict(waveform_tf)
                bird_index = tf.argmax(pp, 1, name=None)
                print(bird_index)

