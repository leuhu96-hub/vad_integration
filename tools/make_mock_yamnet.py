"""Builds a stand-in for the retrained YAMNet VAD model so the app runs end to end until the
real file is available. Same contract as the real one:
  input  "waveform": float32 [15360]  (0.96 s, mono, 16 kHz, [-1, 1])
  output float32 [1, 1]               (speech probability of that window)
The mock is just a log-energy detector, NOT a speech model. To switch to the real model,
overwrite app/src/main/assets/models/yamnet.tflite (or import the file from the app).
Usage: python make_mock_yamnet.py out.tflite   (needs tensorflow)
"""
import sys
import tensorflow as tf

WINDOW = 15360


class Mock(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec([WINDOW], tf.float32, name="waveform")])
    def __call__(self, waveform):
        rms = tf.sqrt(tf.reduce_mean(tf.square(waveform)) + 1e-10)
        db = 20.0 * tf.math.log(rms + 1e-6) / tf.math.log(10.0)   # dBFS
        return tf.reshape(tf.sigmoid((db + 45.0) / 4.0), [1, 1])  # 0.5 at -45 dBFS


m = Mock()
conv = tf.lite.TFLiteConverter.from_concrete_functions([m.__call__.get_concrete_function()], m)
open(sys.argv[1], "wb").write(conv.convert())
