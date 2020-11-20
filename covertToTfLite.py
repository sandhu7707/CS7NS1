import tensorflow as tf

model = tf.keras.models.load_model('var_len_captcha_classifier.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("var_len_captcha_classifier.tflite", "wb").write(tflite_model)
