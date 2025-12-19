import os
import tensorflow as tf

print(f"TF Version: {tf.__version__}")

try:
    print("\n--- Testing fresh H5 save/load ---")
    m = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(10,))])
    m.save('test_fresh.h5')
    m2 = tf.keras.models.load_model('test_fresh.h5')
    print("SUCCESS: Fresh H5 load works")
except Exception as e:
    print(f"FAILED: Fresh H5 load fails with: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n--- Testing fresh .keras save/load ---")
    m = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(10,))])
    m.save('test_fresh.keras')
    m2 = tf.keras.models.load_model('test_fresh.keras')
    print("SUCCESS: Fresh .keras load works")
except Exception as e:
    print(f"FAILED: Fresh .keras load fails with: {e}")
