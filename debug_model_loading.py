
import os
import pickle
import sys
import traceback

# Mock streamlit for error logging
class MockSt:
    def error(self, msg): print(f"ST ERROR: {msg}")
    def warning(self, msg): print(f"ST WARNING: {msg}")

st = MockSt()

MODELS_DIR = "models"
SCALERS_FILE = "scalers.pkl"

def load_models():
    """Load trained models and scalers (supports .pkl and .h5)."""
    models = {}
    scalers = None
    
    print(f"Checking models dir: {os.path.abspath(MODELS_DIR)}")
    
    try:
        # Load Scalers
        if os.path.exists(SCALERS_FILE):
            print(f"Loading scalers from {SCALERS_FILE}...")
            with open(SCALERS_FILE, 'rb') as f:
                scalers = pickle.load(f)
            print("Scalers loaded successfully.")
        else:
            print(f"Scalers file not found: {SCALERS_FILE}")
        
        # Helper to load model
        def load_specific_model(type_name, filename_base):
            print(f"\n--- Loading {type_name} ({filename_base}) ---")
            
            # 0. Try .keras (Modern Keras 3) first
            keras_path = os.path.join(MODELS_DIR, f"{filename_base}.keras")
            h5_path = os.path.join(MODELS_DIR, f"{filename_base}.h5")
            
            # Paths to try for deep learning
            dl_paths = [keras_path, h5_path]
            
            for dl_path in dl_paths:
                if os.path.exists(dl_path):
                    print(f"Found Deep Learning model: {dl_path}")
                    try:
                        import tensorflow as tf
                        print(f"TensorFlow version: {tf.__version__}")
                        
                        # Get custom objects
                        custom_objects = {}
                        try:
                            from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention
                            custom_objects = {'LayerNormalization': LayerNormalization, 'MultiHeadAttention': MultiHeadAttention}
                        except Exception as e:
                            print(f"Could not import custom layers: {e}")

                        # Sequence of attempts
                        loading_strategies = [
                            lambda p: tf.keras.models.load_model(filepath=p, custom_objects=custom_objects, compile=False),
                            lambda p: tf.keras.models.load_model(filepath=p, custom_objects=custom_objects),
                            lambda p: tf.keras.models.load_model(filepath=p, compile=False),
                            lambda p: tf.keras.models.load_model(p),
                        ]

                        for i, loader in enumerate(loading_strategies):
                            try:
                                print(f"Attempt {i+1} to load {dl_path}...")
                                model = loader(dl_path)
                                if model: 
                                    print("Success!")
                                    return model
                            except Exception as e:
                                print(f"Attempt {i+1} failed: {e}")
                                # traceback.print_exc()
                                continue
                    except Exception as e:
                        print(f"TensorFlow import or setup failed: {e}")
                        traceback.print_exc()
            
            # 2. Try .pkl (Standard Best Model)
            pkl_path = os.path.join(MODELS_DIR, f"{filename_base}.pkl")
            if os.path.exists(pkl_path):
                print(f"Found Pickle model: {pkl_path}")
                try:
                    with open(pkl_path, 'rb') as f:
                        model = pickle.load(f)
                        print("Success loading pickle!")
                        return model
                except Exception as e:
                    print(f"Failed to load pickle: {e}")
                    traceback.print_exc()

            # 3. Tertiary Fallback: Explicit baseline backup
            baseline_path = os.path.join(MODELS_DIR, f"{filename_base}_baseline.pkl")
            if os.path.exists(baseline_path):
                print(f"Found Baseline backup: {baseline_path}")
                try:
                    with open(baseline_path, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    print(f"Failed to load baseline backup: {e}")
            
            print(f"FAILED to load any model for {type_name}")
            return None

        models['Reg'] = load_specific_model("Regression", "btc_model_reg")
        models['Cls'] = load_specific_model("Classification", "btc_model_cls")

        if not models['Reg'] or not models['Cls']:
             print("WARNING: Could not load one or more models (Reg or Cls is None).")
        else:
             print("All models loaded successfully.")

    except Exception as e:
        print(f"Top level error: {e}")
        traceback.print_exc()
        
    return models, scalers

if __name__ == "__main__":
    load_models()
