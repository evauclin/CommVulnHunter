import os
import pickle
import tensorflow as tf

# Ajoutez vos corrections
tf.config.set_visible_devices([], 'GPU')
os.environ['TF_USE_LEGACY_KERAS'] = '1'

print("🧪 Test avec TensorFlow récent...")
print("TensorFlow version:", tf.__version__)

try:
    # Test 1: Import basique
    from tensorflow.keras.models import load_model

    print("✅ Import load_model OK")

    # Test 2: Charger votre vrai modèle
    model_path = r"C:\Users\farin\OneDrive\Documents\Python Scripts\CommVulnHunter\app\model\best_lstm_model.keras"

    if os.path.exists(model_path):
        print(f"📁 Modèle trouvé: {model_path}")

        # Test de chargement
        try:
            model = load_model(model_path)
            print("✅ Modèle chargé avec succès!")
            print("📊 Inputs:", [inp.shape for inp in model.inputs])
        except Exception as e:
            print(f"❌ Erreur chargement modèle: {e}")
            # Test avec compile=False
            try:
                model = load_model(model_path, compile=False)
                print("✅ Modèle chargé (sans compilation)")
                print("📊 Inputs:", [inp.shape for inp in model.inputs])
            except Exception as e2:
                print(f"❌ Échec total: {e2}")
    else:
        print(f"❌ Modèle non trouvé: {model_path}")

except Exception as e:
    print(f"❌ Erreur import: {e}")