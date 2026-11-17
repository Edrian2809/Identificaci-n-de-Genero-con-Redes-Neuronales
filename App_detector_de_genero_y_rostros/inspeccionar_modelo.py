import zipfile
import json
import os

# Ruta a tu archivo original
RUTA_MODELO = r"C:\Users\GMGIN\Downloads\gender_model_mobilenet.keras"

if not os.path.exists(RUTA_MODELO):
    print(f"❌ No se encontró el archivo en: {RUTA_MODELO}")
else:
    try:
        with zipfile.ZipFile(RUTA_MODELO, 'r') as z:
            # Leer el archivo de configuración
            config = json.loads(z.read('config.json'))
            
            print("="*60)
            print("ARQUITECTURA DEL MODELO")
            print("="*60)
            
            # Recorrer la configuración para listar las capas
            def print_layers(layer_list, indent=0):
                for layer in layer_list:
                    class_name = layer.get('class_name', 'Desconocida')
                    config_layer = layer.get('config', {})
                    name = config_layer.get('name', 'Sin nombre')
                    
                    # Imprimir la capa
                    print("  " * indent + f"🔹 {class_name} (nombre: '{name}')")
                    
                    # Si tiene subcapas (ej. un modelo dentro de otro), mostrarlas
                    if 'layers' in config_layer:
                        print_layers(config_layer['layers'], indent + 1)
            
            # Empezar a imprimir desde la raíz del modelo
            print_layers([config])
            
    except Exception as e:
        print(f"❌ Error al leer el archivo: {e}")