# Oil Spill Detection

## Requisitos

- **Conda:** Asegúrate de tener Miniconda o Anaconda instalados.
- **Git:** Para clonar el repositorio e inicializar los submódulos.

## Configuración

1. **Clonar el Repositorio y los Submódulos:**

   ```bash
   git clone --recurse-submodules <repository_url>
   cd oilspill
   ```

2. **Crear y Activar el Entorno Conda:**

   ```bash
   conda env create -f env/environment.yml
   conda activate oilspill
   ```

3. **Instalar PaddleSeg en Modo Editable:**

    La carpeta `PaddleSeg` se incluye como un submódulo git pero no se instala automáticamente. Para instalarla manualmente, ejecuta:

    ```bash
    pip install -r PaddleSeg/requirements.txt
    pip install -e PaddleSeg
    ```

## Preparación de Datos

1. **Descargar el Conjunto de Datos:**

   La carpeta `data/raw/` contiene el archivo `files.txt` con una lista de URLs para descargar el conjunto de datos comprimidos en 7z. Para descargar todas las partes, puedes ejecutar un script o usar un gestor de descargas que soporte listas de URLs. Por ejemplo, utilizando `wget`:

   ```bash
   cd data/raw
   wget -i files.txt
   ```

2. **Extraer los Archivos 7z:**

   Una vez descargados, extrae los archivos 7z en la misma carpeta `data/raw`. Puedes utilizar una herramienta como [7-Zip](https://www.7-zip.org/) o un equivalente de línea de comandos.

3. **Convertir a la Estructura para el Modelo:**

   Después de la extracción, ejecuta los scripts de preprocesamiento para reorganizar el los datos a la estructura necesaria para el entrenamiento y la evaluación:

   ```bash
   # Reorganizar datos de entrenamiento y validación
   python scripts/preprocess/restructure.py

   # Reorganizar datos de prueba
   python scripts/preprocess/restructure_test.py
   ```

   Los datos procesados se ubicarán en `data/`, organizados en carpetas separadas para entrenamiento y prueba.

## Entrenamiento

1. **Configurar el Entrenamiento:**

   Edita el archivo de configuración `configs/config.yml` según sea necesario para ajustar los parámetros de entrenamiento (rutas de los conjuntos de datos, parámetros del modelo, configuraciones del optimizador, etc.).

2. **Iniciar el Entrenamiento:**

   Ejecuta el script de entrenamiento:

   ```bash
   bash scripts/train.sh
   ```

   Los resultados del entrenamiento (modelos, logs, etc.) se guardarán en el directorio `outputs/`.

**Nota:** Los pesos del modelo entrenado se almacenan en la carpeta `checkpoints/`. Para usar pesos diferentes durante la inferencia, actualiza el parámetro `--model_path` en el script [predict.sh](scripts/predict.sh) según corresponda.

## Inferencia

1. **Ejecutar las Predicciones:**

   Después del entrenamiento, utiliza el script de predicción para generar los resultados de segmentación:

   ```bash
   bash scripts/predict.sh
   ```

## Herramientas Adicionales

- **Herramientas de Visualización:**

  Para visualizar imágenes de entrenamiento y máscaras o predicciones de segmentación, utiliza el script de visualización (agrega el flag --pred para predicciones):

  ```bash
  python scripts/viewer.py [--pred]
  ```
