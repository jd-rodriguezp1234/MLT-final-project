# Generacion del dataset
Los archivos de esta carpeta permiten generar los archivos .pt, .pkl y .csv necesarios para poder entrenar el modelo de texto a malla-3d.

## Dataset original
El dataset original del que se partio para generar el dataset que se usa para entrenar el modelo es [MedShapeNet](https://medshapenet-ikim.streamlit.app/). Este dataset contiene mas de 100 mil mallas biomedicas en formato STL, cada una con su descripcion en texto plano. Debido a su gran tamano, solo se tomo una porcion de 3.589 mallas para entrenar que corresponden a las categorias de organos:

- aorta
- liver
- kidney
- heart

Este muestreo se encuentra en el [archivo csv de organos](organs_selected.csv), donde para cada ejemplo se tienen los siguientes campos:
- **Name:** Nombre del archivo STL.
- **Category:** Descripcion textual de lo que tiene la malla.
- **URL**: Link de descarga del archivo STL.
- **Subcategory:** Subcategoria textual de lo que tiene la malla, aunque dice lo mismo que Category.

## Instalacion de prerrequisitos
Para generar las mallas de entrenamiento se requiere instalar blender y shap-e, el primero siendo un programa de manipulacion de objetos 3D y el segundo el modelo que se va a entrenar en futuros pasos, pero que se usa en este caso para generar la representacion de entrada que requiere el entrenamiento, es decir, los estados latentes.

### Generacion de environment
Se recomienda generar un environment de python dentro de esta carpeta con el comando: `python -m venv datasetvenv` y activarlo con el comando `source datasetenv/bin/activate` para evitar problemas de dependencias.

### Instalacion de requirements
Una vez activado el environement, se instalan prerrequisitos varios con el comando `pip install -r requirements.txt`

### Instalacion de blender
Para instalar blender es necesario descargar el script ejecutable de la pagina oficial, en version 3.3.1, con los siguientes comandos de shell:
 ```
sudo apt-get install xvfb
wget https://download.blender.org/release/Blender3.3/blender-3.3.1-linux-x64.tar.xz
tar -xf blender-3.3.1-linux-x64.tar.xz
 ```
La ruta donde quede la carpeta `blender-3.3.1-linux-x64` es de vital importancia porque los demas scripts dependen de encontrar blender para generar el dataset. Por ejemplo, si los pasos se ejecutaron desde la carpeta de descargas, la ruta de blender sera la siguiente: `~/Downloads/blender-3.3.1-linux-x64/blender`

### Instalacion de Shap-e
Para instalar Shap-e se usa el siguiente comando que instala desde el repositorio de OpenAI: `pip install "git+https://github.com/openai/shap-e"`.

### Posibles errores
Si ocurren errores de instalacion al instalar `Shap-e` o el archivo de `requirements.txt`, se recomienda crear de nuevo el ambiente e instalar primero `Shap-e` antes que `requirements.txt`.

## Creacion de mallas de entrenamiento
Para crear las mallas de entrenamiento se tiene un proceso de ejecucion de dos pasos, en que primero se crea la carpeta de archivos .pt, es decir, las estados latentes, y despues se utilizan estas mallas para generar particiones train-val-test.

### 1. Creacion de estados latentes .pt
Para crear los estados latentes .pt se usa el script de python `encode_model_blender.py`, el cual debe ser ejecutado desde esta carpeta con el comando `python encode_model_blender.py`. Antes de su ejecucion se deben seguir los siguientes pasos:

- Cambiar la ruta de blender en la linea 28 `os.environ["BLENDER_PATH"] = "./blender-3.3.1-linux-x64/blender"` a la encontrada en el paso de `Instalacion de blender`.
- Cambiar el tamano de muestra en la linea 22 `SAMPLE_SIZE = 600` al numero de ejemplos con el cual se quiere entrenar.

Luego de ejecutar este script, se generara una carpeta `dataset_obj` con un archivo `.pt` por cada ejemplo seleccionado. Cada uno de estos archivos se genera en un proceso en que se descarga el archivo STL original, se convierte a formato OBJ y luego se convierte a un tensor de torch con estados latentes en formato `.pt` utilizando el modulo `transmitter` de `Shap-e`.


### 2. Division en train-val-test
Para dividir en train-val-test se usa el notebook `train_test_split.ipynb`, que se puede usar desde jupyter notebook. Para lanzar jupyter notebook se activa el environment y luego se ejecuta el comando `jupyter notebook`. Luego, desde la interfaz grafica de IPython se puede abri este notebook, donde se pueden elegir, en la segunda celda, las siguientes configuraciones:

- **TEST_SIZE:** Porcion que se va a usar para validacion y test.
- **VAL_SIZE:** Porcion que se va a usar de validacion y test para validacion.
- **DATASET_CSV:** Archivo csv que tiene la informacion del dataset original y se recomienda no cambiar.
- **DATASET_FOLDER:** Carpeta que tiene los estados latentes y se recomienda no cambiar.

Una vez estan configuradas estas variables se puede ejecutar el notebook completo, para generar los siguientes archivos:

- **Particiones de train-val-test:**  Son tres archivos .pkl que contienen los ids de los archivos .pt que corresponden a las particiones de train, val y test, siendo estos respectivamente: [`train_organs.pickle`](train_organs.pickle), [`val_organs.pickle`](val_organs.pickle), [`test_organs.pickle`](test_organs.pickle). 
- **Tuplas de id y descripcion:** Es el archivo [`organs_selected_complete.csv`](organs_selected_complete.csv) que contiene todos los pares de id de latente y descripcion del ejemplo que se puedan encontrar en las particiones.