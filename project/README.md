# Entrenamiento y despliegue
Los archivos en esta carpeta permiten entrenar y desplegar un modelo de difusion de tipo Shap-e. Para ello es necesario haber ejecutado todo lo necesario en la carpeta [dataset](../dataset).

## Instalacion de prerrequisitos
El entrenamiento de modelo consiste en hacer finetuning a Shap-e con mallas propias. Es por esto que se requiere instalar Shap-e. Todo el proceso esta basado en el repositorio [Cap3D](https://github.com/crockwell/Cap3D/tree/main/text-to-3D), por lo que la instalacion y ejecucion se basa fuertemente en las instrucciones que se brindan alli.

### Generacion de environment
Se recomienda generar un environment de python dentro de esta carpeta con el comando: `python -m venv datasetvenv` y activarlo con el comando `source datasetenv/bin/activate` para evitar problemas de dependencias.

### Instalacion de requirements
Una vez activado el environement, se instalan prerrequisitos varios con el comando `pip install -r requirements.txt`

### Instalacion de Shap-e
Shap-e se instala en modo desarrollo por si hay que hacer alteraciones del codigo fuente. Para ello, estando en esta carpeta desde una terminal se ejecutan los siguientes comandos:

```
git clone https://github.com/openai/shap-e
pip install -e ./shap-e
```

### Posibles errores
Si ocurren errores de instalacion al instalar `Shap-e` o el archivo de `requirements.txt`, se recomienda crear de nuevo el ambiente e instalar primero `Shap-e` antes que `requirements.txt`.

## Entrenamiento
Para el entrenamiento se tiene el archivo [finetune_shapE.py](train_eval_scripts/finetune_shapE.py), que usa los archivos .pt, .csv y .pickle generados al seguir los pasos descritos en la [carpeta de creacion del dataset](../dataset) para hacer fine tuning del modelo base de Shap-e. Para poder hacer uso de este archivo, hay que copiarlo a la [carpeta raiz de Shap-e](shap-e). Luego se tienen que modificar las siguientes lineas:

- **Lineas 51 y 70**: Cambiar las lineas que dicen `self.captions = pd.read_csv('/home/estudiante/dataset/organs_selected_complete.csv', header=None)` para que tengan la ruta del archivo csv de ids y descripciones de los estados latentes.
- **Lineas 52 y 71**: Cambiar las lineas que dicen `self.valid_uid = list(pickle.load(open('/home/estudiante/dataset/{split}_organs.pickle','rb')))` para que tengan la ruta de los archivos de ids de train y validacion respectivamente.
- **Linea 248**: Cambiar la condicion de guardado del epoch `if epoch in [0, 12, 24]:` para que se guarde un snapshot de modelo en los epochs de interes.

Una vez se hacen estos cambios, se puede ejecutar el entrenamiento desde una terminal como un script de python dentro de la carpeta [shap-e](shap-e). Por ejemplo, se uso el siguiente comando en la maquina de la universidad:

```
python finetune_ShapE.py --gpus 1 --batch_size 8 --save_name med_shape_e --latent_code_path /home/estudiante/dataset/dataset_obj/
```

Para ver mas parametros se puede usar el comando `python finetune_ShapE.py -h`. Por otro lado, esta maquina tenia 24 GB de VRAM, por lo que se pudo usar un batch size de 8, pero al experimentar en maquinas con 12 GB o menos, el batch_size usable es de tan solo 1. Notese tambien, que se agrega como parametro `latent_code_path` la ruta donde estan los estados latentes de los ejemplos, en formato .pt, los cuales se crearon con los scripts de la [carpeta de creacion del dataset](../dataset).

Al ejecutar el script, se generara la carpeta de checkpoints de modelo model_ckpts dentro de la [carpeta raiz de Shap-e](shap-e), donde se tendran los pesos de modelo guardados en formato .pth. Tambien se guardara el csv de estadisticos de entrenamiento sin header en la ruta `shap-e/logs/med_shap_e.csv`, que por cada epoch de interes tendra informacion de:
- Duracion del batch
- Numero del epoch
- Numero del batch dentro del epoch
- Numero del batch total
- Loss de train
- Loss de validacion
- Loss de batch

### Posibles errores
Puede ocurrir que al ejecutar el script de entrenamiento, salga un error de que no existe la carpeta `logs` y la carpeta `model_ckpts`. Por ende, se recomienda crear estas carpetas antes de ejecutar el script.

## Evaluacion cuantitativa
Para evaluar el modelo en test se tiene el archivo [test_metrics.py](train_eval_scripts/test_metrics.py), el cual se tiene que copiar a la carpeta [shap-e](shap-e). Luego de ser copiado, se modifican las siguientes lineas:

- **Linea 23**: Se modifica para tener la ruta de los archivos de latentes .pt.
- **Linea 26**: Se modifica para tener la ruta a **un solo** archivo de pesos de modelo. Se puede dejar vacio para tomar los pesos originales de Shap-e.
- **Linea 35**: Se modifica para que tenga la ruta del archivo csv de ids y descripciones de los estados latentes.
- **Linea 36**: Se modifica para que tengan la ruta de los archivos de ids de test.

Una vez configurado, se puede llamar el script con el comando `python test_metrics.py` dentro de la carpeta [shap-e](shap-e), lo cual generara un archivo json con el error MSE de evaluacion dentro de la carpeta `shap-e/test_metrics`.

## Prediccion cualitativa
Para evaluar de manera cualitativa se tiene el archivo [text2ply_shapE.py](train_eval_scripts/text2ply_shapE.py), que permite generar mallas en formato .ply a partir de la particion de test. Para ello, se copia el archivo a la carpeta [shap-e](shap-e) y semodifican las siguientes lineas:

- **Linea 32**: Se modifica para tener la ruta del snapshot de modelo a evaluar.
- **Linea 41**: Se modifica para que tengan la ruta de los archivos de ids de test.
- **Linea 42**: Se modifica para que tenga la ruta del archivo csv de ids y descripciones de los estados latentes.

Una vez modificados estos valores, se ejecuta el script con el comando `python text2ply_shapE.py` dentro de la carpeta [shap-e](shap-e), y todas las mallas generadas para test se guardaran en la carpeta `shap-e/shapE_inference` con formato .ply

## Despliegue
Para desplegar se tienen los archivos [main.py](deploy_scripts/main.py) y [run.sh](deploy_scripts/run.sh), que son respectivamente el archivo main de despliegue con streamlit y el script de bash para reiniciar la ejecucion si hay problemas de memoria o cache. Para lanzar la aplicacion, se tiene que ir a la carpeta [shap-e](shap-e) desde la terminal y ejecutar los siguientes comandos:

```
sudo chmod +x run.sh
./run.sh 
```

El primero es para obtener permisos de ejecucion y el segundo para ejecutar el loop que reinicia en caso de fallos. Con esto, se puede acceder a la URL `http://localhost:8501` en un navegador donde se puede seleccionar modelo y e ingresar un prompt para generar una malla en formato .ply, la cual se puede descargar.
