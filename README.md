# Proyecto final de Machine Learning Techinques

## Integrantes
- Nicolas Laverde Manotas
- Melissa Robles Carmona
- Johan Rodriguez Portela

## Generalidades del proyecto
Este proyecto consiste en desarrollar un flujo de texto a malla 3D especializado en mallas biomedicas. Para ver detalles de la propuesta inicial se recomienda leer la [propuesta de investigacion](proposal.pdf). Para, detalles acerca de resultados y marco teorico completo se recomenda leer el [articulo de investigacion](article.pdf).

## Estructura del repositorio
El repositorio se encuentra dividido en 2 segmentos de ejecucion:

### 1. Generacion del dataset
Contiene el csv, los scripts y notebooks necesarios para generar el dataset de entrenamiento. Esto se puede hallar en la carpeta [dataset](dataset).

### 2. Entrenamiento de modelo
Contiene los scripts y notebooks necesarios para entrenar el modelo de texto a malla, asi como desplegar el modelo para generar mallas no vistas en entrenamiento. Todo esto, se encuentra en la carpeta [project](project).

## Requisitos generales
El proyecto se hizo en una maquina compatible con CUDA 12.2 y pytorch >= 2.2.0. En cuanto a hardware, se requiere una GPU con al menos 6 GB de VRAM.