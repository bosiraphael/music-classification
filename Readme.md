# Projet de Deep Learning

## Contexte du projet

Ce projet à été réalisé dans le cadre du cours de Deep Learning de H. Le Borgne dispensé à CentraleSupélec.
C'est un projet d'étude qui a pour but d'appliquer ce que nous avons appris en cours à un problème de notre intérêt.

Ce projet a été réalisé par :

- Bosi Raphaël
- Muller Sylvain

## Utilisation du code

Ce dépôt git est composé de différents dossiers:

- data_processing
- scripts
- train
- trained-models

### requirements.txt

Installer les packages à l'aide de la commande

<code>pip install -r requirements.txt</code>

### data_processing

Ce dossier contient un script python pour créer à partir de la base de donnée GTZAN, les MFCC de chaque morceau de musique.
Pour que cela fonctionne il faut que vous créiez un dossier "data" et que vous téléchargiez les données qui se trouvent ici : https://drive.google.com/drive/folders/1SUyYCu18c5J1GShQBNSN70uSq6KVP4rB?usp=sharing

### scripts

Ce dossier contient des fichiers .sh qui permettent de lancer des jobs sur le cluster de calcul que nous utilisions.

### train

Ce dossier contient les scripts Python qui permettent d'entrainer 3 types de réseau de neurones :

- un LSTM
- un CNN
- Un ResNet en utilisant du transfert learning

Les fichiers suivant définissent l'architectures des réseaux de neurones que nous utilisons :

- CNN.py
- LSTM.py
- pretrained_resnet.py

Les scripts suivants permettent d'entrainer différents réseaux:

- train_CNN.py
- train_LSTM.py
- train_ResNet.py

Les scripts suivants permettent de faire de la recherche d'hyperparamètre :

- CNN_optimisation.py
- LSTM_optimisation.py

Le fichier "data_loader" contient différentes fonction qui mettent en forme les données contenue dans le fichier "data/data_10.json", ce fichier est dans le drive contenant les données, et peut être regénéré grâce au script "data-processing/to_convert_wav_to_data_in_json"

Le fichier cross_validation.py permet de comparer différents réseaux en effectuant une validation croisée à 5 plis.

Le fichier confusion_matrix.py permet de plot les matrices de confusion des différents réseaux.

Le fichier compare_confusion_matrices.py permet de plot les matrice de confusion des différents réseaux sur un même exemple.

### trained_models

Ce dossier contient des réseaux de neurones que nous avons entrainés ainsi que les résultats et les plots.

### comparison

Ce dossier contient les résultats des cross-validations ainsi que des matrices de confusion sur un même exemple.
