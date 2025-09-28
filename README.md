# Déploiement d’un modèle CIFAR‑10 sur microcontrôleur

> Paul Guillemin - Marc Scholkopf – Déploiement d’un DNN pré‑entraîné CIFAR‑10 sur microcontrôleur (STM32)

---

## Résumé
- **Modèle initial** : CNN (≈3,2 M paramètres, 83% accuracy) → trop lourd pour l’embarqué.
- **Cible finale** : **STM32L4R9 (Cortex‑M4F @ 120 MHz, 640 KB SRAM, 2 Mo Flash)**.
- **Nouveau modèle** : **DS‑CNN CIFAR‑Lite** (depthwise separable) ≈ **160 k** paramètres.
- **Quantification** : **INT8** + **CMSIS‑NN** via **TensorFlow Lite Micro** / **STM32Cube.AI**.
- **Résultats sur cible** : 86,7 % top‑1 | **~12,6 ms** / image | **~160 KB** de poids | **~100–120 KB** de RAM de travail | **~0,8–0,9 mJ** / inférence.

- ## Table des matières
1. [Analyse du modèle existant](#1-analyse-du-modèle-existant)
2. [Étude du microcontrôleur cible](#2-étude-du-microcontrôleur-cible)
3. [Évaluation de l’embarquabilité du modèle initial](#3-évaluation-de-lembarquabilité-du-modèle-initial)
4. [Conception et implémentation de nouveaux modèles plus optimisés](#4-conception-et-implementation-de-nouveaux-modelès-plus-optimisés)
5. [Sélection d’un nouveau microcontrôleur](#5-sélection-dun-nouveau-microcontrôleur)
6. [Arborescence du dépôt](#arborescence-du-dépôt)
7. [Reproduire les résultats](#reproduire-les-résultats)
8. [Limites & pistes](#limites--pistes)

## 1. Analyse du modèle existant

### 1.A. Présentation de la banque d'images CIFAR-10

La première question à se poser quand on souhaite concevoir un modèle d'Intelligence Artificielle (IA) est : Quelles sont les données que je vais utiliser pour entrainer et tester le modèle. Dans le cadre de notre projet, nous allons utiliser la banque d'images nommée CIFAR-10. CIFAR-10 correspond au nom attribué à une banque de données d'entrainements qui contient au total 60 000 images de taille (32 x 32)px en couleur. Ainsi, généralement, 50 000 images vont être dédiées à l'entraînement du modèle et 10 000 images seront dédiées à l'évaluation du modèle. 

La banque d'images CIFAR-10 contient, comme son nom nous l'indique, 10 classes d'images :  

0. Avion
1. Voiture
2. Oiseau
3. Chat 
4. Cerf
5. Chien
6. Grenouille
7. Cheval
8. Bateau
9. Camion

Le modèle d'IA que nous allons analyser aura donc pour objectif de classer les différentes images de la banque CIFAR-10 suffisamment précisément parmi les 10 classes énoncées juste ci-dessus.  

### 1.B. Présentation du modèle d'IA

Le modèle d'intelligence artificielle que nous utilisons à l’origine pour classer les images de CIFAR-10 est un CNN (Convolutional Neural Network) de type VGG (Visual Geometry Group) adapté à CIFAR‑10. Il utilise donc des couches de convolution lui permettant de détecter les caractéristique inhérentes à chaque image pour pouvoir les classer. Il possède alors des blocs de deux convolutions 3×3 suivis d’un max‑pooling, avec un nombre de canaux qui double à chaque étage avant de terminer par deux couches entièrement connectées.
Cette organisation est efficace sur GPU, car elle exploite bien le parallélisme. On atteint alors environ 83 % de précision sur l’ensemble de validation.
Cependant, les couches denses, augmentent la taille du modèle. À 3,2 millions de paramètres, le binaire de poids atteint ~15,5 Mo.

Voici les caractéristiques du modèle d'IA initial :

- **Taille :** 5.12 Mo
- **Nombre de paramètres :** 1,344,042 paramètres
- **Précision :** 83.7%
- **Image à introduire**

### 1.C. Analyse structurelle du modèle d'IA

Le modèle d'IA est un CNN, modèle particulièrement adapté à la reconnaissance d'images. En effet, il prend en entrée, une seule image. Chaque image en entrée du modèle d'IA correspond à un tableau de 3 matrices pour les couleurs RGB et chacune des matrices est de taille 32x32 du nombre de pixels présent dans chaque image. Dans un premier temps, l'objectif, pour le modèle d'IA, est de transformer l'image d'entrée pour pouvoir faire ressortir ses caractéristiques (convolution). Celles-ci permettront alors, dans un second temps, au modèle d'IA, de comparer l'ensemble des caractéristiques de l'image aux caractéristiques des images des différentes classes. Le modèle produira établira alors les probabilités de ressemblances pour chacune des classes ce qui nous amènera à conclure de la classification faite par le modèle.
organisé de la manière suivante :

[Entrée : Image-(32x32x3)]  
↓  
[6 Couches convolutives]   
↓  
[1 Couche de vectorisation 1D]    
↓  
[3 Couches Fully Connected]    
↓  
[Sortie :Prédiction]   

Cette organisation va lui permettre de classifier les images en entrée. 

#### 1.C.1. Les couches convolutives

Les couches convolutives du modèle vont permettre de pouvoir faire ressortir les caractéristiques de l'image en entrée. En effet, chaque couche de convolution de notre modèle est composée des éléments suivants successivement :

1. n masques de convolution
2. une fonction d'activation non-linéaire (ReLU, dans notre cas)
3. Phase de dropout (pas sur toutes les couches)
4. Batch-normalisation
5. Phase de pooling (pas sur toutes les couches)

**Masque de convolution :** Chaque couche de convolution possède un certain nombre de masques de convolution différents. Ces masques correspondent à des matrices de taille généralement 3x3 avec une profondeur de 3 matrices (pour les couleurs RGB) qui vont naviguer sur l'ensemble des pixels de l'image d'entrée. Ainsi, des produits scalaires vont être réalisés entre les valeur du masque convolutionnel et les valeurs des pixels de l'image d'entrée. De cette manière, une nouvelle image est créée dont les valeurs de chaque pixel équivaut au produit scalaires du pixel correspondant sur l'image d'entrée par la valeur de la matrice du masque convolutionnel. Ainsi, on fabrique autant de nouvelles images qu'il y a de masques convolutionnels différents sur la couche. De cette manière, les caractéristiques de l'image vont commencer à ressortir sur les nouvelles images ainsi créées. 

**Fonction d'activation :** La fonction d'activation permet d'introduire de la non-linéarité dans les calculs réalisés par les neurones du réseau. Il existe plusieurs fonction d'activation, toutes non-linéaires, mais, dans notre cas, nous utilisons la fonction ReLU (Rectified Linear Unit. Cette fonction produit une sortie de 0 pour tout x∈]-∞;0] et x pour tout x∈[0;+∞[. Elle est simple à calculer donc tout à fait aux réseaux de neurones profonds. Elle permet une meilleure performance de calcul du réseau. C'est pourquoi elle est utilisée dans ce modèle.

**Dropout :** La notion de Dropout n'intervient pas à chaque couche convolutionnelle. Le dropout prend en argument une probabilité. En effet, pendant l'exécution du modèle, aléatoirement, la phase de Dropout va venir mettre à 0 plusieurs neurones avec une probabilité égale à la probabilité marquée en entrée. Cette phase permet notamment d'équilibrer l'entrainement du modèle et empêcher la prépondérance de certains paramètres devant d'autres. Cette phase va donc permettre "d'éteindre" certains neurones selon une certaine probabilité afin de ne pas les prendre en compte dans la suite des calculs pour diminuer leur importance. Cela permet donc d'équilibrer l'importance de tous les paramètres et donc d'optimiser la précision du modèle d'IA.

**Batch-normalisation :** Cette phase correspond à la normalisation des sorties selon des mini-batch de la couche convolutionnelle entre 0 et 1 pendant l'entraienement ce qui va permettre de stabiliser l'apprentissage du modèle et de poursuivre le travail de régularisation et d'équilibrage de l'ensemble des paramètres du modèle.

**Pooling :** Il s'agit de la phase finale de quelques couches convolutionnelles du modèle. Cette phase agit en tant que compression des données de sortie de la couche. En effet, en sortie de la couche, nous avons un certain nombre d'images provenant directement de l'image d'entrée égale au nombre de masques convolutionnels présents sur la couche. Afin de diminuer la taille des données du modèle, cette phase va compresser les différentes images de sorties en divisant notamment leur taille par 2 par le biais de l'utilisation d'une fonction maximum (Max-pooling). Pendant cette phase, on va reconstruire chaque image en prenant la valeur maximale des pixels présente dans un carré de (2x2)px. On compresse alors chacune des images de sortie. 

Le modèle d'IA possède au total 6 couches convolutives dont le nombre de masques convolutifs est multiplié par 2 toutes les 2 couches. Voici un schéma des couches convolutionnelles du modèle :

AJOUT D'UN SCHEMA

#### 1.C.2. Couche de vectorisation 1D

Maintenant que le processus a parcouru l'ensemble des couches convolutives, le modèle dispose de 128 images de taille (2x2)px. Chacune de ces images traite d'une caractéristique particulière de l'image d'entrée. Les 128 images permettent de décrire l'image d'entrée du modèle. Cependant, pour classifier l'image d'entrée du modèle et donc, traiter l'ensemble de ces 128 matrices 2x2, il est nécessaire d'utiliser une réseau de neurone de type "Fully Connected". Ce type de réseau s'organise selon des couches de neurones qui sont tous reliés entre eux : chaque neurone de la couche précédente est relié à tous les neurones de la couche suivante. Ce type de réseau est particulièrement adapté pour la classification de données. C'est pourquoi il constitue la 2ème partie du modèle que l'on analyse. Cependant, ce type de réseau admet, en entrée, des vecteurs de données. Or, actuellement, le modèle dispose de 128 tenseurs 2D de taille 2x2. Une couche de vectorisation 1D est alors introduite entre la partie convolutionnelle et la partie "Fully Connected" du modèle. Cette couche permet de fabriquer un vecteur 1D à partir de l'ensemble des 128 tenseurs 2D 2x2 du modèle. Ce vecteur 1D va alors être positionné en entrée de la partie Fully Connected.

AJOUT D'UN SCHEMA

#### 1.C.3. Couches Fully Connected

Dans cette partie du modèle, les neurones vont réaliser des combinaisons linaires des données du vecteur d'entrée avec l'ajout de poids et y ajouter la fonction d'activation qui va permettre d'introduire une non-linéarité ce qui est essentiel au neurone pour son bon fonctionnement. 

Dans notre modèle, il y a 3 couches Fully Connected qui permettent de classifier l'image d'entrée du modèle. La première couche dispose de 1024 neurones ayant comme fonction d'activation la fonction "ReLU" que l'on a détaillé précédemment. Cette couche dispose également d'une phase de Dropout de probabilité 0.5 qui agit exactement de la même manière et de le même but que pour les couches convolutives. Une seconde couche est présente avec 512 neurones. Celle-ci est identique structurellement parlant à la première couche. Enfin, le modèle possède une 3ème et dernière couche qui possède autant de neurones que de classe de sortie de la banque CIFAR-10. En effet, cette dernière couche va permettre de produire un vecteur de probabilités associées à chacune des classes pour lesquels l'image d'entrée du modèle pourrait appartenir. La fonction d'activation utilisée pour réaliser cela est la fonction "softmax" qui est tout à fait adaptée à cela. L'image est classée dans la classe pour laquelle le modèle d'IA a fournit la probabilité la plus élevée d'appartenance.

On en conclut alors que notre modèle d'IA est bien adapté pour la classification d'image et notamment celles de la banque CIFAR-10.

## 2. Étude du microcontrôleur cible

Maintenant que nous avons pu analyser le modèle d'IA de classification des images de la banque CIFAR-10, notre objectif va être de déterminer son embarquabilité dans la carte choisie au départ.

Tout d'abord, nous avons choisis d'utiliser la carte STM32L4R9 Discovery Kit qui possède les caractéristiques techniques suivantes :

- **MCU :** STM32L4R9AII6
- **Performances :** 120 MHz/150 DMIPS
- **Taille Flash :** 2 Mo
- **Taille SRAM :** 640 ko

Afin de pouvoir embarqué le modèle d'IA sur cette carte, il nous faut prendre en compte des considérations techniques importantes qui sont la taille de la mémoire Flash et de la SRAM de la carte par rapport à la taille du modèle à embarquer et également la performance du MCU pour qu'il soit suffisamment performant pour exécuter le modèle d'IA. En effet, dans le processus d'implémentation du modèle d'IA dans la carte, l'ensemble des poids et la structure de l'ensemble du modèle vont être stockés dans la mémoire Flash de la carte tandis que les entrées/sorties du modèle, les activations intermédiaires et les variables temporaires vont être stockés dans la SRAM. Cependant, il faut également prendre en compte le fait que le modèle d'IA ne doit prendre toute la place de la mémoire de la carte. En effet, il faut pouvoir garder de la place en mémoire pour laisser l'utilisateur construire les différents programmes nécessaires à l'utilisation du modèle d'IA ou autres.

Il va donc falloir analyser l'embarquabilité du modèle sur la carte selon les différents critères que l'on a donnés.

## 3. Évaluation de l’embarquabilité du modèle initial

Ainsi, sur microcontrôleur, où la mémoire et la puissance sont limitées, ce modèle est donc trop lourd. Notre travail est de conserver au maximum la précision tout en allégeant fortement le modèle pour qu’il tienne dans la mémoire disponible du microcontrôleur.

## 4. Conception et implémentation de nouveaux modèles plus optimisés

### 4.A. Conception et implémentation d'un 1er modèle 

#### 4.A.1. Conception du modèle

#### 4.A.2 Implémentation du modèle sur le MCU cible

### 4.B Conception et implémentation d'un 2ème modèle

#### 4.B.1. Conception du modèle

#### 4.B.2 Implémentation du modèle sur le MCU cible

## 5. Sélection d'un nouveau microcontrôleur


