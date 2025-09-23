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
4. [Conception d’un nouveau modèle](#4-conception-dun-nouveau-modèle)
5. [Conversion du modèle pour la cible embarquée](#5-conversion-du-modèle-pour-la-cible-embarquée)
6. [Sélection d’un nouveau microcontrôleur](#6-sélection-dun-nouveau-microcontrôleur)
7. [Intégration dans un projet embarqué](#7-intégration-dans-un-projet-embarqué)
8. [Évaluation](#8-évaluation)
9. [Arborescence du dépôt](#arborescence-du-dépôt)
10. [Reproduire les résultats](#reproduire-les-résultats)
11. [Limites & pistes](#limites--pistes)

## 1. Analyse du modèle existant
Le modèle d’origine est un CNN de type VGG adapté à CIFAR‑10 : des blocs de deux convolutions 3×3 suivis d’un max‑pooling, avec un nombre de canaux qui double à chaque étage avant de terminer par deux couches entièrement connectées. Cette organisation est efficace sur GPU, car elle exploite bien le parallélisme. On atteint alors environ 83 % de précision sur l’ensemble de validation.
Cependant, les couches denses, augmentent la taille du modèle. À 3,2 millions de paramètres, le binaire de poids atteint ~15,5 Mo.

Sur microcontrôleur, où la mémoire et la puissance sont limitées, ce modèle est donc trop lourd. Notre travail est de conserver au maximum la précision tout en allégeant fortement le modèle pour qu’il tienne dans la mémoire disponible du microcontrôleur.

## 2. Étude du microcontrôleur cible
