# ACE For Bias Detection
 
## Introduction

Explainable AI is critical in data scientists' toolbelts to help mitigate bias in machine learning algorithms, especially facial recognition technologies. However, many explainable AI algorithms may not necessarily be understandable to the layperson. As noted in the paper “Impact of Explainable AI on Reduction of Algorithm Bias in Facial Recognition Technologies,”: “the success of XAI largely depends on whether users can understand explanations provided.” (B et al., 2024). For instance, the feature importance scores or saliency maps generated by a LIME or SHAP (or another feature-based explanation model) might not be understandable to a non-data science audience. Therefore, bias may be hard for the layperson to identify in features or pixels provided. Consequently, the data science community must explore other ways of making artificial intelligence, such as concepts or abstract, interpretable units (e.g., "striped pattern" in images, "tumor size" in medical data).

In “Towards Automatic Concept-based Explanations,” authors Ghorbani, Wexler, Zou, and  Kim develop a new algorithm, ACE, to extract visual concepts automatically (Ghorbani et al., 2020). The paper provides systematic experiments to demonstrate that ACE discovers concepts that are meaningful to humans, coherent, and essential for the neural network’s predictions. ACE aims to explain high-level “concepts” instead of assigning importance to individual features or pixels.

This repository runs a MobileNetV2 image classifier to classify whether an image is of a male or female individual and then uses the ACE algorithm to create concepts of images to explain why the classifer classified the iamges the way they did. The purpose of this repository is to investigate whether or not ACE can be used to identify bias in a ML algorithm.

## Concept-Based Explanations on Bias Detection

 While TCAV (an algorithm that also uses concepts) needed to have concepts supplied by humans in order for it to run its algorithm (Kim et al., 2018), ACE aims to automatically identify higher-level concepts that are both meaningful to humans as well as important to a machine learning model.  The original TCAV algorithm required the model user to provide concepts, for example, for a zebra, striped, dotted, or zig-zag, and then provide a score back for the importance of that concept in making a classification decision. Those human-supplied concepts themselves may be biased, making an explanation prejudiced. However, ACE generates concepts (instead of them being created and supplied by humans) by first segmenting images, then grouping similar segments as examples of the same concept, and then returning essential concepts, as scored by TCAV scores (Ghorbani et al., 2020), making it a prime tool for identifying bias in the supplied black box algorithm. 

 ## Running ACE on the UTKFaces Dataset

 First this project runs a  MobileNetV2 image classifier to classify whether an image from the UTKFaces Part1 Dataset[LINK] is of a male or female individual, in the file [FILE NAME] with a test accuracy of  0.70.
 
Then, in two separate files, [FILE NAMES], it generates concepts for the female and male classes.

### Checking for Bias: Homogenity in Produced Concepts



