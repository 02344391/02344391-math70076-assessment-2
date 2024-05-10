# 02344391-math70076-assessment-2

In this project, we focus on the explainability of machine learning models using Shap values. Specifically, our aim is to compute the Shap values for tree-based methods and ensemble methods that incorporate categorical features. 

Categorical features are commonly encoded using One-hot encoding. Consequently, additional features are introduced into the model, and explainer models assign them their own Shap values. One approach to grasp the contribution of the original categorical feature is to aggregate all the Shap values of the encoded subfeatures. However, calculating Shap values for each subfeature may lack statistical rigor. Even if we assume independence among all initial features of the model, treating encoded subfeatures as independent is illogical.

Shap module: 
https://github.com/shap/shap

Tree shap recursive algorithm from 
title={Consistent Individualized Feature Attribution for Tree Ensembles}, 
author={Scott M. Lundberg and Gabriel G. Erion and Su-In Lee},

bibliography:
  - bibliography.bib