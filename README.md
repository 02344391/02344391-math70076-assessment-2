# 02344391-math70076-assessment-2

In this project, we focus on the explainability of machine learning models using SHAP values. Specifically, our aim is to compute the SHAP values for tree-based methods and ensemble methods that incorporate categorical features. 

Categorical features are commonly encoded using One-hot encoding. Consequently, additional features are introduced into the model, and explainer models assign them their own SHAP values. One approach to grasp the contribution of the original categorical feature is to aggregate all the SHAP values of the encoded subfeatures. However, calculating SHAP values for each subfeature may lack statistical rigor. Even if we assume independence among all initial features of the model, treating encoded subfeatures as independent is illogical.

License:

The MIT License (MIT)

Copyright (c) 2018 Scott Lundberg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.