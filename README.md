
# Code to accompany 'A feature selection method based on Shapley values robust to concept shift in regression'

This repository has been created in order to allow the reproducibility of the results of the paper 'A feature selection method based on Shapley values robust to concept shift in regression'

**Carlos Sebastián, Carlos E. González-Guillén**

## Abstract

Feature selection is one of the most relevant processes in any methodology for creating a statistical learning model. Usually, existing algorithms establish some criterion to select the most influential variables, discarding those that do not contribute to the model with any relevant information. This methodology makes sense in a static situation where the joint distribution of the data does not vary over time. However, when dealing with real data, it is common to encounter the problem of the dataset shift and, specifically, changes in the relationships between variables (concept shift). In this case, the influence of a variable cannot be the only indicator of its quality as a regressor of the model, since the relationship learned in the training phase may not correspond to the current situation. In tackling this problem, our approach establishes a direct relationship between the Shapley values and prediction errors, operating at a more local level to effectively detect the individual biases introduced by each variable. The proposed methodology is evaluated through various examples, including synthetic scenarios mimicking sudden and incremental shift situations, as well as two real-world cases characterized by concept shifts. Additionally, we perform three analyses of standard situations to assess the algorithm's robustness in the absence of shifts. The results demonstrate that our proposed algorithm significantly outperforms state-of-the-art feature selection methods in concept shift scenarios, while matching the performance of existing methodologies in static situations.




## Author

- Carlos Sebastián Martínez-Cava, Fortia Energía, UPM.

