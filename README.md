
# SHAPEffects: 'A feature selection method based on Shapley values robust to concept shift in regression'

This repository has been created in order to allow the reproducibility of the results of the paper 'A feature selection method based on Shapley values robust to concept shift in regression'.

## Abstract

Feature selection is one of the most relevant processes in any methodology
for creating a statistical learning model. Generally, existing algorithms
establish some criterion to select the most influential variables, discarding
those that do not contribute any relevant information to the model. This
methodology makes sense in a classical static situation where the joint
distribution of the data does not vary over time. However, when dealing
with real data, it is common to encounter the problem of the dataset
shift and, specifically, changes in the relationships between variables
(concept shift). In this case, the influence of a variable cannot be the only
indicator of its quality as a regressor of the model, since the relationship
learned in the traning phase may not correspond to the current situation.
Thus, this paper proposes a new feature selection methodology for
regression problems that takes this fact into account, using Shapley
values to study the effect that each variable has on the predictions. Five
examples are analysed: four correspond to standard situations where
the method matches the state of the art in terms of MAE, RMSE
and R2, and one example related to electricity price forecasting (EPF)
where a concept shift phenomenon has occurred in the Iberian market.
In this case the proposed algorithm improves the results significantly.




## Authors

- Carlos Sebastián Martínez-Cava, Fortia Energía, UPM.
- Carlos E. González Guillén, UPM.

