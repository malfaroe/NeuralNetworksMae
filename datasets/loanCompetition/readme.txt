yes and it's impossible to get 100% accuracy in the real-world even 80% is impossible but as we know this is a competition. so Here are some tips 

1. Use full train data to train the model 

2. Use XGBoost or lightgbm algorithm with some parameter tuning

3. Don't do binning or any heavy feature engineering

4. Use AUC metrics to evaluate your model

I think that's it, if there is anything you can add let me know, please.

Best score using datset original without polynomials, and no correlation elimination. Only OHE. Score: 78.4722222 with LR and same with ensemble XT + LR
Best score so far was achieved using the algorithms with the best validation scores in tuning vieja