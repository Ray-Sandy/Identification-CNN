The Machine Learning Object Detection Using CNN tensorflow model 

In this repo using CNN with 2 diffrent models :
1. ConvMix CNN model ( For Freshness Beef Meat Classification )
2. CNN based model with tensorflow keras ( Identification Fresh Beef meat and Melted Beef meat )
3. CNN Transfer Learning ( Ongoing )

Using 2 diffrent Dataset Beefe :
1. LOCBEEF Dataset Mandaley ( https://data.mendeley.com/datasets/nhs6mjg6yy/1 ) only using Fresh classes for Fresh Beef Meat.
2. Meat Freshness Dataset Keggle ( https://www.kaggle.com/datasets/vinayakshanawad/meat-freshness-image-dataset ) only using Fresh and Half Fresh classes join them to 1 dataset classes Melted.

The Dataset Combination 80% train 10% valid 10% test
The CNN based model have accuracy 99% and val_accuracy 100% ( identifikasi_daging.py | identifikasi_daging_test.ipynb )
The ConvMix CNN model have train accuracy 93% test accuracy 90% ( ConvMix_test.py )
