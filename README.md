# Plugging Self-Supervised Monocular Depth into Unsupervised Domain Adaptation for Semantic Segmentation

The usual pipeline is:

train_net1.py 

train_net2.py 

train_transfer.py

generate_augmented_labels.py

train_dbst.py

For each step of the pipleine, please refer to the scripts in the 'launcher' folder.
To launch generate_augmented_labels.py, you fist need to dowanload a pretrained UDA model such as ProDA or LITIR.

