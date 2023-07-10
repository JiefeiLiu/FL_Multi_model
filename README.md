# Federated Learning with Multi Model 
===
**Framework Architecture**<br>


****
**Experiments set up**<br>
The ```requirement.txt``` contains all the libraries we need to run the program. <br>
You can use ```pip install requirements.txt``` to install all the libraries you need. 

Or you need to install important libraries with specific version: <br>
```Python >= 3.8``` <br>
```pytorch >= 1.12```<br>
```flwr >= 1.0.0```<br>
```scikit_learn >= 1.1.2```<br>

**Set up environment on NMSU CS machine** <br>
Please follow the steps shown in [link](https://github.com/JiefeiLiu/Federated_learning_env_set_up)

****
**Example to Run the Experiments**

1. Run static clustering use the following command:<br>
```python main_multi_FL_static_clustering.py```<br>
2. Run dynamic clustering use the following command:<br>
```python main_multi_FL_dynamic_clustering.py```<br>
3. Run static random use the following command:<br>
```python main_multi_FL_static_random.py```<br>
4. Run dynamic random use the following command:<br>
```python main_multi_FL_dynamic_random.py```<br>


****
**To Set Up the Different Parameters**<br>
Following shows how you can find/modify the crucial parameters in the script. <br><br>
**Client side parameters setting**<br>
client_epochs : client epoch<br>
learning_rate : client side learning rate <br>
batch_size : batch size for local training<br><br>
**Server side parameters setting**<br>
num_clients : require number of clients (should be same as number of partition)<br>
round : server rounds<br>
conf_filter : confidence filtering (use for filtering the prediction in testing)<br>
percentage_of_noise : noise size based on total attack instances<br><br>
**Dataset parameters**<br>
dataset : dataset name <br>
num_classes : number of classes in the dataset<br>
num_features : number of features in the dataset<br><br>
**Other parameter setting**<br>
over_lapping_clients_selection : overlapping clients selection option<br>

