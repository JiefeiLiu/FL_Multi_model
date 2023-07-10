# Multi-Model Federated Learning (MMFL)
**Framework Architecture**<br>

![alt text](https://github.com/JiefeiLiu/FL_Multi_model/blob/main/plots/Multi_model_clustering-Clustering.jpg)

****
**Experiments set up**<br>
The ```requirement.txt``` contains all the libraries we need to run the program. <br>
You can use ```pip install requirements.txt``` to install all the libraries you need. 

Or you need to install important libraries with specific versions: <br>
```Python >= 3.8``` <br>
```pytorch >= 1.12```<br>
```flwr >= 1.0.0```<br>
```scikit_learn >= 1.1.2```<br>

**Set up the environment on the NMSU CS machine** <br>
Please follow the steps shown in [link](https://github.com/JiefeiLiu/Federated_learning_env_set_up)

****
**Dataset**
You can download the CICIDS2017 dataset from [here](https://www.unb.ca/cic/datasets/ids-2017.html) and the CICDDoS2019 dataset from [here](https://www.unb.ca/cic/datasets/ddos-2019.html). Preprocessing (Clean data, feature selection and etc.) before partition. 

**Example to Run the Experiments**

1. Run static clustering using the following command:<br>
```python main_multi_FL_static_clustering.py```<br>
2. Run dynamic clustering using the following command:<br>
```python main_multi_FL_dynamic_clustering.py```<br>
3. Run static random using the following command:<br>
```python main_multi_FL_static_random.py```<br>
4. Run dynamic random using the following command:<br>
```python main_multi_FL_dynamic_random.py```<br>


****
**To Set Up the Different Parameters**<br>
The following shows how you can find/modify the crucial parameters in the script. <br><br>
**Client side parameters setting**<br>
1. client_epochs : client epoch<br>
2. learning_rate : client-side learning rate <br>
3. batch_size : batch size for local training<br><br>

**Server side parameters setting**<br>
1. num_clients : require the number of clients (should be same as the number of the partition)<br>
2. round : server rounds<br>
3. conf_filter : confidence filtering (used for filtering the prediction in testing)<br>
4. percentage_of_noise : noise size based on total attack instances<br><br>

**Dataset parameters**<br>
1. dataset : dataset name <br>
2. num_classes : number of classes in the dataset<br>
3. num_features : number of features in the dataset<br><br>

**Other parameter setting**<br>
1. over_lapping_clients_selection : overlapping clients selection option<br>

