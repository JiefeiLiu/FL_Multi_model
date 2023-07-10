**Flower based Federated Learning Experiments**
===

****
**To Run the Experiments**
This experiments only need one local machine. Folling shows the simple running steps:

1. Change the data directory: You need to change the data directory on both ```server.py``` and ```client.py```

1. Run the server: Open one terminal and use the following command:
    ```python server.py```
    
1. Run the clients: Open as many terminals as you need for clients, each terminal represent one client. To run each clients you can use following command:
    ```python client.py```

Now you run a simple Federated Learning Framework. 

****
**To Set Up the Different Parameters**<br>
Following shows how you can find/modify the crucial parameters in the experiments. <br>
main method in ```server.py``` <br>
1. Aggregration function: main method (You can use following aggregration: FedAvg, FedAvgM, QFedAvg, FaultTolerantFedAvg, FedOpt, FedAdagrad, FedAdam, FedYogi)
2. min_fit_clients (int, optional) – Minimum number of clients used during training. Defaults to 2.
3. min_evaluate_clients (int, optional) – Minimum number of clients used during validation. Defaults to 2.
4. min_available_clients (int, optional) – Minimum number of total clients in the system. Defaults to 2.
5. fraction_fit (float, optional) – Fraction of clients used during training. In case min_fit_clients is larger than fraction_fit * available_clients, min_fit_clients will still be sampled. Defaults to 1.0.
6. fraction_evaluate (float, optional) – Fraction of clients used during validation. In case min_evaluate_clients is larger than fraction_evaluate * available_clients, min_evaluate_clients will still be sampled. Defaults to 1.0.
7. server_momentum: *Only for FedAvgM* (float) – Server-side momentum factor used for FedAvgM. Defaults to 0.0.

main method in ```client.py```
1. client_epochs : Clients eopchs for all the clients 
2. learning_rate : Learning rate of all the clients
3. batch_size : Batch size of all the clients 
4. partition_num : Data partition parameter (apply when load partitioned data) *should be same as min_available_clients*
5. num_classes : Total number of classes for the whole FL system. 
