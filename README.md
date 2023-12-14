# FairTrade: Achieving Pareto-Optimal Trade-offs Between Balanced Accuracy and Fairness in Federated Learning
As Federated Learning (FL) gains prominence in distributed machine learning applications, achieving fairness without compromising predictive performance becomes paramount. The data being gathered from distributed clients in an FL environment often leads to class imbalance. In such scenarios, balanced accuracy rather than accuracy is the true representation of model performance. However, most state-of-the-art fair FL methods report accuracy as the measure of performance,  which can lead to misguided interpretations of the model's effectiveness to mitigate discrimination. To the best of our knowledge, this work presents the first attempt towards achieving Pareto-optimal trade-offs between balanced accuracy and fairness in a federated environment (FairTrade). By utilizing multi-objective optimization, the framework negotiates the intricate balance between model's balanced accuracy and fairness. The framework's agnostic design adeptly accommodates both statistical and causal fairness notions, ensuring its adaptability across diverse FL contexts. We provide empirical evidence of our novel framework's efficacy through extensive experiments on five real-world datasets and comparisons with six competing baselines. The empirical results underscore the significant potential of our framework in improving the trade-off between fairness and balanced accuracy in FL applications.
## The datsets used in this project
* [Adult Census](https://archive.ics.uci.edu/dataset/2/adult)
* [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
* [Default](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
* [Law School](https://github.com/iosifidisvasileios/FABBOO/blob/master/Data/law_dataset.arff)
## Code
### Dataset Processing Scripts

The `datasets` directory contains all the datasets used in this project. Below is a description of python scripts written to process datasets:

- `load_data_utilities.py`: Utility script for loading and preprocessing all the datasets (Adult, Bank, Default, Law).

### Utility Scripts
- `utilities.py`: Utility script for computing evaluation metrics including 'statistical parity', average treatment effect (ATE), balanced accuracy, and accuracy.

### FairTrade main scripts
The following scripts constitute the complete methodology of FairTrade
- `Fairtrade-crypten.py`: Main script for the 'FairTrade' framework that orchestrates the fairness aware federated learning process on different datasets with secure multiparty protocol.
- `Fairtrade.py`: Main script for the 'FairTrade' framework that orchestrates the fairness aware federated learning process on different datasets without secure multiparty protocol.

- `constraint.py`: The script contains the implementation of fairness constraints for discrimination mitigation.
  
## Running the FairTrade-crypten.py Script

To run the `FairTrade-crypten.py` script with the default settings, you can use the following command:

```bash
python FairTrade-crypten.py --fairness_notion 'stat_parity' --num_clients 3 --dataset_name 'bank' --epochs 15 --communication_rounds 50 --mobo_optimization_rounds 10 --distribution_type 'random'
```
## Running the FairTrade.py Script
To run the `FairTrade.py` script with the default settings, you can use the following command:

```bash
python FairTrade.py --fairness_notion 'stat_parity' --num_clients 3 --dataset_name 'bank' --epochs 15 --communication_rounds 50 --mobo_optimization_rounds 10 --distribution_type 'random'
```
## Prerequisites

Before running the script, ensure you have the following Python libraries installed:

- torch==2.0.1
- torchvision==0.15.2
- scikit-learn==0.24.2
- pandas==1.5.3
- gpytorch==1.10
- botorch==0.8.5
- crypten==0.4.1
- cvxopt==1.3.1
- cvxpy==1.3.2

## Citation Request
If you find this work useful in your research, please consider citing:
```bash
@inproceedings{badar2024fairtrade,
  title={FairTrade: Achieving Pareto-Optimal Trade-offs Between Balanced Accuracy and Fairness in Federated Learning},
  author={Badar, Maryam and Sikdar, Sandipan and Nejdl, Wolfgang and Fisichella, Marco},
  booktitle={Proceedings of the 38th Annual AAAI Conference on Artificial Intelligence},
  year={2024}
}
```
