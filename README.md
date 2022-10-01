# MILP - Few-shot Inductive Link Prediction

This is the code necessary to run experiments on MILP algorithm described in the paper .

## Requiremetns

dgl==0.4.2
lmdb==0.98
networkx==2.4
scikit-learn==0.22.1
torch==1.4.0
tqdm==4.43.0

## Inductive relation prediction experiments

All train-graph and ind-test-graph pairs of graphs can be found in the `data` folder. We use WN18RR_v1 as a runninng example for illustrating the steps.

### MILP
To start training a MILP model, run the following command. 
`python train.py -d WN18RR_v1 -e WN18RR_v1`

To test MILP run the following commands.
- `python test_auc.py -d WN18RR_v1_ind -e WN18RR_v1`
- `python test_ranking.py -d WN18RR_v1_ind -e WN18RR_v1`
Change the file test_auc.py while using fine-tuning mechanism
The trained model and the logs are stored in `experiments` folder. 

## Citation

If you found the provided code with our paper useful in your work, we kindly request that you cite our work. </br>

```BibTex
@article{yang2022few,
  title={A Few-shot Inductive Link Prediction Model in Knowledge Graphs},
  author={Yang, Ruiting and Wei, Zhongcheng and Fan, Yongjian and Zhao, Jijun},
  journal={IEEE Access},
  year={2022},
  publisher={IEEE}
}
```
