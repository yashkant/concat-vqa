Contrast and Classify: Training Robust VQA Models
===================================================
<h4>
Yash Kant, Abhinav Moudgil, Dhruv Batra, Devi Parikh, Harsh Agrawal
</br>
<span style="font-size: 14pt; color: #555555">
Pre-print, Under Review
</span>
</h4>
<hr>

**Paper:** [https://arxiv.org/abs/2010.06087](https://arxiv.org/abs/2010.06087)

**Project Page:** [yashkant.github.io/projects/concat-vqa](https://yashkant.github.io/projects/concat-vqa.html)

<p align="center">
  <img src="tools/concat-vqa-large.png">
</p>

Recent Visual Question Answering (VQA) models have shown impressive performance on the VQA benchmark but remain sensitive to small linguistic variations in input questions. Existing approaches address this by augmenting the dataset with question paraphrases from visual question generation models or adversarial perturbations. These approaches use the combined data to learn an answer classifier by minimizing the standard cross-entropy loss. To more effectively leverage augmented data, we build on the recent success in contrastive learning. We propose a novel training paradigm (ConClaT) that optimizes both cross-entropy and contrastive losses. The contrastive loss encourages representations to be robust to linguistic variations in questions while the cross-entropy loss preserves the discriminative power of representations for answer prediction.

We find that optimizing both losses -- either alternately or jointly -- is key to effective training. On the VQA-Rephrasings benchmark, which measures the VQA model's answer consistency across human paraphrases of a question, ConClaT improves Consensus Score by 1 .63% over an improved baseline. In addition, on the standard VQA 2.0 benchmark, we improve the VQA accuracy by 0.78% overall. We also show that ConClaT is agnostic to the type of data-augmentation strategy used.

## Repository Setup

Create a fresh conda environment and install all dependencies.

```text
conda create -n concat python=3.6
conda activate concat
cd code
pip install -r requirements.txt
```

Install PyTorch and CUDA tooklt.
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

## Data Setup
Check `README.md` under `data-release` for more details.  

## Training
We provide commands to run both the baseline and ConCAT experiments.

#### Baseline

To train the baseline experiment with only Cross Entropy on train set use:
```
python train.py \
--task_file configs/baseline-train.yml  \
--tag baseline-train \      # output folder
```
To train the baseline experiment with only Cross Entropy on train +val set use:
```
python train.py \
--task_file configs/baseline-trainval.yml  \
--tag baseline-trainval \      # output folder
```

#### ConCAT
To train with ConCAT alternate training on train set use:
```
python train.py \
--task_file configs/concat-train.yml  \
--tag concat-train \      # output folder
```
To train with ConCAT alternate training on train+val set use:
```
python train.py \
--task_file configs/concat-trainval.yml  \
--tag concat-trainval \      # output folder
```

## Result Files
We share result files from above runs which could be submitted to the EvalAI challenge server to obtain the results in the paper:
  
  Method  |  val   |  test-dev   |  test-std  |
 ------- | ------ | ------ | ------ |
Baseline (train)  | 66.31 | - | - |
ConCAT (train)  | **66.98** | - | - |
Baseline (trainval)  | - | 69.51 | 69.22 |
ConCAT (trainval)  | - | **69.80** | **70.00** |

These files are placed under `results/<experiment-name>`. 


## Acknowledgements
Parts of this codebase were borrowed from the following repositories:
- [12-in-1: Multi-Task Vision and Language Representation Learning](https://github.com/facebookresearch/vilbert-multi-task): Training Setup
- [Supervised Contrastive Learning](https://github.com/HobbitLong/SupContrast): Loss Function

We thank  <a href="https://abhishekdas.com/">Abhishek Das</a>,  <a href="https://arjunmajum.github.io/">Arjun Majumdar
</a> and  <a href="https://prithv1.xyz//">Prithvijit Chattopadhyay</a> for their feedback. The Georgia Tech effort was supported in part by NSF, AFRL, DARPA, ONR YIPs, ARO PECASE, Amazon. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of the U.S. Government, or any sponsor.



## License
MIT

