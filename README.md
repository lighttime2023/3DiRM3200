# 3DiRM3200
This repository contains the 3DiRM3200 dataset and codes for the paper "R2Net: 2D Deep Residual Learning with Height Embedding for 3D Radio Map Estimation", which has been accepted by IEEE Transactions on Vehicular Technology. This paper was presented in part in IEEE International Conference on Communications (ICC), Denver, United States, June 2024.

## Dataset
Our 3DiRM3200 dataset consists of 3,200 3D radio maps, 200 building layout images, 200 furniture layout images and 16 transmitter location images for each building. The dataset has be fully available in this repository. Specifically, 3D radio maps are in the folder "3D indoor radio map", building and furniture layout images are in "buildings.rar", and transmitter location images are in "antennas.rar". Due to the size limitation of upload data on the github, 3D radio maps are saved as multiple compressed packages, which can be decompressed directly under the folder "3D indoor radio map". The usage code of the dataset is given in the folder "R2Net/loader_indoor.py", where "dir_dataset" should be set as the path of the dataset.

## R2Net
Our R2Net is  a novel 2D deep residual learning approach to estimate 3D radio maps by taking into account the impact of object heights. As pathloss exhibits different characteristics in indoor and outdoor scenarios, the R2Net is tailored to enhance feature extraction according to pathloss characteristics, including R2Net-In for indoor scenarios and R2Net-Out for outdoor scenarios. In case that only a small training dataset is available, R2Net-Outlite is further proposed based on R2Net-Out to improve the generalization ability of the model.

The codes of R2Net for training and testing are in the folder "R2Net" as the Jupyter Notebooks "R2Net-In.ipynb", "R2Net-Out.ipynb" and "R2Net-Outlite.ipynb" with the corresponding results. The trained model of R2Net-Outlite is in "R2Net-Outlite.pt". Due to the size limitation of upload data on the github, the trained models of R2Net-In and R2Net-Out cannot be uploaded, which are available by contacting me at huiting_rao@tongji.edu.cn.

## Citation

If you use our dataset or codes in your research, please cite our paper:

```bibtex
@ARTICLE{R2Net,
  author={Rao, Huiting and Wang, Junyuan and Zhu, Huiling and Wang, Cheng-Xiang},
  journal={IEEE Transactions on Vehicular Technology}, 
  title={R$^{2}$Net: {2D} Deep Residual Learning with Height Embedding for {3D} Radio Map Estimation}, 
  year={2026},
  volume={},
  number={},
  pages={1-16}
}
