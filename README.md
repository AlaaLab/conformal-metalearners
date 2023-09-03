# Conformal Meta-learners for Predictive Inference of Individual Treatment Effects
#### Ahmed Alaa, Zaid Ahmad, and Mark van der Laan

![comformal_metalearners_anim](https://github.com/AlaaLab/conformal-metalearners/assets/21158134/4ebc6a38-aa6a-4183-9cb6-65c74d7f1ce7)

This is the codebase for the paper "Conformal Meta-learners for Predictive Inference of Individual Treatment Effects". It includes the implementation of a general framework for issuing predictive intervals for individual treatment effects (ITEs) by applying the standard conformal prediction (CP) procedure on top of pseudo-outcome regression models of the conditional average treatment effects (CATEs). You can use this codebase to apply conformal meta-learners to new datasets or reproduce the experiments in our [paper](https://arxiv.org/abs/2308.14895).

## Installation

Download the codebase from source and install all dependencies in requirements.txt.

## Usage

To execute all experiments in the paper, you can run the following shell script

```
bash experiments.sh
```

To change the parameters of an individual experiment, you can use the following command

```
python run_conformal_metalearners.py -t "test fracion" -b "List of baselines" \
                                     -s "Synthetic data setup" -e "Dataset type" \
                                     -n "Number of synthetic data points" \
                                     -d "Number of feature dimensions"  -q "Use of quantile regression"\
                                     -v "Saving figures" -x "Number of experiments" \
                                     -c "Target coverage" -w "Sweep all coverage probabilities"
```
The list of meta-learners baselines is ["X", "DR", "IPW"]. The parameter -s can be set to "A" or "B" to select the synthetic data setup in Section 5.1. The parameter -e designates the dataset type, with three options: "Synthetic", "IHDP" and "NLSM". The number of data points and feature dimensions are ignored in real data experiments. The parameter -q determines if the meta-learners use quantile regression in the base learners (mean regression is used if -q is set to False). 


## Citation

If you use our code in your research, please cite:
```sh
@article{alaa2023conformal,
  title={Conformal Meta-learners for Predictive Inference of Individual Treatment Effects},
  author={Alaa, Ahmed and Ahmad, Zaid and van der Laan, Mark},
  journal={arXiv preprint arXiv:2308.14895},
  year={2023}
}
```
