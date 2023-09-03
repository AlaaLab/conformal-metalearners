# Conformal Meta-learners for Predictive Inference of Individual Treatment Effects
#### Ahmed Alaa, Zaid Ahmad, and Mark van der Laan

![comformal_metalearners_anim](https://github.com/AlaaLab/conformal-metalearners/assets/21158134/4ebc6a38-aa6a-4183-9cb6-65c74d7f1ce7)

This is the codebase for the paper "Conformal Meta-learners for Predictive Inference of Individual Treatment Effects". It includes the implementation of a general framework for issuing predictive intervals for individual treatment effects (ITEs) by applying the standard conformal prediction (CP) procedure on top of 

## Installation

## Usage

```
python run_conformal_metalearners.py -t 0.1 -b "DR" "IPW" "X" \
                                     -s "B" -e "Synthetic" -n 1000 \
                                     -d 10  -q True -v True -x 100 \
                                     -c 0.1 -w True
```

```
bash experiments.sh
```

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
