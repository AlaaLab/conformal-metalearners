# Conformal Meta-learners for Predictive Inference of Individual Treatment Effects
#### Ahmed Alaa, Zaid Ahmad, and Mark van der Laan


https://github.com/AlaaLab/conformal-metalearners/assets/21158134/256da5dc-bc1a-4ce9-b1d5-d18bda43c07f


Codebase for the paper "Conformal Meta-learners for Predictive Inference of Individual Treatment Effects"

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
