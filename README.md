# NATTACK: A STRONG AND UNIVERSAL GAUSSIAN BLACK-BOX ADVERSARIAL ATTACK


Data and model can be found here: [data and model](https://1drv.ms/f/s!AlXveXe2-CcAhc9mY5XOfDMJjZIiVQ).
 

Please download the data&&model and unzip them to './cifar-data' and './all_models'
 
 
Below is Table 1 from our paper, where we show the robustness of each accepted defense to the adversarial examples we can construct:



| Defense | Dataset | Distance | Success rate |
|---|---|---|---|
| THERM-ADV [Madry et al. (2018)](https://arxiv.org/abs/1706.06083) | CIFAR | 0.031 (linf) | 91.2% |
| LID [Ma et al. (2018)](https://arxiv.org/abs/1801.02613) | CIFAR | 0.031 (linf) | 100.0% |
| THERM [Buckman et al. (2018)](https://openreview.net/forum?id=S18Su--CW) | CIFAR | 0.031 (linf) | 100.0% |
| SAP [Dhillon et al. (2018)](https://arxiv.org/abs/1803.01442) | CIFAR | 0.031 (linf) | 100.0% |
| RSE [Liu et al. (2018)](https://arxiv.org/abs/1712.00673) | CIFAR | 0.031 (linf) | 100.0% |
| CAS-ADV [Na et al. (2018)](https://arxiv.org/abs/1708.02582) | CIFAR | 0.015 (linf) | 97.7% |
| GUIDED DENOISER [(Liao et al., 2018)](https://arxiv.org/abs/1711.00117) | ImageNet | 0.031 (linf) | 95.5% |
| RANDOMIZATION [Xie et al. (2018)](https://arxiv.org/abs/1711.01991) | ImageNet | 0.031 (linf) | 96.5% |
| INPUT-TRANS [Guo et al. (2018)](https://arxiv.org/abs/1711.00117) | ImageNet | 0.005 (l2) | 100.0% |
| PIXEL DEFLECTION [Prakash et al. (2018)](https://arxiv.org/abs/1801.08926) | ImageNet | 0.031 (linf) | 100.0% |




## Paper

**Abstract:**

Recent works find that DNNs are  vulnerable to adversarial examples, whose changes from the benign ones are imperceptible and yet lead DNNs to make wrong predictions. One can find various adversarial examples for the same input to a DNN using different attack methods. In other words, there is a population of adversarial examples, instead of only one, for any input to a DNN. By explicitly modeling this adversarial population with a Gaussian distribution, we propose a new black-box attack called NATTACK. The adversarial attack is hence formalized as an optimization problem, which searches the mean of the Gaussian under the guidance of increasing the target DNN's prediction error. NATTACK achieves 100%  attack success rate  on six out of ten recently published defense methods (and greater than 90% for the other four), all using the same algorithm. Such results are on par with or better than  powerful state-of-the-art white-box attacks. While the white-box attacks are often model-specific or defense-specific, the proposed black-box NATTACK is universally applicable to different defenses. 


## Source code

This repository contains our instantiations of the general attack techniques
described in our paper, 6 defenses (SAP, LID, RANDOMIZATION, INPUT-TRANS, THERM,
and THERM-DAV) are based on BPDA [Anish et al. (2018)](https://arxiv.org/abs/1802.00420), the defended models of GUIDED DENOISER and PIXEL DEFLECTION are based on [Athalye & Carlini, (2018)](https://arxiv.org/abs/1804.03286), and the models defended by RSE
and CAS-ADV come from the original papers.

## Note

Please note that this paper is still on the process of ICML double blind review.

