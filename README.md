# Scalable Uncertainty Quantification in Deep Operator Networks using Randomized Priors

Code and data accompanying the manuscript titled "Scalable Uncertainty Quantification in Deep Operator Networks using Randomized Priors", authored by Yibo Yang, Georgios Kissas and Paris Perdikaris.

# Abstract

We present a simple and effective approach for posterior uncertainty quantification in deep operator network (DeepONet); an emerging paradigm for supervised learning in function spaces. We adopt a frequentist approach based on randomized prior ensembles, and put forth an efficient vectorized implementation for fast parallel inference on accelerated hardware. Through a collection of representative benchmarks we show that the merits of the proposed approach are fourfold. (1) It can provide more robust and accurate predictions when compared against deterministic DeepONet. (2) It shows great capability in providing reliable uncertainty estimates on scarce data-sets with diverse magnitudes. (3) It can effectively detect out-of-distribution and adversarial examples. (4) It can seamlessly quantify uncertainty due to model bias, as well as noise corruption in the data. Finally, we provide an optimized JAX library called {\em UQDeepONet} that can accommodate large model architectures, large ensemble sizes, as well as large data-sets with excellent parallel performance on accelerated hardware, thereby enabling uncertainty quantification for DeepONet in realistic large-scale applications.

# Citation

    @article{yang2022scalable,
      title={Scalable Uncertainty Quantification for Deep Operator Networks using Randomized Priors},
      author={Yang, Yibo and Kissas, Georgios and Perdikaris, Paris},
      journal={arXiv preprint arXiv:2203.03048},
      year={2022}
    }

# Data

The data for the Burgers equation example could be found: [https://drive.google.com/file/d/1togEEX40dgLBOyvyW0UsJpb3VehT3Bsa/view?usp=sharing](https://drive.google.com/drive/folders/1si9qnhtvE0B93Rov_Jk68rg5QIEr8N26)

The data for the Reaction - Diffusion and Antiderivative examples can be generated using the python scripts in the respective folders.

The data for the climate modeling could be found:
[https://drive.google.com/file/d/1wJ6Mcj0o80YTv0i0Jf91HZhQuLidCI_C/view?usp=sharing](https://drive.google.com/file/d/1qIhflnpYxC0VVaxVjuHOro74ryy87cku/view?usp=share_link)

