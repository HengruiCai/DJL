##################################################################################################
# This text describes the supplementary material and official implementation of		   #
#										           #
#     Deep Jump Learning for Off-Policy Evaluation in Continuous Treatment Settings	           #
################################################################################################


######################################################################################

I. Requirements

 - Python 3.7
 - `numpy`
 - `pandas`
 - `sklearn`
 - `argparse`
 - `pickle`
 - `os`
 - `random`
 - `time` 
 - `datetime` 
 - `multiprocessing`
 - `warnings`
 - `tqdm`
 - `functools`

######################################################################################

II. Contents

  1. `README.txt`: implementation details of source code and contents

  2. `Supplementary Article.pdf`: technical proofs and additional simulation studies

  3. `Source Code and Simulated Datasets`: Source code of DJL and data generation environment

     a). `DJL.py`: main function for algorithm Deep Jump Learning (DJL).

     b). `Experiments.py`: main code for experiments, including simulation studies and real data analysis.
     
     c). `data_generator.py`: data generation environment, including simulation studies and real data analysis.

     d). `real_envir.pickle`: Note that the real Warfarin Dosing dataset cannot be shared due to privacy protocol. This is a calibrated environment for reproducing the real data.
 

######################################################################################

III. Training (Simulation Studies)

To reproduce the main experimental results in the simulation studies, run this command:

```train
python Experiments.py --envir_type=<CHOICE1> --sample_size=<CHOICE2> --rep_number=<CHOICE3>
```
CHOICE1 = 'simu1', 'simu2', 'simu3', 'simu4', corresponding to four scenarios in Section 5.1, respectively;
CHOICE2 = 50, 100, 200, 300, corresponding to four choices of the sample size;
CHOICE3 = 100, which is the default number of replications.

(Check more configurations in Section 5.1 and `Experiments.py`)
 
Example: 

```For Scenario 1 with sample size as 50 and replication number as 100:
python Experiments.py --envir_type='simu1' --sample_size=50 --rep_number=100
```

######################################################################################

IV. Evaluation (Real Data of Warfarin Dosing)

To evaluate the proposed DJL on the Real Data of Warfarin Dosing with 20 replication, run: 

```eval
python Experiments.py --envir_type='real' --real_data_file='real_envir.pickle' --sample_size=500 --rep_number=20
```

See more details of the calibrated real dataset in Section 5.2 of our main text.  


######################################################################################

V. Source Codes for Kernel-Based Methods

We provide the implementation details for kernel-based methods in Section 5.1 of our main text. Their source codes can be found in the following repositories.


* Kallus & Zhou 2018: https://github.com/CausalML/continuous-policy-learning

* SLOPE by Su, Srinath & Krishnamurthy 2020: https://github.com/VowpalWabbit/slope-experiments

 

