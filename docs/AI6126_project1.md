AI6126 Project 1
CelebAMask Face Parsing
Project 1 Specification
Important Dates
Issued: 13 February 2026 05:30 PM SGT
Development Phase: 13 Feb 2026 11:59 PM SGT – 13 Mar 2026 11:59 PM SGT
Test Phase: 14 March 2026 12:00 AM SGT – 20 March 2026 11:59 PM SGT
Group Policy
This is an individual project
Late Submission Policy
Late submissions will be penalized (each day at 5% up to 3 days)
Challenge Description
Face parsing assigns pixel-wise labels for each semantic components, e.g., eyes, nose,
mouth. The goal of this mini challenge is to design and train a face parsing network. We
will use the data from the CelebAMask-HQ Dataset [1] (See Figure 1). For this challenge,
we prepared a mini-dataset, which consists of 1000 training and 100 validation pairs of
images, where both images and annotations have a resolution of 512 x 512.
The performance of the network will be evaluated based on the F-measure between the
predicted masks and the ground truth of the test set (the ground truth of the test set will
not be released).
Figure 1. Sample images in CelebAMask-HQ
Assessment Criteria
We will evaluate and rank the performance of your network model on our given 100 test
images based on the F-measure.
We host the Codabench benchmark for your submission:
https://www.codabench.org/competitions/13609/
In total, the grade of this project is 30% of the final course grade. Specifically, it consists
of four parts:
- Prediction Accuracy (40%)
- Optimization and regularization (15%)
- Experimental analysis (15%)
- Report (30%)
In terms of Prediction Accuracy, we use Codabench above for benchmarking. The higher
the rank of your solution, the higher the score you will receive. In general, scores will be
awarded based on the Table below.
Percentile
in ranking
≤ 5% ≤ 15% ≤ 30% ≤ 50% ≤ 75% ≤ 100% *
Scores 40 36 32 28 24 20 0
Notes:
• The benchmark only affects your score for Prediction Accuracy, which is 40% of
the total score for this project.
• We will award bonus marks (up to 2 marks) if the solution is interesting or novel.
• Marks will be deducted for incomplete submissions, such as missing key code
components, inconsistencies between predictions and code, significantly poorer
results than the baseline, or failure to submit a short report
Submission Guidelines
● Download dataset: this link
● Train and test your network using our provided training set (train.zip).
● [Optional] Evaluate your model on an unseen CodaBench validation set (val.zip)
during development, with up to 5 submissions per day and 60 in total.
● [Required] During test phase, submit your (1) test set predictions, (2) source code,
and (3) pretrained models to CodaBench. The test set will be released one week
before the deadline, following standard vision challenge practices. You are allowed
up to 5 submissions per day, with a total limit of 15.
Restrictions
● To maintain fairness, your model should contain fewer than 1,821,085 trainable
parameters which is 120% of the trainable parameters in SRResNet [2] (your
baseline network). You can use
sum(p.numel() for p in model.parameters())
to compute the number of parameters in your network. Please note it in your report.
● No external data and pretrained models are allowed in this mini challenge. You are
only allowed to train your models from scratch using the 1000 image pairs in our
given training dataset.
● You should not use an ensemble of models.
Step-by-step Submission Procedure
We host the validation and test sets on CodaBench. Please follow the guidelines to
ensure your results to be recorded.
The website of the competition is https://www.codabench.org/competitions/13609/
1. Register the CodaBench account with your NTU email (ends with @e.ntu.edu.sg),
with your matric number as your username. In case you already have and
account with a different username, create an organization with the name as your
matric number, as shown in the figure below.
2. One the registration for Codabench is complete, fill in this microsoft forms for us to
keep track of your name, matriculation number and codabench username – form
link
3. Register for this competition and waits for approval.
4. Submit a file with your prediction results as follows. Include source code and
pretrained models in the test phase; not required for the dev phase.
IMPORANT NOTE Please refer “Get Started → Submission” on the CodaBench
page to for the file structure of your submission. Please adhere to the required
file structure. Submissions that do not follow the structure cannot be properly
evaluated, which may affect your final marks.
If your submission status is “failed”, check the error logs to identify the issue. The
evaluation process may take a few minutes.
5. Submit the following files (all in a single zip file named with your matric number,
e.g., A12345678B.zip) to NTULearn before the deadline:
a. A short report in pdf format of not more than five A4 pages (Arial 10 font)
to describe the model that you use, the loss functions and any processing
or operations that you have used to obtain your results. Report the Fmeasure of your model on the test set, and also the number of parameters
of your model.
Name your report as: [YOUR_NAME]_[MATRIC_NO]_[project_1].pdf
b. A screenshot from the CodaBench leaderboard, with your username and
best score. We will use the score from CodaBench for marking, but will
keep your screenshot here for double-check reference.
c. The results (i.e., the predicted masks) from your model on the 100 test
images. Put them in a subfolder and use the same file name as the input
image. (e.g. If your input image is named as 0001.png, your result should
also be named as 0001.png)
d. All necessary codes you used in this project.
e. The model checkpoint (weights) of your submitted model.
f. A Readme.txt containing the following info:
i. Description of the files you have submitted.
ii. References to the third-party libraries you are using in your solution
(leave blank if you are not using any of them).
iii. Any details you want the person who tests your solution to know
when he/she tests your solution, e.g., which script to run, so that we
can check your results, if necessary.
Each student can only turn in one submission. Resubmission is allowed. But only the
latest one will be counted.
Tips
• You can use <CelebA>(https://github.com/switchablenorms/CelebAMask-HQ) or
all the other online resources for this project. Please specify in your report the
codebase you used.
• The following techniques may help you to boost the performance:
o data augmentation
o deeper model (but be careful of the parameter constraint)
Computational Resource
You can use the computational resources assigned by the MSAI course. Alternatively,
you can use Google CoLab for computation
References
[1] Cheng-Han Lee, Ziwei Liu, Lingyun Wu, Ping Luo, MaskGAN: Towards Diverse and
Interactive Facial Image Manipulation, CVPR 2020
[2] Ledig et al., Photo-Realistic Single Image Super-Resolution Using a Generative
Adversarial Network, CVPR 2017