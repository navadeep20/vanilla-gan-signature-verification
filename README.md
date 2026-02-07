Vanilla GAN for Offline Signature Verification

This project implements a Vanilla Generative Adversarial Network (GAN) to generate synthetic handwritten signatures and use them to improve the robustness of an offline signature verification system.

Many real-world systems such as banks, universities, and e-governance platforms rely on handwritten signatures for authentication. However, these systems often suffer from a lack of sufficient genuine signature samples, making it difficult to train reliable verification models.

This project addresses that problem by using a Vanilla GAN to learn the distribution of genuine signatures and generate synthetic variants. These synthetic samples are then used to augment the training dataset and improve verification performance.

Key Objectives

Learn the distribution of real handwritten signatures using a Vanilla GAN.

Generate realistic synthetic signature images.

Augment training data for a signature verification model.

Evaluate the impact of synthetic data on:

Accuracy

False Acceptance Rate (FAR)

False Rejection Rate (FRR)

Equal Error Rate (EER)

Provide a simple UI for generating synthetic signature datasets.

Real-World Applications

Bank cheque and mandate verification

KYC form validation

University exam and certificate verification

E-governance document authentication

Research in offline signature verification systems

Project Workflow

Data Preprocessing

Convert signatures to grayscale

Crop and resize to 64×64

Normalize pixel values

Vanilla GAN Training

Generator produces synthetic signatures

Discriminator distinguishes real vs fake

Adversarial training improves realism

Synthetic Data Augmentation

Generated signatures added to training set

Increased variation in genuine samples

Verification Model Training

Baseline model: real signatures only

Augmented model: real + synthetic signatures

Performance Evaluation

Accuracy improved from 60% to 70%

FAR reduced from 40% to 20%

Demonstrates reduction in forgery acceptance

Project Structure
vanilla_gan_signatures/
│
├── src/
│   ├── generator_vanilla_gan.py
│   ├── discriminator_vanilla_gan.py
│   ├── train_vanilla_gan_signatures.py
│   ├── generate_signatures.py
│   ├── signature_verifier_train.py
│   ├── signature_verifier_eval.py
│   └── app_vanilla_gan_signatures.py
│
├── data/
├── checkpoints/
├── samples/
├── docs/
└── requirements.txt

Technologies Used

Python

PyTorch

Torchvision

Streamlit (for UI)

NumPy

Matplotlib

Key Results
Metric	Baseline (Real Only)	Augmented (Real + GAN)
Accuracy	0.60	0.70
FAR	0.40	0.20
FRR	0.40	0.40
EER	~0.40	~0.30

Result:
Synthetic signatures reduced forgery acceptance by 50%.

Future Improvements

Conditional GAN for person-specific signatures

Higher resolution (128×128 or 256×256)

DCGAN or StyleGAN architectures

Siamese network–based verification

Integration with real-world banking datasets
