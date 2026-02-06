# new-ru
VAE-Based Anomaly Detection

This project implements a Variational Autoencoder (VAE) for detecting anomalies in multivariate data, such as sensor readings from industrial systems. The VAE learns the distribution of normal data, and deviations from it are flagged as anomalies.

Table of Contents

Overview

Features

Requirements

Installation

Usage

Dataset

Methodology

Results

License

Overview

Detecting anomalies in sensor or manufacturing data is crucial to prevent defects or failures. This VAE-based approach is unsupervised, meaning it only needs normal data for training.

Features

Unsupervised anomaly detection using VAE

Reconstruction-based anomaly scoring

Customizable hidden layers and latent dimensions

Easy to adapt to any numerical dataset

Visualization of training loss and reconstruction errors

Requirements

Python >= 3.8

PyTorch

NumPy

Matplotlib

scikit-learn

Install via:

pip install torch numpy matplotlib scikit-learn

Installation

Clone the repository:

git clone https://github.com/yourusername/vae-anomaly-detection.git
cd vae-anomaly-detection


(Optional) Use a virtual environment:

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

Usage

Prepare your dataset: A 2D NumPy array of shape (n_samples, n_features).

Train the VAE:

model = VAE(input_dim=20)
losses = train_vae(model, train_loader, epochs=50)


Save training loss plot:

plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("loss_plot.png")


Detect anomalies:

test_errors = get_reconstruction_error(model, test_dataset.data)
threshold = np.percentile(test_errors[:len(normal_data)], 95)
predictions = (test_errors > threshold).astype(int)


Visualize reconstruction errors:

plt.hist(test_errors[:1000], bins=50, alpha=0.7, label="Normal")
plt.hist(test_errors[1000:], bins=50, alpha=0.7, label="Anomalous")
plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
plt.legend()
plt.title("Reconstruction Errors")
plt.savefig("error_histogram.png")

Dataset

normal_data: Represents normal operations.

anomalous_data: Synthetic or real anomalies for testing.

Data is normalized using StandardScaler for consistent reconstruction.

⚠️ Train the VAE only on normal data to capture the normal distribution.

Methodology

VAE Architecture:

Encoder compresses data to latent variables (mu, logvar)

Reparameterization samples latent vector

Decoder reconstructs original input

Loss Function:

Reconstruction Loss (MSE): Measures input reconstruction error

KL Divergence: Regularizes latent space to approximate Gaussian

Anomaly Detection:

Compute reconstruction error for each sample

Define a threshold (e.g., 95th percentile of normal errors)

Flag samples exceeding threshold as anomalies

Results

Threshold: Determined from normal data

Example metrics:

Threshold: 4.5123
Anomalies detected: 102 out of 1100
Accuracy: 0.9636


Training Loss Plot:

Reconstruction Error Histogram:

The histogram clearly separates normal and anomalous samples, demonstrating effective detection.
