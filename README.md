# 📘 Beyond Scores: Explainable Image Quality Assessment via Restoration Parameter Estimation

### 🔎 Project Overview

Traditional Image Quality Assessment (IQA) models predict a subjective quality score (e.g., 4/10). While useful for benchmarking, such scores are not actionable and do not indicate how to fix a degraded image.

This project reformulates blind IQA as a restoration parameter regression task.

Instead of predicting a quality score, the model estimates the exact degradation parameters applied to an image:
	•	Blur radius
	•	Noise level
	•	JPEG compression quality
	•	Gamma exposure shift

These predicted parameters can be directly used to guide image restoration.

⸻

### 🧠 Key Idea

Input: Degraded image
Output: Parameter vector

Example output:
Blur Sigma: 1.82
Noise Level: 7.22
JPEG Quality: 42.33
Gamma Shift: 0.92

This makes the system:
	•	Interpretable
	•	Explainable
	•	Restoration-guiding

  
⸻

### 🧪 Model Details

	•	Backbone: CNN-based regression network (ResNet-based architecture)
	•	Loss Function: Mean Squared Error (MSE)
	•	Output: 4-dimensional parameter vector
	•	Evaluation Metric: Mean Absolute Error (MAE)

⸻

### 🎯 Research Contribution

This project proposes an alternative formulation of blind IQA:
	•	Replaces subjective score prediction with parameter regression
	•	Enables explainable degradation analysis
	•	Provides actionable outputs for restoration pipelines
	•	Eliminates dependence on human Mean Opinion Scores (MOS)
