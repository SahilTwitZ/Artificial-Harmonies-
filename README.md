# 🎵 Artificial Harmonies: Training, Generating, and Converting Melodies

Artificial Harmonies is a project focused on generating melodies using deep learning. It includes training a simple neural network, generating melodies, and saving them as MIDI files. Additionally, the project provides tools for visualizing training metrics and the model architecture.

---

## 🚀 Features
- Generate melodies based on a seed sequence.
- Save generated melodies as MIDI files.
- Visualize training metrics (accuracy, loss) for the model.
- Optional visualization of model architecture.

---

## 📂 Project Structure
- `train_and_plot.py`: 
  - Train a neural network.
  - Save training metrics as JSON.
  - Visualize and save training/validation accuracy and loss plots.
- `melody_generator.py`: 
  - Generates melodies based on a trained model.
  - Saves melodies as MIDI files.
- `requirements.txt`: Lists all dependencies.

---

## 🛠️ Dependencies
Ensure you have the following installed:
- Python 3.8 or later
- TensorFlow
- NumPy
- Matplotlib
- pydot (for optional model architecture visualization)
- graphviz (for optional model architecture visualization)
- music21

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## 🔧 Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/artificial-harmonies.git
   cd artificial-harmonies
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure `graphviz` is installed for model visualization (optional):
   - Install Graphviz: [Download and Install Graphviz](https://graphviz.org/download/)
   - Add Graphviz to your PATH (if not automatically added).

---

## 🖥️ Usage
### 1. Train the Model
Run the training script:
```bash
python train_and_plot.py
```
- Outputs:
  - `training_history.json`: Training and validation metrics.
  - `training_metrics.png`: Visualization of accuracy and loss.
  - `model_architecture.png` (optional): Model architecture diagram.

### 2. Generate Melodies
Use the `melody_generator.py` script to generate melodies:
```bash
python melody_generator.py
```
Modify the parameters in the script:
- `seed`: Initial seed for melody generation.
- `num_steps`: Number of notes to generate.
- `temperature`: Controls randomness in melody generation.

Outputs:
- A generated melody saved as a MIDI file.

### Model Architecture
If `graphviz` and `pydot` are installed, a diagram of the model architecture will be saved as `model_architecture.png`.

---

## 🤝 Contributions
Contributions, issues, and feature requests are welcome! Feel free to fork the repository and submit a pull request.
