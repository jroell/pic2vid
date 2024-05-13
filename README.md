## 3D Facial Animation from 2D Images and Audio

This project demonstrates how to generate 3D facial animations from 2D images and audio input using Python, dlib, OpenCV, and the eos 3D morphable model library. This guide will help you set up and run the project in your Windows Subsystem for Linux (WSL) environment on Windows 11.

### Prerequisites

Before you begin, ensure you have the following installed:

- **Windows Subsystem for Linux (WSL)**: Ensure WSL is installed on your Windows 11 system. [Follow the official guide to install WSL](https://docs.microsoft.com/en-us/windows/wsl/install).
  
- **Python 3.8+**: This project requires Python 3.8 or newer. You can check your Python version by running `python3 --version` in your terminal.

- **Git**: Needed to clone the repository. Install Git using `sudo apt update` and `sudo apt install git`.

- **Pip**: Python package installer, should come with Python. If not, install it using `sudo apt install python3-pip`.

### Installation

1. **Open your WSL terminal** and update your package list:
   ```bash
   sudo apt update
   ```

2. **Install essential libraries** needed for the project:
   ```bash
   sudo apt install build-essential cmake pkg-config
   sudo apt install libx11-dev libatlas-base-dev
   sudo apt install libgtk-3-dev libboost-python-dev
   ```

3. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/your-repository-name.git
   cd your-repository-name
   ```

4. **Set up a Python virtual environment** (recommended to manage dependencies):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

5. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` is not provided, install the necessary libraries manually:
   ```bash
   pip install numpy opencv-python dlib librosa eos-py
   ```

   For the `eos` library, you may need to follow the installation instructions provided in the [eos documentation](https://github.com/patrikhuber/eos). It usually involves cloning the repository and building from source if the python package does not work.

6. **Install dlib with dependencies** if not already successful:
   ```bash
   pip install dlib
   ```

### Setting Up the Models

Download the required model files and place them in your project directory:

- **shape_predictor_68_face_landmarks.dat**: Download from [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
- **sfm_shape_3448.bin**, **expression_blendshapes_3448.bin**, **ibug_to_sfm.txt**, **sfm_3448_edge_topology.json**, and **sfm_model_contours.json**: These can be found in the public [eos share](https://github.com/patrikhuber/eos/wiki/Model-share).
- You can try downloading them all using wget:
```bash
# Download sfm_shape_3448.bin
wget -O sfm_shape_3448.bin "https://github.com/patrikhuber/eos/raw/master/share/sfm_shape_3448.bin"

# Download expression_blendshapes_3448.bin
wget -O expression_blendshapes_3448.bin "https://github.com/patrikhuber/eos/raw/master/share/expression_blendshapes_3448.bin"

# Download ibug_to_sfm.txt
wget -O ibug_to_sfm.txt "https://github.com/patrikhuber/eos/raw/master/share/ibug_to_sfm.txt"

# Download sfm_3448_edge_topology.json
wget -O sfm_3448_edge_topology.json "https://github.com/patrikhuber/eos/raw/master/share/sfm_3448_edge_topology.json"

# Download sfm_model_contours.json
wget -O sfm_model_contours.json "https://github.com/patrikhuber/eos/raw/master/share/sfm_model_contours.json"
```

Here is an example of how you might download the `shape_predictor_68_face_landmarks.dat`:

```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```

### Running the Project

1. **Prepare your input files**:
   - Place your input image (e.g., `input_image.jpg`) in the root directory of the project.
   - Place your audio file (e.g., `input_audio.wav`) in the root directory of the project.

2. **Run the script**:
   ```bash
   python3 main.py input_image.jpg input_audio.wav output_video.avi
   ```

   Replace `main.py` with the actual name of your Python script.

### Troubleshooting

- **ModuleNotFoundError**: Ensure all dependencies are installed. Activate your virtual environment and run `pip install -r requirements.txt`.
- **dlib or eos fails to install**: Make sure you have installed all the required system dependencies. For `dlib`, you might need to install additional tools like `cmake`.
- **Performance Issues**: Running heavy computations in WSL can be slower than native Linux. If performance is an issue, consider using a native Linux system or adjusting the model to a lighter version.
- **File not found errors**: Double-check the paths to your models and input files. They should be relative to the root of your project directory unless specified otherwise.

### Additional Notes

- For any additional help, refer to the documentation of the respective libraries used (OpenCV, dlib, librosa, eos).
- If you encounter any bugs or have suggestions, feel free to open an issue or submit a pull request on the repository.

### Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

This README aims to guide you through setting up and running the 3D facial animation project. If you have any questions or need further assistance, don't hesitate to ask in the community or create an issue in the repository.
