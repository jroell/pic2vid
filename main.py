import cv2
import dlib
import numpy as np
import librosa
import os
import logging
import eos

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
MODEL_PATH = "sfm_shape_3448.bin"
BLENDSHAPES_PATH = "expression_blendshapes_3448.bin"
MAPPING_PATH = "ibug_to_sfm.txt"
EDGE_TOPOLOGY_PATH = "sfm_3448_edge_topology.json"
CONTOUR_LANDMARKS_PATH = "ibug_to_sfm.txt"
MODEL_CONTOUR_PATH = "sfm_model_contours.json"
FRAME_RATE = 25
NUM_LANDMARKS = 68

# Check if necessary files exist
missing_files = [
    path
    for path in [
        PREDICTOR_PATH,
        MODEL_PATH,
        BLENDSHAPES_PATH,
        MAPPING_PATH,
        EDGE_TOPOLOGY_PATH,
        CONTOUR_LANDMARKS_PATH,
        MODEL_CONTOUR_PATH,
    ]
    if not os.path.exists(path)
]
if missing_files:
    missing_files_str = ", ".join(missing_files)
    logging.error(f"Missing required files: {missing_files_str}")
    raise FileNotFoundError(f"Missing required files: {missing_files_str}")

# Initialize face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# Load models and mappings
try:
    morphable_model = eos.morphablemodel.load_model(MODEL_PATH)
    blendshapes = eos.morphablemodel.load_blendshapes(BLENDSHAPES_PATH)
    landmark_mapper = eos.core.LandmarkMapper(MAPPING_PATH)
    edge_topology = eos.morphablemodel.load_edge_topology(EDGE_TOPOLOGY_PATH)
    contour_landmarks = eos.fitting.ContourLandmarks.load(CONTOUR_LANDMARKS_PATH)
    model_contour = eos.fitting.ModelContour.load(MODEL_CONTOUR_PATH)
except Exception as e:
    logging.error(f"Error loading models or mappings: {e}")
    raise


def get_landmarks(image: np.ndarray) -> np.ndarray:
    """
    Detects facial landmarks in the given image using dlib's frontal face detector.

    Parameters:
        image (np.ndarray): The input image in which to detect facial landmarks.

    Returns:
        np.ndarray: An array of shape (68, 2) containing the detected landmarks, or None if no faces are found.
    """
    try:
        faces = detector(image, 1)
        if not faces:
            logging.warning("No faces found in the image.")
            return None
        face = faces[0]
        landmarks = predictor(image, face)
        return np.array(
            [[landmarks.part(i).x, landmarks.part(i).y] for i in range(NUM_LANDMARKS)]
        )
    except Exception as e:
        logging.error(f"Error detecting landmarks: {e}")
        return None


def fit_3dmm_to_landmarks(image: np.ndarray, landmarks_2d: np.ndarray) -> tuple:
    """
    Fits the 3D Morphable Model to 2D landmarks from an image.

    Parameters:
        image (np.ndarray): The image from which landmarks were detected.
        landmarks_2d (np.ndarray): The 2D landmarks to which the 3DMM is fitted.

    Returns:
        tuple: A tuple containing the mesh, pose, shape coefficients, and blendshape coefficients, or None if fitting fails.
    """
    if landmarks_2d is None or landmarks_2d.size == 0:
        logging.error("No landmarks provided for fitting.")
        return None
    try:
        image_height, image_width = image.shape[:2]
        eos_landmarks = [
            eos.core.Landmark(str(idx + 1), point.tolist())
            for idx, point in enumerate(landmarks_2d)
        ]
        return eos.fitting.fit_shape_and_pose(
            morphable_model,
            eos_landmarks,
            landmark_mapper,
            image_width,
            image_height,
            edge_topology,
            contour_landmarks,
            model_contour,
            num_iterations=5,
            num_shape_coefficients_to_fit=30,
            lambda_p=30,
            lambda_b=40,
        )
    except Exception as e:
        logging.error(f"Failed to fit the 3DMM to the landmarks: {e}")
        return None


def animate_mesh(audio_path: str, blendshape_coeffs: np.ndarray) -> list:
    """
    Generates animation coefficients based on audio input using the MFCC feature of the audio.

    Parameters:
        audio_path (str): The path to the audio file.
        blendshape_coeffs (np.ndarray): Initial blendshape coefficients.

    Returns:
        list: A list of blendshape coefficients for each frame based on the audio.
    """
    try:
        y, sr = librosa.load(audio_path)
        phonemes = librosa.feature.mfcc(y=y, sr=sr)
        max_phoneme = np.max(np.abs(phonemes), axis=0)
        max_phoneme_value = max(np.max(max_phoneme), 1)  # Ensure non-zero division
        return [
            blendshape_coeffs
            * (0.5 + 0.5 * np.sin(2 * np.pi * phoneme / max_phoneme_value))
            for phoneme in max_phoneme
        ]
    except Exception as e:
        logging.error(f"Failed to animate mesh based on audio: {e}")
        return []


def render_animation(
    mesh: eos.morphablemodel.Mesh,
    animation_coeffs: list,
    pose: eos.core.Pose,
    shape_coeffs: np.ndarray,
    image_width: int,
    image_height: int,
    output_path: str,
):
    """
    Renders the animation of the mesh and saves it as a video.

    Parameters:
        mesh (eos.morphablemodel.Mesh): The 3D mesh to animate.
        animation_coeffs (list): The animation coefficients for each frame.
        pose (eos.core.Pose): The pose of the mesh.
        shape_coeffs (np.ndarray): The shape coefficients of the mesh.
        image_width (int): The width of the output video.
        image_height (int): The height of the output video.
        output_path (str): The path to save the output video.
    """
    try:
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"XVID"),
            FRAME_RATE,
            (image_width, image_height),
        )
        for coeffs in animation_coeffs:
            mesh.vertices = morphable_model.draw_sample(shape_coeffs, coeffs)
            image = eos.render.render(
                mesh,
                pose,
                morphable_model.get_texture_coordinates(),
                image_width,
                image_height,
                eos.render.make_orthographic_projection_matrix(
                    image_width, image_height
                ),
            )
            out.write(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        out.release()
    except Exception as e:
        logging.error(f"Failed to render animation: {e}")


def process_image_to_animation(
    image_path: str, audio_path: str, output_video_path: str
):
    """
    Main processing function to convert an image and audio input into an animated video.

    Parameters:
        image_path (str): Path to the input image.
        audio_path (str): Path to the input audio.
        output_video_path (str): Path to save the output animated video.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Image at {image_path} could not be loaded.")
            return
        landmarks_2d = get_landmarks(image)
        if landmarks_2d is None or landmarks_2d.size == 0:
            logging.error("No landmarks detected in the image.")
            return

        mesh, pose, shape_coeffs, blendshape_coeffs = fit_3dmm_to_landmarks(
            image, landmarks_2d
        )
        if mesh is None:
            logging.error("3DMM fitting failed.")
            return

        animation_coeffs = animate_mesh(audio_path, blendshape_coeffs)
        if not animation_coeffs:
            logging.error("Failed to create animation coefficients.")
            return

        render_animation(
            mesh,
            animation_coeffs,
            pose,
            shape_coeffs,
            image.shape[1],
            image.shape[0],
            output_video_path,
        )
    except Exception as e:
        logging.error(f"Failed to process image to animation: {e}")


# Example usage
if __name__ == "__main__":
    process_image_to_animation("input_image.jpg", "input_audio.wav", "output_video.avi")
