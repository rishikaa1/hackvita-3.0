import cv2 as cv
import numpy as np
import tensorflow as tf
import segmentation_models as sm

# Register custom objects for serialization
@tf.keras.utils.register_keras_serializable()
def dice_coef(y_true, y_pred, smooth=1):
    """Dice coefficient metric."""
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, 'float32'))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, 'float32'))
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

@tf.keras.utils.register_keras_serializable()
def dice_coef_loss(y_true, y_pred):
    """Dice coefficient loss function."""
    return 1 - dice_coef(y_true, y_pred)

@tf.keras.utils.register_keras_serializable()
def iou_score(y_true, y_pred, smooth=1):
    """IoU (Jaccard) score metric."""
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, 'float32'))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred > 0.5, 'float32'))  # Threshold like sm.metrics.iou_score
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

class RoadSegmentationModel:
    @staticmethod
    def preprocess_image(image_path):
        """Preprocess image to match training pipeline."""
        img = cv.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        original_height, original_width = img.shape[:2]
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_resized = cv.resize(img_rgb, (512, 512), interpolation=cv.INTER_CUBIC)
        img_normalized = img_resized.astype(np.float32) / 255.0
        return np.expand_dims(img_normalized, axis=0), (original_height, original_width)

    @staticmethod
    def predict(model, image_path):
        """Predict and return mask in original size."""
        img, (original_height, original_width) = RoadSegmentationModel.preprocess_image(image_path)
        pred = model.predict(img, verbose=0)
        
        # Extract the single-channel prediction
        pred = pred[0, :, :, 0]  # Get the first batch item and first channel
        
        # Make binary mask (0 or 255)
        binary_mask = (pred > 0.5).astype(np.uint8) * 255
        
        # Resize to original dimensions
        mask_resized = cv.resize(binary_mask, (original_width, original_height), interpolation=cv.INTER_NEAREST)
        
        return mask_resized

    @staticmethod
    def load_model(model_path):
        """Load model with all custom objects."""
        custom_objects = {
            'dice_coef': dice_coef,
            'dice_coef_loss': dice_coef_loss,
            'iou_score': iou_score,  # Define our own iou_score
            'binary_crossentropy_plus_jaccard_loss': sm.losses.bce_jaccard_loss
        }
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects)