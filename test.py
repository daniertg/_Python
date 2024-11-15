import cv2
import numpy as np
import pickle
from scipy.interpolate import griddata

def load_model(model_path):
    """Load the trained model and label names"""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['label_names']

def preprocess_image(image_path, N=80):
    """Preprocess a single image for prediction"""
    epsilon = 1e-8
    
    # Read and process image
    img = cv2.imread(image_path, 0)
    if img is None:
        raise ValueError(f"Tidak dapat membaca gambar: {image_path}")
    
    # Compute FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fshift += epsilon
    
    # Compute magnitude spectrum
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    
    # Calculate radial profile
    def azimuthalAverage(image, center=None):
        y, x = np.indices(image.shape)
        if center is None:
            center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
        r = np.hypot(x - center[0], y - center[1])
        ind = np.argsort(r.flat)
        r_sorted = r.flat[ind]
        i_sorted = image.flat[ind]
        r_int = r_sorted.astype(int)
        deltar = r_int[1:] - r_int[:-1]
        rind = np.where(deltar)[0]
        nr = rind[1:] - rind[:-1]
        csim = np.cumsum(i_sorted, dtype=float)
        tbin = csim[rind[1:]] - csim[rind[:-1]]
        radial_prof = tbin / nr
        return radial_prof
    
    psd1D = azimuthalAverage(magnitude_spectrum)
    
    # Interpolate to fixed size
    points = np.linspace(0, N, num=psd1D.size)
    xi = np.linspace(0, N, num=N)
    interpolated = griddata(points, psd1D, xi, method='cubic')
    
    # Normalize
    interpolated = interpolated / (interpolated[0] + epsilon)
    
    return interpolated.reshape(1, -1)

def predict_soil_type(model, image_path, label_names):
    """Predict soil type for a new image"""
    # Preprocess image
    try:
        features = preprocess_image(image_path)
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None
    
    # Make prediction
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    # Get top 3 predictions with probabilities
    top_3_idx = np.argsort(probabilities)[-3:][::-1]
    top_3_predictions = []
    
    for idx in top_3_idx:
        label = idx + 1  # Adjust index to match label numbers
        prob = probabilities[idx]
        soil_type = label_names[label]
        top_3_predictions.append((soil_type, prob))
    
    return top_3_predictions

if __name__ == "__main__":
    # Load model dan label names
    model_path = '/content/drive/MyDrive/ML/tanah/soil_classifier_rf.pkl'
    model, label_names = load_model(model_path)
    
    # Path gambar yang akan diprediksi
    test_image_path = "/content/drive/MyDrive/ML/tanah/inceptisol-004.jpg"  # Ganti dengan path gambar yang ingin diprediksi
    
    # Lakukan prediksi
    print("Memprediksi jenis tanah...")
    try:
        predictions = predict_soil_type(model, test_image_path, label_names)
        print("\nHasil Prediksi:")
        for soil_type, probability in predictions:
            print(f"{soil_type}: {probability:.2%}")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")