import numpy as np
from sklearn.mixture import GaussianMixture
import joblib

class GestureNoteMapper:
    def __init__(self, n_components=8):
        self.gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        self.note_labels = []  # Store note names matching clusters

    def train(self, X, notes):
        """
        X: np.array shape (num_samples, features)
        notes: list of note labels corresponding to each sample
        """
        print("Training GMM on sensor data...")
        self.gmm.fit(X)

        # Assign most probable note to each GMM component
        cluster_labels = self.gmm.predict(X)
        component_note_map = {}
        for comp in range(self.gmm.n_components):
            assigned_notes = [notes[i] for i in range(len(notes)) if cluster_labels[i] == comp]
            if assigned_notes:
                # Most common note for this component
                common_note = max(set(assigned_notes), key=assigned_notes.count)
            else:
                common_note = "Unknown"
            component_note_map[comp] = common_note

        self.note_labels = [component_note_map[c] for c in range(self.gmm.n_components)]
        print("Note mapping per GMM component:", self.note_labels)

    def predict_note(self, x):
        """
        Predict note and confidence for a single sensor reading x.
        """
        x = x.reshape(1, -1)
        probs = self.gmm.predict_proba(x)[0]
        best_comp = np.argmax(probs)
        note = self.note_labels[best_comp]
        confidence = probs[best_comp] * 100  # percent
        return note, confidence

if __name__ == "__main__":
    # Example dummy training
    num_samples = 500
    features = 10

    # Generate synthetic sensor data (replace with your real dataset)
    X_train = np.random.randn(num_samples, features)
    notes_train = np.random.choice(["C", "D", "E", "F", "G", "A", "B"], num_samples)

    mapper = GestureNoteMapper(n_components=7)
    mapper.train(X_train, notes_train)

    # Example prediction
    test_sample = np.random.randn(features)
    predicted_note, confidence = mapper.predict_note(test_sample)
    print(f"Predicted Note: {predicted_note} with confidence {confidence:.2f}%")

    # Save model
    joblib.dump(mapper, "gesture_note_mapper.pkl")
