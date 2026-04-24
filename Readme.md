# CASMOCART AI: Precision Dermatological Intelligence 🧬✨

**Casmocart AI** is a next-generation AI-powered skincare analysis tool. It uses real-time computer vision and neural mesh tracking to provide clinical-grade skin assessments and personalized dermatological recommendations.

---

## 🚀 Key Features

- **Neural Mesh Tracking**: Utilizes real-time biometric face mapping to analyze specific facial zones (Forehead, Cheeks, Nose, Chin).
- **Advanced Skin Metrics**:
    - **Lipid Balance**: Detects surface oiliness and sebum distribution.
    - **Smoothness Index**: Analyzes skin texture, pores, and active breakouts.
    - **Biometric Identity**: Recognizes returning users based on facial geometry.
- **AI Expert Reports**: Generates professional-grade consultation reports using the Gemini 2.0 Flash model.
- **Interactive Dashboard**: A premium, glassmorphic UI for real-time visualization of skin health.
- **PDF Export**: Generate and download professional PDF reports for your skincare history.

---

## 🛠️ Technology Stack

- **Backend**: FastAPI (Python)
- **Frontend**: Vanilla HTML5, CSS3 (Glassmorphism), JavaScript
- **AI/ML**: 
    - **Mediapipe**: For high-fidelity face mesh tracking.
    - **OpenCV**: For image enhancement and texture analysis.
    - **OpenRouter (Gemini)**: For intelligent dermatological insights.
- **Styling**: Custom CSS with "Outfit" typography and dynamic micro-animations.

---

## 📦 Installation

### 1. Prerequisites
- Python 3.9 or higher installed on your system.
- A webcam for live analysis.

### 2. Clone the Repository
```bash
git clone https://github.com/your-repo/ai-for-her.git
cd ai-for-her
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configuration
Open `config.py` and add your OpenRouter API Key:
```python
OPENROUTER_API_KEY = "your_key_here"
```

---

## 🏃 How to Run

1. Start the application server:
   ```bash
   python app_server.py
   ```
2. Open your browser and navigate to:
   **[http://localhost:8001](http://localhost:8001)**

---

## 📂 Project Structure

- `app_server.py`: The FastAPI backend server handling routing and video streaming.
- `face_detect2.py`: The core engine for Mediapipe face mesh and skin texture analysis.
- `supervisor.py`: The AI logic layer that process user queries and skin data.
- `config.py`: Central configuration for API keys and camera settings.
- `static/`: Contains the frontend assets (HTML, CSS, Images).
- `user_profile.py`: Handles user identification and profile persistence.

---

## 👥 Meet the Team
Developed with passion by dedicated contributors:
*   **Sanjay** (@sanjayCodeXdev) — Lead Developer & AI Architect

---

## 🛡️ Disclaimer

*This application is for educational and experimental purposes. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified dermatologist for skin concerns.*

---

## 📄 License
This project is licensed under the MIT License.
