# 🧠 Skin Cancer Classification API (TFLite + FastAPI)

This is a FastAPI backend that uses a TensorFlow Lite (TFLite) model to classify skin cancer images into 7 different lesion types. The model is loaded once into memory and accepts input via either an image URL or a base64-encoded image.

---

## 📦 Features

- 🚀 FastAPI backend
- 🧠 TFLite model for efficient inference
- 🔥 Model caching for speed
- 🖼️ Accepts image via URL or base64
- ⚠️ Fallback error handling
- 🌐 Deployable to [Vercel](https://vercel.com/)

---

## 🧪 Supported Lesion Types

| Code   | Lesion Type                         |
|--------|-------------------------------------|
| `nv`   | Melanocytic nevi                    |
| `mel`  | Melanoma                            |
| `bkl`  | Benign keratosis-like lesions       |
| `bcc`  | Basal cell carcinoma                |
| `akiec`| Actinic keratoses                   |
| `vasc` | Vascular lesions                    |
| `df`   | Dermatofibroma                      |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/skin-cancer-api
cd skin-cancer-api
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Place the TFLite Model

Place your `model.tflite` file inside the `models/` folder:

```
/models
  └── model.tflite
```

### 4. Start the Server

```bash
uvicorn app.main:app --reload
```

---

## 📥 Example Payloads

### ✅ Option 1: Image via URL

```json
{
  "image_url": "https://upload.wikimedia.org/wikipedia/commons/7/7e/Melanoma.jpg"
}
```

### ✅ Option 2: Image via Base64

```json
{
  "base64_image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
}
```

---

## 🧪 Test with `curl`

```bash
curl -X POST http://127.0.0.1:8000/predict \
-H "Content-Type: application/json" \
-d "{\"image_url\": \"https://upload.wikimedia.org/wikipedia/commons/7/7e/Melanoma.jpg\"}"
```

---

## 🌐 Deploying to Vercel

This project includes a `vercel.json` to support deploying the FastAPI app to [Vercel](https://vercel.com/).

### 1. Install Vercel CLI

```bash
npm install -g vercel
```

### 2. Deploy

```bash
vercel
```

---

## 🧑‍💻 Folder Structure

```
.
├── app
│   ├── main.py         # API entrypoint
│   ├── model.py        # TFLite model loading + inference
│   └── utils.py        # Image loading/preprocessing
├── models
│   └── model.tflite    # Your local model file
├── requirements.txt
├── vercel.json
└── README.md
```

---

## 🛠 Tech Stack

- Python 3.10+
- FastAPI
- TensorFlow Lite
- Uvicorn
- Vercel (for optional deployment)

---

## 📬 License

MIT License. Use freely, credit appreciated!

---

## 💬 Questions?

Open an issue or ping [@yourname](https://github.com/yourgithub) if you have questions!
```

---

Let me know if you'd like:
- A badge section (build, deploy, version)
- Swagger/Redoc endpoint URLs
- To generate a logo/banner for the project header

I can also package this project into a ZIP with the full structure if you'd like.
