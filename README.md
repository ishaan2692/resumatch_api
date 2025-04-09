# Resume Analyzer API - Setup and Deployment Guide



## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.8+
- pip (Python package manager)
- Google API key for Gemini AI
- Git (for version control)

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/resume-analyzer-api.git
cd resume-analyzer-api
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_google_api_key_here
```

## Running the Application

### Development mode
```bash
uvicorn main:app --reload
```

### Production mode
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`
