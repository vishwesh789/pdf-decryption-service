# PDF Decryption Service

A FastAPI service for decrypting password-protected PDF files, designed to work with the Expense Tracker React Native app.

## Features

- Decrypt password-protected PDF files
- Return decrypted PDF as base64
- Check PDF encryption status
- CORS enabled for React Native integration
- Health check endpoint

## API Endpoints

### `POST /decrypt-pdf`
Decrypt a password-protected PDF file.

**Parameters:**
- `file`: PDF file (multipart/form-data)
- `password`: Password string

**Response:**
```json
{
  "success": true,
  "decrypted_base64": "base64-encoded-pdf",
  "page_count": 5,
  "file_size": 12345,
  "was_encrypted": true,
  "message": "PDF decrypted successfully"
}
```

### `POST /check-encryption`
Check if a PDF is password-protected.

**Parameters:**
- `file`: PDF file (multipart/form-data)

**Response:**
```json
{
  "is_encrypted": true,
  "page_count": null,
  "file_size": 12345,
  "filename": "document.pdf"
}
```

### `GET /health`
Health check endpoint for monitoring.

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

3. Access the API at `http://localhost:8000`
4. View API docs at `http://localhost:8000/docs`

## Railway Deployment

1. Connect this repository to Railway
2. Railway will automatically detect the Python app and deploy
3. The service will be available at your Railway app URL
4. Set up auto-deployment by connecting to GitHub

## Environment Variables

No environment variables required for basic functionality.

## Dependencies

- FastAPI: Web framework
- PyPDF2: PDF processing library
- uvicorn: ASGI server
- python-multipart: File upload support