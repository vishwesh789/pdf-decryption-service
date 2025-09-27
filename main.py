from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import io
import base64
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF Decryption Service", version="1.0.0")

# Add CORS middleware to allow React Native app to call this service
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "PDF Decryption Service is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "pdf-decryption"}

@app.post("/decrypt-pdf")
async def decrypt_pdf(
    file: UploadFile = File(...),
    password: str = Form(...)
):
    """
    Decrypt a password-protected PDF and return the decrypted content as base64.

    Args:
        file: The password-protected PDF file
        password: The password to decrypt the PDF

    Returns:
        {
            "success": true,
            "decrypted_base64": "base64-encoded-decrypted-pdf",
            "page_count": 5,
            "file_size": 12345
        }
    """
    try:
        logger.info(f"Received PDF decryption request for file: {file.filename}")
        logger.info(f"File size: {file.size} bytes")
        logger.info(f"Password provided: {'Yes' if password else 'No'}")

        # Read the uploaded file
        pdf_content = await file.read()
        logger.info(f"Read PDF content: {len(pdf_content)} bytes")

        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))

        # Check if PDF is encrypted
        if not pdf_reader.is_encrypted:
            logger.info("PDF is not encrypted, returning original content")
            decrypted_base64 = base64.b64encode(pdf_content).decode('utf-8')
            return {
                "success": True,
                "decrypted_base64": decrypted_base64,
                "page_count": len(pdf_reader.pages),
                "file_size": len(pdf_content),
                "was_encrypted": False,
                "message": "PDF was not encrypted"
            }

        logger.info("PDF is encrypted, attempting to decrypt...")

        # Try to decrypt with the provided password
        if not pdf_reader.decrypt(password):
            logger.error("Failed to decrypt PDF with provided password")
            raise HTTPException(
                status_code=400,
                detail="Invalid password. Could not decrypt the PDF."
            )

        logger.info("PDF decrypted successfully")

        # Create a new PDF writer with decrypted content
        pdf_writer = PyPDF2.PdfWriter()

        # Copy all pages from decrypted reader to writer
        page_count = len(pdf_reader.pages)
        logger.info(f"Copying {page_count} pages to decrypted PDF")

        for page_num in range(page_count):
            page = pdf_reader.pages[page_num]
            pdf_writer.add_page(page)

        # Write the decrypted PDF to a bytes buffer
        decrypted_buffer = io.BytesIO()
        pdf_writer.write(decrypted_buffer)
        decrypted_bytes = decrypted_buffer.getvalue()

        # Encode to base64
        decrypted_base64 = base64.b64encode(decrypted_bytes).decode('utf-8')

        logger.info(f"Decryption successful. Original size: {len(pdf_content)}, Decrypted size: {len(decrypted_bytes)}")

        return {
            "success": True,
            "decrypted_base64": decrypted_base64,
            "page_count": page_count,
            "file_size": len(decrypted_bytes),
            "original_file_size": len(pdf_content),
            "was_encrypted": True,
            "message": "PDF decrypted successfully"
        }

    except HTTPException:
        # Re-raise HTTP exceptions (like invalid password)
        raise
    except Exception as e:
        logger.error(f"Error during PDF decryption: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during PDF decryption: {str(e)}"
        )

@app.post("/check-encryption")
async def check_pdf_encryption(file: UploadFile = File(...)):
    """
    Check if a PDF is password-protected without attempting to decrypt it.

    Args:
        file: The PDF file to check

    Returns:
        {
            "is_encrypted": true,
            "page_count": null,
            "file_size": 12345
        }
    """
    try:
        logger.info(f"Checking encryption status for file: {file.filename}")

        # Read the uploaded file
        pdf_content = await file.read()

        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))

        is_encrypted = pdf_reader.is_encrypted
        page_count = len(pdf_reader.pages) if not is_encrypted else None

        logger.info(f"Encryption check result: {is_encrypted}")

        return {
            "is_encrypted": is_encrypted,
            "page_count": page_count,
            "file_size": len(pdf_content),
            "filename": file.filename
        }

    except Exception as e:
        logger.error(f"Error checking PDF encryption: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error checking PDF encryption: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)