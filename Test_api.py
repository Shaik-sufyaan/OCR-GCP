#!/usr/bin/env python3
"""
Test script for OlmOCR API
Usage: python Test_api.py <api_url> [image_or_pdf_path]
"""

import requests
import sys
import json
from pathlib import Path


def test_health(base_url):
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{base_url}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    return response.status_code == 200


def test_root(base_url):
    """Test root endpoint"""
    print("Testing root endpoint...")
    response = requests.get(base_url)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    return response.status_code == 200


def test_ocr(base_url, file_path):
    """Test OCR endpoint with an image or PDF"""
    print(f"Testing OCR endpoint with: {file_path}")

    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        return False

    with open(file_path, 'rb') as f:
        files = {'file': (Path(file_path).name, f)}
        response = requests.post(f"{base_url}/ocr", files=files)

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result.get('success')}")
        if result.get('metadata'):
            print(f"Metadata: {json.dumps(result.get('metadata'), indent=2)}")
        print(f"Extracted text:\n{result.get('text')}\n")
        return True
    else:
        print(f"Error: {response.text}\n")
        return False


def test_pdf_ocr(base_url, pdf_path, pages="1"):
    """Test PDF OCR endpoint with specific pages"""
    print(f"Testing PDF OCR endpoint with: {pdf_path} (pages={pages})")

    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        return False

    with open(pdf_path, 'rb') as f:
        files = {'file': (Path(pdf_path).name, f)}
        response = requests.post(f"{base_url}/ocr/pdf?pages={pages}", files=files)

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Total pages processed: {result.get('total_pages')}")
        for page_result in result.get('results', []):
            print(f"\n--- Page {page_result.get('page')} ---")
            print(f"Success: {page_result.get('success')}")
            if page_result.get('success'):
                print(f"Text:\n{page_result.get('text')[:500]}...")
            else:
                print(f"Error: {page_result.get('error')}")
        return True
    else:
        print(f"Error: {response.text}\n")
        return False


def test_batch_ocr(base_url, file_paths):
    """Test batch OCR endpoint with multiple files"""
    print(f"Testing batch OCR endpoint with {len(file_paths)} files...")

    files = []
    for fp in file_paths:
        if Path(fp).exists():
            files.append(('files', (Path(fp).name, open(fp, 'rb'))))

    if not files:
        print("Error: No valid files found")
        return False

    response = requests.post(f"{base_url}/ocr/batch", files=files)

    # Close file handles
    for _, (_, f) in files:
        f.close()

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        results = response.json()
        print(f"Processed {len(results.get('results', []))} files")
        for result in results.get('results', []):
            print(f"\nFile: {result.get('filename')}")
            print(f"Success: {result.get('success')}")
            if result.get('success'):
                print(f"Text: {result.get('text')[:200]}...")
            else:
                print(f"Error: {result.get('error')}")
        return True
    else:
        print(f"Error: {response.text}\n")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python Test_api.py <api_url> [file_path]")
        print("Example: python Test_api.py http://localhost:8080 image.jpg")
        print("Example: python Test_api.py http://localhost:8080 document.pdf")
        sys.exit(1)

    base_url = sys.argv[1].rstrip('/')
    file_path = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Testing API at: {base_url}\n")
    print("=" * 60)

    # Test health and root
    health_ok = test_health(base_url)
    root_ok = test_root(base_url)

    if not (health_ok and root_ok):
        print("Basic health checks failed!")
        sys.exit(1)

    # Test OCR if file provided
    if file_path:
        ext = Path(file_path).suffix.lower()
        ocr_ok = test_ocr(base_url, file_path)

        # If it's a PDF, also test the dedicated PDF endpoint
        if ext == ".pdf":
            test_pdf_ocr(base_url, file_path, pages="1")

        if ocr_ok:
            print("All tests passed!")
        else:
            print("OCR test failed!")
            sys.exit(1)
    else:
        print("No file provided. Skipping OCR tests.")
        print("Basic tests passed!")

    print("=" * 60)


if __name__ == "__main__":
    main()
