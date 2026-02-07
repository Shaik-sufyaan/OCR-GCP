#!/usr/bin/env python3
"""
Test script for OlmOCR API
Usage: python test_api.py <api_url> <image_path>
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


def test_ocr(base_url, image_path):
    """Test OCR endpoint with image"""
    print(f"Testing OCR endpoint with image: {image_path}")
    
    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        return False
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{base_url}/ocr", files=files)
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result.get('success')}")
        print(f"Extracted text:\n{result.get('text')}\n")
        return True
    else:
        print(f"Error: {response.text}\n")
        return False


def test_batch_ocr(base_url, image_paths):
    """Test batch OCR endpoint with multiple images"""
    print(f"Testing batch OCR endpoint with {len(image_paths)} images...")
    
    files = []
    for img_path in image_paths:
        if Path(img_path).exists():
            files.append(('files', open(img_path, 'rb')))
    
    if not files:
        print("Error: No valid image files found")
        return False
    
    response = requests.post(f"{base_url}/ocr/batch", files=files)
    
    # Close file handles
    for _, f in files:
        f.close()
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        results = response.json()
        print(f"Processed {len(results.get('results', []))} images")
        for i, result in enumerate(results.get('results', [])):
            print(f"\nImage {i+1}: {result.get('filename')}")
            print(f"Success: {result.get('success')}")
            if result.get('success'):
                print(f"Text: {result.get('text')[:100]}...")
            else:
                print(f"Error: {result.get('error')}")
        return True
    else:
        print(f"Error: {response.text}\n")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <api_url> [image_path]")
        print("Example: python test_api.py http://localhost:8080 image.jpg")
        sys.exit(1)
    
    base_url = sys.argv[1].rstrip('/')
    image_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"Testing API at: {base_url}\n")
    print("=" * 60)
    
    # Test health and root
    health_ok = test_health(base_url)
    root_ok = test_root(base_url)
    
    if not (health_ok and root_ok):
        print("Basic health checks failed!")
        sys.exit(1)
    
    # Test OCR if image provided
    if image_path:
        ocr_ok = test_ocr(base_url, image_path)
        if ocr_ok:
            print("✓ All tests passed!")
        else:
            print("✗ OCR test failed!")
            sys.exit(1)
    else:
        print("No image provided. Skipping OCR tests.")
        print("✓ Basic tests passed!")
    
    print("=" * 60)


if __name__ == "__main__":
    main()