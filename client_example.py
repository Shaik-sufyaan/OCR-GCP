#!/usr/bin/env python3
"""
Example client for OlmOCR API
Demonstrates how to use the deployed API
"""

import requests
import argparse
from pathlib import Path


class OlmOCRClient:
    """Client for OlmOCR API"""
    
    def __init__(self, api_url):
        """
        Initialize client
        
        Args:
            api_url: Base URL of the API (e.g., https://your-service.run.app)
        """
        self.api_url = api_url.rstrip('/')
    
    def health_check(self):
        """Check if API is healthy"""
        response = requests.get(f"{self.api_url}/health")
        response.raise_for_status()
        return response.json()
    
    def ocr_image(self, image_path):
        """
        Perform OCR on a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict: OCR result containing text
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        with open(image_path, 'rb') as f:
            files = {'file': (Path(image_path).name, f)}
            response = requests.post(f"{self.api_url}/ocr", files=files)
        
        response.raise_for_status()
        return response.json()
    
    def ocr_batch(self, image_paths):
        """
        Perform OCR on multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            dict: Batch OCR results
        """
        files = []
        for img_path in image_paths:
            if not Path(img_path).exists():
                print(f"Warning: Skipping {img_path} (not found)")
                continue
            files.append(('files', (Path(img_path).name, open(img_path, 'rb'))))
        
        if not files:
            raise ValueError("No valid image files provided")
        
        try:
            response = requests.post(f"{self.api_url}/ocr/batch", files=files)
            response.raise_for_status()
            return response.json()
        finally:
            # Close all file handles
            for _, (_, f) in files:
                f.close()


def main():
    parser = argparse.ArgumentParser(description='OlmOCR API Client')
    parser.add_argument('api_url', help='API base URL')
    parser.add_argument('images', nargs='+', help='Image file(s) to process')
    parser.add_argument('--batch', action='store_true', help='Process as batch')
    
    args = parser.parse_args()
    
    # Initialize client
    client = OlmOCRClient(args.api_url)
    
    # Health check
    print("Checking API health...")
    health = client.health_check()
    print(f"✓ API is {health['status']}")
    print()
    
    # Process images
    if args.batch:
        print(f"Processing {len(args.images)} images in batch mode...")
        results = client.ocr_batch(args.images)
        
        for result in results['results']:
            print(f"\n{'='*60}")
            print(f"File: {result['filename']}")
            print(f"Success: {result['success']}")
            if result['success']:
                print(f"\nExtracted Text:\n{result['text']}")
            else:
                print(f"Error: {result.get('error')}")
    else:
        for image_path in args.images:
            print(f"\nProcessing: {image_path}")
            print('-' * 60)
            
            result = client.ocr_image(image_path)
            
            if result['success']:
                print(f"Extracted Text:\n{result['text']}")
            else:
                print(f"Error: {result.get('message')}")
    
    print(f"\n{'='*60}")
    print("✓ Processing complete!")


if __name__ == "__main__":
    main()