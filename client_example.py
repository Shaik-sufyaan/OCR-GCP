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
        self.api_url = api_url.rstrip('/')

    def health_check(self):
        """Check if API is healthy"""
        response = requests.get(f"{self.api_url}/health")
        response.raise_for_status()
        return response.json()

    def ocr(self, file_path):
        """
        Perform OCR on a single image or PDF (first page).

        Args:
            file_path: Path to image or PDF file

        Returns:
            dict with text, metadata, success
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f)}
            response = requests.post(f"{self.api_url}/ocr", files=files)

        response.raise_for_status()
        return response.json()

    def ocr_pdf(self, file_path, pages="1"):
        """
        Perform OCR on a PDF with specific pages.

        Args:
            file_path: Path to PDF file
            pages: Page spec like "1", "1-5", or "1,3,5"

        Returns:
            dict with per-page results
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f)}
            response = requests.post(
                f"{self.api_url}/ocr/pdf?pages={pages}", files=files
            )

        response.raise_for_status()
        return response.json()

    def ocr_batch(self, file_paths):
        """
        Perform OCR on multiple files.

        Args:
            file_paths: List of file paths

        Returns:
            dict with batch results
        """
        files = []
        for fp in file_paths:
            if not Path(fp).exists():
                print(f"Warning: Skipping {fp} (not found)")
                continue
            files.append(('files', (Path(fp).name, open(fp, 'rb'))))

        if not files:
            raise ValueError("No valid files provided")

        try:
            response = requests.post(f"{self.api_url}/ocr/batch", files=files)
            response.raise_for_status()
            return response.json()
        finally:
            for _, (_, f) in files:
                f.close()


def main():
    parser = argparse.ArgumentParser(description='OlmOCR API Client')
    parser.add_argument('api_url', help='API base URL')
    parser.add_argument('files', nargs='+', help='File(s) to process')
    parser.add_argument('--batch', action='store_true', help='Process as batch')
    parser.add_argument('--pages', default='1', help='PDF pages (e.g. "1-5")')

    args = parser.parse_args()

    client = OlmOCRClient(args.api_url)

    # Health check
    print("Checking API health...")
    health = client.health_check()
    print(f"API is {health['status']}\n")

    if args.batch:
        print(f"Processing {len(args.files)} files in batch mode...")
        results = client.ocr_batch(args.files)
        for result in results['results']:
            print(f"\n{'='*60}")
            print(f"File: {result['filename']}")
            print(f"Success: {result['success']}")
            if result['success']:
                print(f"\nExtracted Text:\n{result['text']}")
            else:
                print(f"Error: {result.get('error')}")
    else:
        for file_path in args.files:
            ext = Path(file_path).suffix.lower()
            print(f"\nProcessing: {file_path}")
            print('-' * 60)

            if ext == '.pdf' and args.pages != '1':
                result = client.ocr_pdf(file_path, pages=args.pages)
                for page_result in result['results']:
                    print(f"\n--- Page {page_result['page']} ---")
                    if page_result['success']:
                        print(page_result['text'])
                    else:
                        print(f"Error: {page_result.get('error')}")
            else:
                result = client.ocr(file_path)
                if result['success']:
                    print(f"Extracted Text:\n{result['text']}")
                else:
                    print(f"Error: {result.get('message')}")

    print(f"\n{'='*60}")
    print("Processing complete!")


if __name__ == "__main__":
    main()
