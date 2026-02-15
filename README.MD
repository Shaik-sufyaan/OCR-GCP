# OlmOCR API

OCR API powered by [allenai/olmOCR-2-7B-1025-FP8](https://huggingface.co/allenai/olmOCR-2-7B-1025-FP8), deployed on GCP Cloud Run with an NVIDIA L4 GPU.

**Base URL:** `https://olmocr-api-5ejysn2nla-uk.a.run.app`

## Endpoints

### Health Check

```
GET /health
```

```bash
curl https://olmocr-api-5ejysn2nla-uk.a.run.app/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "backend": "vllm"
}
```

---

### OCR - Single Image or PDF (first page)

```
POST /ocr
```

Upload a single image (JPEG, PNG, BMP, TIFF, WebP) or PDF (processes first page only).

```bash
curl -X POST https://olmocr-api-5ejysn2nla-uk.a.run.app/ocr \
  -F "file=@image.jpg"
```

**Response:**
```json
{
  "text": "Extracted text content...",
  "metadata": {
    "primary_language": "English",
    "is_table": "False",
    "is_diagram": "False"
  },
  "success": true,
  "message": "OCR completed successfully"
}
```

---

### OCR - Multi-Page PDF

```
POST /ocr/pdf?pages={page_spec}
```

Process specific pages from a PDF. Pages are processed in parallel for speed.

| Parameter | Type   | Default | Description                                      |
|-----------|--------|---------|--------------------------------------------------|
| `file`    | file   | required| PDF file upload                                  |
| `pages`   | string | `"1"`   | Page spec: `"1"`, `"1-5"`, `"1,3,5"`, or `"1-3,7"` |

```bash
# Single page
curl -X POST "https://olmocr-api-5ejysn2nla-uk.a.run.app/ocr/pdf?pages=1" \
  -F "file=@document.pdf"

# Page range
curl -X POST "https://olmocr-api-5ejysn2nla-uk.a.run.app/ocr/pdf?pages=1-5" \
  -F "file=@document.pdf"

# Specific pages
curl -X POST "https://olmocr-api-5ejysn2nla-uk.a.run.app/ocr/pdf?pages=1,3,7" \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "total_pages": 5,
  "results": [
    {
      "page": 1,
      "text": "Page 1 text...",
      "metadata": { "primary_language": "English" },
      "success": true
    },
    {
      "page": 2,
      "text": "Page 2 text...",
      "metadata": { "primary_language": "English" },
      "success": true
    }
  ]
}
```

---

### OCR - Batch (Multiple Files)

```
POST /ocr/batch
```

Upload multiple images/PDFs at once. PDFs are processed first page only.

```bash
curl -X POST https://olmocr-api-5ejysn2nla-uk.a.run.app/ocr/batch \
  -F "files=@image1.jpg" \
  -F "files=@image2.png" \
  -F "files=@doc.pdf"
```

**Response:**
```json
{
  "results": [
    { "filename": "image1.jpg", "text": "...", "metadata": {}, "success": true },
    { "filename": "image2.png", "text": "...", "metadata": {}, "success": true },
    { "filename": "doc.pdf", "text": "...", "metadata": {}, "success": true }
  ]
}
```

---

## Python Integration

```python
import requests

API_URL = "https://olmocr-api-5ejysn2nla-uk.a.run.app"

# Single image/PDF (first page)
with open("document.pdf", "rb") as f:
    resp = requests.post(f"{API_URL}/ocr", files={"file": f})
    print(resp.json()["text"])

# Multi-page PDF
with open("document.pdf", "rb") as f:
    resp = requests.post(f"{API_URL}/ocr/pdf?pages=1-5", files={"file": f})
    for page in resp.json()["results"]:
        print(f"Page {page['page']}: {page['text']}")
```

## JavaScript/Node.js Integration

```javascript
// Using fetch + FormData
const form = new FormData();
form.append("file", fileBlob, "document.pdf");

const resp = await fetch("https://olmocr-api-5ejysn2nla-uk.a.run.app/ocr/pdf?pages=1-5", {
  method: "POST",
  body: form,
});
const data = await resp.json();
data.results.forEach((page) => {
  console.log(`Page ${page.page}: ${page.text}`);
});
```

## Limits

| Limit              | Value   |
|---------------------|---------|
| Max file size        | 50 MB   |
| Max concurrent pages | 4       |
| Request timeout      | 300s    |
| Supported images     | JPEG, PNG, BMP, TIFF, WebP |
| Supported documents  | PDF     |

## Error Responses

| Status | Meaning                          |
|--------|----------------------------------|
| 400    | Unsupported file type or file too large |
| 500    | OCR processing failed            |
| 503    | Model not ready (cold start ~60s) |

Errors return:
```json
{
  "text": "",
  "success": false,
  "message": "Error description"
}
```

## Notes

- First request after idle may take ~60s (cold start: container + model loading).
- The API uses CORS `*` so it can be called from any frontend.
- Interactive docs available at `/docs` (Swagger UI).
