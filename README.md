(Due to technical issues, the search service is temporarily unavailable.)

Apologies for the confusion earlier! Let me provide you with a **complete and clean README.md** file for your Research Assistant RAG System project. This will include all the necessary details, code snippets, and instructions in proper Markdown format.

---

```markdown
# Research Assistant RAG System

A **Retrieval-Augmented Generation (RAG)** system designed to assist researchers by extracting and analyzing information from research papers (PDFs). The system can extract text and images from PDFs, retrieve relevant information, and generate answers to user queries using a Large Language Model (LLM).

---

## Features

- **PDF Text Extraction**: Extract text from research papers using `PyMuPDF`.
- **Image Extraction**: Extract images from PDFs and perform OCR (Optical Character Recognition) using `pytesseract`.
- **Retrieval-Augmented Generation (RAG)**:
  - Generate embeddings for text and image metadata using `sentence-transformers`.
  - Retrieve relevant information using FAISS for similarity search.
  - Generate answers using a pre-trained LLM (e.g., GPT-2).
- **API Integration**: Expose the system as a REST API using Flask.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/research-assistant-rag-system.git
   cd research-assistant-rag-system
   ```

2. **Install Dependencies**:
   ```bash
   pip install transformers faiss-cpu sentence-transformers flask PyMuPDF pytesseract pillow
   ```

3. **Install Tesseract OCR**:
   - Download and install Tesseract OCR from [here](https://github.com/tesseract-ocr/tesseract).
   - Add Tesseract to your system PATH.

---

## Usage

### Step 1: Extract Text and Images from PDFs
Place your research paper PDFs in the `data/` directory.

```python
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)
        for img in image_list:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            images.append(image_bytes)
    return images

def extract_text_from_image(image_bytes):
    image = Image.open(image_bytes)
    text = pytesseract.image_to_string(image)
    return text

# Example usage
pdf_text = extract_text_from_pdf("data/research_paper.pdf")
pdf_images = extract_images_from_pdf("data/research_paper.pdf")
for i, img in enumerate(pdf_images):
    text = extract_text_from_image(img)
    print(f"Text from image {i}:\n{text}\n")
```

### Step 2: Build the RAG System
Run the RAG system to retrieve and generate answers.

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# Load models
embedder = SentenceTransformer('all-MiniLM-L6-v2')
generator = pipeline("text-generation", model="gpt2")

# Generate embeddings for text and image metadata
text_embeddings = embedder.encode([pdf_text])
image_metadata = [extract_text_from_image(img) for img in pdf_images]
image_embeddings = embedder.encode(image_metadata)

# Build a FAISS index
all_embeddings = np.vstack([text_embeddings, image_embeddings])
dimension = all_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(all_embeddings)

def retrieve_information(query, top_k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for i in indices[0]:
        if i < len(text_embeddings):
            results.append(("text", pdf_text))
        else:
            results.append(("image", pdf_images[i - len(text_embeddings)]))
    return results

def generate_answer(query, context):
    prompt = f"Question: {query}\nContext: {context}\nAnswer:"
    answer = generator(prompt, max_length=100, num_return_sequences=1)
    return answer[0]['generated_text']

# Example query
query = "What is the main contribution of this paper?"
retrieved_results = retrieve_information(query)
context = " ".join([result for result_type, result in retrieved_results if result_type == "text"])
answer = generate_answer(query, context)
print("Generated Answer:", answer)
```

### Step 3: Run the Flask API
Start the Flask API to interact with the RAG system.

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    query = data.get("query")
    retrieved_results = retrieve_information(query)
    context = " ".join([result for result_type, result in retrieved_results if result_type == "text"])
    answer = generate_answer(query, context)
    image_results = [result for result_type, result in retrieved_results if result_type == "image"]
    return jsonify({"query": query, "answer": answer, "image_results": len(image_results)})

if __name__ == '__main__':
    app.run(debug=True)
```

Run the Flask app:
```bash
python app.py
```

Send a POST request to the `/predict` endpoint:
```bash
curl -X POST "http://127.0.0.1:5000/predict" -H "Content-Type: application/json" -d '{"query": "What is the main contribution of this paper?"}'
```

---

## Project Structure

```
research-assistant-rag-system/
├── data/                   # Directory for research paper PDFs
├── app.py                  # Flask API for the RAG system
├── pdf_utils.py            # Utility functions for PDF text and image extraction
├── rag_system.py           # Core RAG system implementation
├── requirements.txt        # List of dependencies
└── README.md               # Project documentation
```

---

## Dependencies

- Python 3.8+
- Libraries:
  - `transformers`
  - `faiss-cpu`
  - `sentence-transformers`
  - `flask`
  - `PyMuPDF`
  - `pytesseract`
  - `Pillow`

---

## Future Enhancements

1. **Support for Multiple PDFs**: Scale the system to handle multiple research papers.
2. **Advanced OCR**: Use state-of-the-art OCR models for better text extraction.
3. **Better LLMs**: Integrate GPT-3.5 or GPT-4 for improved answer generation.
4. **Deployment**: Containerize the app using Docker and deploy it on a cloud platform.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for pre-trained models and libraries.
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search.
- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF text and image extraction.
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for optical character recognition.

---

## Contact

For questions or feedback, please contact [Shobhit Kundu] at [shobhitkundu@gmail.com](mailto:shobhitkundu@gmail.com).
``` 