# Research RAG System using Python and LLM

A **Retrieval-Augmented Generation (RAG)** system designed to assist researchers by extracting insights and data from research papers (PDFs). The system uses **LlamaIndex** and **OpenAI's GPT-3.5-turbo** to query research papers, summarize content, and extract specific data.

---

## Features

- **PDF Text Extraction**: Extract text from research papers using `PyPDF`.
- **Retrieval-Augmented Generation (RAG)**:
  - Generate embeddings for text and image metadata using `sentence-transformers`.
  - Retrieve relevant information using FAISS for similarity search.
  - Generate answers using a pre-trained LLM (e.g., GPT-3.5-turbo).
- **Note Saving**: Save user notes to a file for future reference.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Shobhit3244/Research-RAG-System-using-Python-and-LLM.git
   cd Research-RAG-System-using-Python-and-LLM
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**:
   - Create a `.env` file in the root directory and add your OpenAI API key:
     ```plaintext
     OPENAI_API_KEY=your_openai_api_key
     ```

---

## Usage

### Step 1: Add Research Papers
Place your research paper PDFs in the `data/` directory.

### Step 2: Run the Application
Start the application by running:
```bash
python main.py
```

### Step 3: Interact with the System
- Enter your query at the prompt. For example:
  ```
  Enter a prompt (or q to Quit): What is the main contribution of this paper?
  ```
- The system will retrieve relevant information from the research papers and generate an answer.

### Step 4: Save Notes
You can save notes by using the `note_saver` tool. For example:
```
Enter a prompt (or q to Quit): Save note: The paper discusses a novel algorithm for image segmentation.
Note Saved
```

---

## Project Structure

```
Research-RAG-System-using-Python-and-LLM/
├── data/                   # Directory for research paper PDFs and notes
├── main.py                 # Main script to run the application
├── pdf.py                  # Handles PDF loading, indexing, and querying
├── note_engine.py          # Tool for saving user notes
├── prompts.py              # Custom prompt templates for the agent
├── requirements.txt        # List of dependencies
├── README.md               # Project documentation
└── .env                    # Environment variables (e.g., OpenAI API key)
```

---

## Custom Prompt Templates

The `prompts.py` file contains custom prompt templates for the agent. These templates are designed to help the agent extract insights and data from research papers. For example:

```python
research_paper_prompt = PromptTemplate(
    """\
You are a research assistant analyzing the following research paper:
---------------------
{context_str}
---------------------

User Query: {query_str}

Instructions:
{instruction_str}

Answer:
"""
)
```

---

## Dependencies

- Python 3.8+
- Libraries:
  - `llama-index`
  - `openai`
  - `PyPDF`
  - `sentence-transformers`
  - `python-dotenv`

---

## Future Enhancements

1. **Support for More File Types**: Extend the system to handle other document formats (e.g., Word, Excel).
2. **Advanced Querying**: Add support for complex queries involving multiple documents.
3. **Deployment**: Containerize the app using Docker and deploy it on a cloud platform.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [LlamaIndex](https://gpt-index.readthedocs.io/) for the RAG framework.
- [OpenAI](https://openai.com/) for the GPT-3.5-turbo model.
---

## Contact

For questions or feedback, please contact [Shobhit](https://github.com/Shobhit3244) or drop a mail to [shobhitkundu@gmail.com](mailto:shobhitkundu@gmail.com).