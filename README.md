# demo-rag-chat4docs-python
This app is a chatbot for uploaded documents with Retrieval-Augmented Generation (RAG). 
It works with different LLMs, e.g., OpenAI, Google Generative AI, and models hosted by HuggingFace and Ollama.

## Prerequisites
- Python 3.9 or higher+
- OpenAI API Key
- HuggingFace API token
- Ollama installed and the models in the code pulled
- Setting up the Environment Variables
   * `OPENAI_API_KEY`: OpenAI API Key
   * `OPENAI_BASE_URL`: OpenAI Base URL. Optional
   * `HUGGINGFACEHUB_API_TOKEN`: HuggingFace API token
   * `GOOGLE_API_KEY`: Google Generative AI API Key`

## Get Started
1. Clone this repo
2. Install [Python](https://www.python.org/)
3. Run the following command in the terminal
   ```
   pip install -r requirements.txt
   ```
4. Start the app by running the following command
   ```
   streamlit run app.py
   ```
5. Upload your documents, which can be in PDF, DOCX, TXT, or format.
6. Ask questions about your document.

## Built With
- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)
- [OpenAI](https://openai.com/)
- [Faiss](https://github.com/facebookresearch/faiss)
- [Google Generative AI](https://ai.google.dev/)
- [Ollama](https://ollama.com)
- [HuggingFace](https://huggingface.co)

## Next Steps
- Supporting more LLMs
- Supporting more embedding models
- Supporting more vector stores
- Supporting more document formats
- Fixing bugs
- Refactoring the code
