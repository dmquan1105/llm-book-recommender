# 📚 Semantic Book Recommender

The **Semantic Book Recommender** is a web application that helps users discover books based on natural language queries. By leveraging advanced machine learning models and semantic search, the app provides personalized book recommendations based on themes, tones, and categories.

## 🚀 Features

- **Natural Language Search**: Enter a description of the type of book you're looking for, and the app will recommend books that match your query.
- **Category Filtering**: Narrow down recommendations by selecting specific book categories (e.g., Fiction, Nonfiction, Children's Fiction).
- **Tone Filtering**: Refine results based on emotional tones such as Happy, Sad, Surprising, Angry, or Suspenseful.
- **Detailed Book Information**: View book titles, authors, descriptions, and cover images directly in the app.
- **Interactive Popovers**: Click on a book to see more details in an interactive popover.

## 🛠️ How It Works

1. **Data Loading**: The app loads a dataset of books enriched with emotional scores and categories.
2. **Semantic Search**: Using a pre-trained sentence transformer model, the app performs similarity searches to find books that match the user's query.
3. **Filtering and Sorting**: Recommendations are filtered by category and sorted by emotional tone if specified.
4. **Display**: The top recommendations are displayed with their details and cover images.

## 📦 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/llm-book-recommender.git
   cd llm-book-recommender
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the required datasets and place them in the project directory.

## ▶️ Usage

Run the app:

Open the app in your browser at http://localhost:8501.

Enter a description of the book you're looking for, select optional filters, and click Get Recommendations.

## 📂 Project Structure

app.py: The main Streamlit app file.
requirements.txt: List of Python dependencies.
books_with_emotions.csv: Dataset containing book metadata and emotional scores.
tagged_description.txt: Preprocessed text data for semantic search.
notebooks/: Jupyter notebooks for data exploration, sentiment analysis, and vector search.

## 🧠 Models and Libraries

Sentence Transformers: Used for semantic embeddings.
LangChain: For document loading and text splitting.
Pandas: For data manipulation.
Streamlit: For creating the interactive web app.

## 📊 Data Sources

The app uses a dataset of books enriched with metadata, descriptions, and emotional scores. The dataset is preprocessed to include:

Emotional scores (e.g., joy, sadness, anger) derived from book descriptions.
Simplified categories for easier filtering.

## 🛡️ Limitations

The app relies on pre-trained models, which may not always perfectly match user expectations.
Emotional tone filtering depends on the quality of the emotional scores in the dataset.

## 🙌 Acknowledgments

Hugging Face Transformers
LangChain
Streamlit
Dataset: 7k Books with Metadata
Enjoy discovering your next favorite book! 📖✨ ```
