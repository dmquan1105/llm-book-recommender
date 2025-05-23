import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv()

@st.cache_data
def load_books():
    books_df = pd.read_csv("books_with_emotions.csv")
    books_df['large_thumbnail'] = books_df['thumbnail'] + '&fife=w800'
    books_df['large_thumbnail'] = np.where(
        books_df['large_thumbnail'].isna(),
        'cover-not-found.jpg',
        books_df['large_thumbnail'],
    )
    for col in ['title', 'authors', 'description', 'large_thumbnail', 'isbn13', 'simple_categories', 'joy', 'surprise', 'anger', 'fear', 'sadness']:
        if col not in books_df.columns:
            if books_df.get(col, pd.Series([])).dtype == 'object': # Check if column exists before checking dtype
                 books_df[col] = books_df.get(col, pd.Series(["N/A"] * len(books_df)))
            else:
                 books_df[col] = books_df.get(col, pd.Series([0] * len(books_df)))
    return books_df

@st.cache_resource
def load_db():
    raw_documents = TextLoader('tagged_description.txt').load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db_books = Chroma.from_documents(documents, embedding=embedding_model)
    return db_books

books_data = load_books()
db_books_vectorstore = load_db()

st.title("📚 Semantic Book Recommender")
st.markdown("Use natural language to get book recommendations based on theme and tone.")

if "selected_book" not in st.session_state:
    st.session_state.selected_book = None
if "displayed_recommendations" not in st.session_state:
    st.session_state.displayed_recommendations = pd.DataFrame()

query = st.text_input("Enter a book description", placeholder="e.g. A detective solving a mystery")
categories = ['All'] + sorted(books_data['simple_categories'].unique().tolist())
tones = ['All', 'Happy', 'Surprising', 'Angry', 'Suspenseful', 'Sad']

col1, col2 = st.columns(2)
with col1:
    category_filter = st.selectbox("Choose category", categories, index=0, key="category_filter")
with col2:
    tone_filter = st.selectbox("Choose tone", tones, index=0, key="tone_filter")

if st.button("🔍 Get Recommendations"):
    st.session_state.selected_book = None
    if query.strip():
        with st.spinner("Generating recommendations..."):
            try:
                recs = db_books_vectorstore.similarity_search(query, k=50)
                books_list_isbns = []
                for rec in recs:
                    parts = rec.page_content.strip('"').split(maxsplit=1)
                    if parts and parts[0].isdigit():
                        books_list_isbns.append(int(parts[0]))
                    elif parts and parts[0].startswith("ISBN:"):
                        isbn_val = parts[0].replace("ISBN:", "").strip()
                        if isbn_val.isdigit():
                            books_list_isbns.append(int(isbn_val))
                
                filtered_recs = books_data[books_data['isbn13'].isin(books_list_isbns)].copy()

                if category_filter != 'All':
                    filtered_recs = filtered_recs[filtered_recs['simple_categories'] == category_filter]
                
                if tone_filter != 'All':
                    emotion_col_map = {"Happy": "joy", "Surprising": "surprise", "Angry": "anger", "Suspenseful": "fear", "Sad": "sadness"}
                    if tone_filter in emotion_col_map:
                        emotion_col = emotion_col_map[tone_filter]
                        if emotion_col in filtered_recs.columns:
                             filtered_recs = filtered_recs.sort_values(by=emotion_col, ascending=False)
                        else:
                            st.warning(f"Emotion column '{emotion_col}' not found for tone filtering.")
                st.session_state.displayed_recommendations = filtered_recs.head(9)
            except Exception as e:
                st.error(f"❌ Error generating recommendations: {e}")
                st.session_state.displayed_recommendations = pd.DataFrame()
    else:
        st.warning("Please enter a description to get recommendations.")
        st.session_state.displayed_recommendations = pd.DataFrame()

if not st.session_state.displayed_recommendations.empty:
    df_to_display = st.session_state.displayed_recommendations.copy()
    if category_filter != 'All': # Allow live filtering of displayed results
        df_to_display = df_to_display[df_to_display['simple_categories'] == category_filter]
    if tone_filter != 'All':
        emotion_col_map = {"Happy": "joy", "Surprising": "surprise", "Angry": "anger", "Suspenseful": "fear", "Sad": "sadness"}
        if tone_filter in emotion_col_map:
            emotion_col = emotion_col_map[tone_filter]
            if emotion_col in df_to_display.columns:
                df_to_display = df_to_display.sort_values(by=emotion_col, ascending=False)
    df_to_display = df_to_display.head(9)

    if not df_to_display.empty:
        for row_chunk in np.array_split(df_to_display, 3):
            cols = st.columns(3)
            for i, (idx, row) in enumerate(row_chunk.iterrows()):
                with cols[i]:
                    title = str(row.get('title', 'N/A'))
                    authors = str(row.get('authors', 'N/A')).replace(';', ', ')
                    large_thumbnail = row.get('large_thumbnail', 'cover-not-found.jpg')
                    description = row.get('description', 'No description available.')

                    st.image(large_thumbnail, use_container_width=True, caption=title)
                    
                    with st.popover(f"📖 {title}"):
                        if st.button(f"📘 {title}", key=f"popover_btn_{row['isbn13']}_{idx}"):
                            pass  # Just to show popover trigger as a button
                        st.image(large_thumbnail, width=200)
                        st.markdown(f"### {title}")
                        st.markdown(f"*by {authors}*")
                        st.markdown(description)
    elif query.strip():
         st.info("No books match your current filter criteria from the initial recommendations.")

if st.session_state.selected_book:
    book = st.session_state.selected_book
    title = book.get('title', 'N/A')
    authors = str(book.get('authors', 'N/A')).replace(';', ', ')
    large_thumbnail = book.get('large_thumbnail', 'cover-not-found.jpg')
    description = book.get('description', 'No description available.')
        
    with st.popover(f"📖 {title}"):
        st.image(large_thumbnail, width=200)
        st.markdown(f"### {title}")
        st.markdown(f"*by {authors}*")
        st.markdown(description)

    st.session_state.selected_book = None