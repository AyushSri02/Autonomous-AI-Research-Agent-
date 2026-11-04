import streamlit as st
import arxiv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os

st.set_page_config(page_title="Autonomous AI Research Agent", layout="wide")
st.title("📚 Autonomous AI Research Agent")
st.write("Search and summarize academic papers from arXiv automatically.")

# Sidebar for API Key
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

search_query = st.text_input("Enter Research Topic / Keywords (e.g., 'RAG evaluation')", "Retrieval Augmented Generation")
num_papers = st.slider("Number of papers to fetch", 1, 5, 2)

if st.button("Search & Analyze"):
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
        
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    # Create temp directory if it doesn't exist
    if not os.path.exists("temp"):
        os.makedirs("temp")
    
    with st.spinner("Searching arXiv..."):
        search = arxiv.Search(
            query=search_query,
            max_results=num_papers,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        client = arxiv.Client()
        papers = list(client.results(search))
        
    if not papers:
        st.warning("No papers found.")
    else:
        for idx, paper in enumerate(papers):
            st.subheader(f"{idx+1}. {paper.title}")
            st.write(f"**Authors:** {', '.join([a.name for a in paper.authors])}")
            st.write(f"**Published:** {paper.published.strftime('%Y-%m-%d')}")
            
            with st.expander("View Abstract"):
                st.write(paper.summary)
                
            with st.spinner(f"Downloading and summarizing paper {idx+1}..."):
                try:
                    # Modern fix: Explicitly use urllib to fetch the target PDF URL
                    import urllib.request
                    
                    pdf_path = f"temp/paper_{idx}.pdf"
                    urllib.request.urlretrieve(paper.pdf_url, pdf_path)
                    
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()
                    
                    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
                    
                    prompt_template = """Write a detailed structured summary of the following research text. 
Include Key Contributions, Methodology, and Findings.
TEXT: {text}
STRUCTURED SUMMARY:"""
                    PROMPT = PromptTemplate.from_template(prompt_template)
                    
                    chain = PROMPT | llm | StrOutputParser()
                    
                    combined_text = "\n\n".join([doc.page_content for doc in docs[:10]])
                    summary = chain.invoke({"text": combined_text})
                    
                    st.success("Analysis Complete!")
                    st.markdown("### AI Generated Research Summary")
                    st.write(summary)
                    
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)
                        
                except Exception as e:
                    st.error(f"Error processing paper: {str(e)}")