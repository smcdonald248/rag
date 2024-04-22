## To run locally <br>

  

**Install python deps:**
`python3 -m pip install -r requirements.txt`
  

**Create a `.env` file that contains the following values:**
`OCTOAI_API_TOKEN=`
`PINECONE_API_KEY=`
 

**Run the app:**
`streamlit run app.py`
 

### Other Notes <br>


- Grab your API keys from the respective service websites:
	- https://octo.ai/
	- https://pinecone.io/
- Pinecone index defaults to `rag-demo` but you can create an index by any name and manual type your index name into the input box when RAG is enabled