from flask import Flask, render_template, request, redirect, url_for, session
import os, fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pinecone import ServerlessSpec
import pinecone 
#from pinecone.grpc import PineconeGRPC as Pinecone
from mistralai import Mistral
import os
from dotenv import load_dotenv



app = Flask(__name__)
app.secret_key = "supersecretkey"  # required for session

# Ensure uploads folder exists
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
model=SentenceTransformer("all-MiniLM-L6-v2")
load_dotenv()
pinecone_api=os.getenv('PINECONE_API')
pc=pinecone.Pinecone(api_key=pinecone_api)
index_name="upload1"
api_key=os.getenv('MISTRAL_API')
model_llm="open-mistral-7b"
client=Mistral(api_key=api_key)

instruction=f"""You are an expert at understanding text data and your peers often quiz you on the text data
they provide. You will be given a text data and a user query, the data that is most relevant to the user query will be provided
to you. I want you to interact with the user to answer whatever questions they have about the text data they have given you. Your knowledge base
id only the text data given to you and the most relevant information from the text data given to you.
By this, you basically act as an intermediary between the user and the text data, allowing them to know what they want
from the text data, just by asking you, not by actually reading the data."""



def save_vectors(vectors, metadata, docs, batch_size=50):
        index=pc.Index(index_name)
        upserts=[]

        for i, vector in enumerate(vectors):
            vector_id = f"{metadata['id']}_chunk_{i}"  # Unique ID for each chunk
            chunk_metadata = {
                "id": vector_id,
                "source": metadata["source"],
                "chunk": i,
                "text": docs[i][:500] # Add the text of the chunk here
            }
            upserts.append((vector_id, vector.tolist(), chunk_metadata))
            if len(upserts) >= batch_size:
                index.upsert(vectors=upserts)
                upserts = []
        # send remaining
        if upserts:
            index.upsert(vectors=upserts)


@app.route("/")
def home():
    return redirect(url_for("upload_page"))


@app.route("/upload", methods=["GET", "POST"])
def upload_page():
    if request.method == "POST":
        if "pdf" not in request.files:
            return "No file part", 400
        file = request.files["pdf"]

        if file.filename == "":
            return "No selected file", 400

        if file and file.filename.lower().endswith(".pdf"):
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            
            
            if index_name not in pc.list_indexes().names():
                pc.create_index(
                name=index_name,
                dimension=384,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )

            # store filename in session
            session["pdf_file"] = file.filename
            doc=fitz.open(file_path)
            text=""
            for page in doc:
                text+=page.get_text()
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs=text_splitter.split_text(text)
            embeddings = embeddings = model.encode(docs, batch_size=32, convert_to_numpy=True, show_progress_bar=True)
            save_vectors(embeddings, {"id": "doc_1", "source":"upload1"}, docs)
            return redirect("/chat")
        else:
            return "Invalid file type. Only PDF allowed.", 400

    return render_template("upload.html")


@app.route("/chat", methods=["GET", "POST"])
def chat_page():
    if "pdf_file" not in session:
        return redirect(url_for("upload_page"))

    bot_responses = []
    if request.method == "POST":
        user_message = request.form.get("message", "")
        if user_message.strip():
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs=text_splitter.split_text(user_message.strip())
            embeddings=model.encode(docs, batch_size=32, convert_to_numpy=True, show_progress_bar=True)

            # pip install "pinecone[grpc]"
            index=pc.Index(index_name)
            response=index.query(
    namespace="__default__",
    vector=embeddings[0].tolist(),
    top_k=3,
    include_values=False,
    include_metadata=True
)
            metadata_chunks = []
            for match in response.get("matches", []):
                if "metadata" in match and "text" in match["metadata"]:
                    metadata_chunks.append(match["metadata"]["text"])

    # Build augmented prompt
            context = "\n".join(metadata_chunks)
            augmented_prompt = (
        f"User question: {user_message.strip()}\n\n"
        f"Relevant extracted text from the PDF:\n{context}\n\n"
        "Answer the user question using ONLY the extracted text above."
    )
            
            chat_response=client.chat.complete(
        model=model_llm,
        messages=[
            {
                "role":"system",
                "content":instruction
            },
            {
                "role":"user",
                "content": augmented_prompt
            },
        
        ],
        max_tokens=500,
        temperature=0.6

    )
            
            bot_reply = chat_response.choices[0].message.content
            bot_responses.append((user_message, bot_reply))

    return render_template("chat.html", filename=session["pdf_file"], history=bot_responses)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

