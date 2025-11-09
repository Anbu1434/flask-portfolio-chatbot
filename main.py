from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

# Load env variables
load_dotenv()

app = Flask(__name__)
CORS(app)
CORS(app, resources={r"/*": {"origins": "*"}})
# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Resume text
resume_text = """
Anbarasan A is a Computer Science and Engineering student at Anand Institute of Higher Technology (CGPA: 8.06).
Completed higher secondary at Sathya Saai Matriculation Higher Secondary School with GPA 7.6.

Technical Skills:
Languages: Python3, C, SQL, JavaScript, HTML5, CSS3.
Frameworks: Django, React.js, React Native, Node.js, Express.js, Flask.
Developer Tools: Git, GitHub, REST APIs, Visual Studio Code, PyCharm.
UI/UX Design: Figma, Sketch, Wireframing, Prototyping.
Soft Skills: Team Collaboration, Leadership, Adaptability, Conflict Resolution.

Work Experience:
Software Development Intern at Dvein Innovations Pvt Ltd (August 2025 - Present):
- Developed a real-time notifications system, improving user engagement by 30%.
- Implemented secure document tracking, reducing data errors by 25%.
- Optimized API response time by 40% with caching and efficient queries.
- Boosted team productivity by 20% by improving GitHub workflows and sprint planning.

Projects:
AI Travel Planner App (React Native, Gemini API, Firebase) - Dec 2023 to Feb 2024.
- Built a cross-platform AI-based travel planning app.
- Integrated Gemini API for real-time, AI-powered itinerary suggestions.
- Used Firebase for user authentication and trip history storage.

Event Management System (Django, MySQL, Database Masking) - Jan 2025 to July 2025.
- Built a secure web application for event management.
- Used data masking for privacy protection and developed an admin dashboard.

Certificates:
- Programming in Python (Geeks for Geeks, Mar 2023)
- Python Django (Geeks for Geeks, Dec 2024)
- UI/UX Design (Udemy, Mar 2024)
"""

# Split into text chunks
def chunk_text(text, size=300):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

chunks = chunk_text(resume_text)

# Lightweight embedding using TF-IDF
vectorizer = TfidfVectorizer()
embeddings = vectorizer.fit_transform(chunks)

@app.route("/api/chat", methods=["POST"])
def chat_api():
    try:
        user_query = request.json.get("query", "")
        if not user_query:
            return jsonify({"error": "Missing 'query'"}), 400

        # Retrieve top 3 relevant chunks
        query_vec = vectorizer.transform([user_query])
        sims = cosine_similarity(query_vec, embeddings).flatten()
        top_indices = sims.argsort()[-3:][::-1]
        context = " ".join([chunks[i] for i in top_indices])

        prompt = f"""
        You are an AI assistant representing Anbarasan A.
        Use the following resume context to answer professionally.

        Context:
        {context}

        User Question:
        {user_query}

        Answer naturally in first person as Anbarasan A.
        """

        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)

        return jsonify({
            "answer": response.text.strip(),
            "context_used": [chunks[i] for i in top_indices]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Anbarasan's Lightweight RAG Resume Chatbot API is running 🚀",
        "endpoint": "/api/chat",
        "method": "POST"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

