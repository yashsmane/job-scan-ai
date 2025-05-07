import os
import streamlit as st
from dotenv import load_dotenv
import pdfplumber
import openai
from agno.agent import Agent


load_dotenv()

def setup_openai(api_key: str):
    openai.api_key = api_key

class ResumeAgent(Agent):
    """Autonomous Resume Optimization Agent using OpenAI and Agno."""
    def __init__(self, openai_api_key: str, model: str = "gpt-4"):
        setup_openai(openai_api_key)
        self.model = model
        super().__init__(name="ResumeOptimizer", description="Enhances resumes based on job descriptions.")
    
    def optimize_resume(self, resume: str, job_desc: str) -> str:
        """Optimize the resume based on job description using OpenAI's GPT-4."""
        prompt = f"""
        Optimize the following resume to better match the job description.
        Ensure it highlights the most relevant skills and experience while maintaining professionalism.
        
        **Resume:**
        {resume}
        
        **Job Description:**
        {job_desc}
        """
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "system", "content": "You are an expert resume optimizer."},
                      {"role": "user", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"].strip()
    
    def execute(self, resume: str, job_desc: str):
        return self.optimize_resume(resume, job_desc)


def main():
    st.set_page_config(page_title="AI Resume Optimizer", page_icon="📄", layout="wide")
    st.title("📄 AI Resume Optimizer")
    st.info("Optimize your resume to better match job descriptions.")
    
    openai_key = os.getenv("OPENAI_API_KEY", "")
    
    if not openai_key:
        openai_key = st.text_input("Enter your OpenAI API Key", type="password")
    
    if openai_key:
        resume_agent = ResumeAgent(openai_api_key=openai_key)
    else:
        st.warning("Please provide an OpenAI API key to continue.")
        return
    
    with st.form("resume_form"):
        st.subheader("📄 Resume Optimization")
        resume_file = st.file_uploader("Upload Your Resume (PDF or TXT)", type=["pdf", "txt"])
        resume_text = ""
        
        if resume_file is not None:
            try:
                if resume_file.type == "application/pdf":
                    with pdfplumber.open(resume_file) as pdf:
                        resume_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                else:
                    resume_text = resume_file.read().decode("utf-8", errors="replace")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return
        else:
            resume_text = st.text_area("Or Paste Your Resume")
        
        job_desc = st.text_area("Paste Job Description")
        submit_button = st.form_submit_button("Submit")
    
    if submit_button and resume_text.strip() and job_desc.strip():
        with st.spinner("Optimizing resume..."):
            optimized_resume = resume_agent.execute(resume_text, job_desc)
            st.subheader("✅ Optimized Resume")
            st.markdown(optimized_resume)
            st.download_button("Download Optimized Resume", optimized_resume, "optimized_resume.txt")

if __name__ == "__main__":
    main()
