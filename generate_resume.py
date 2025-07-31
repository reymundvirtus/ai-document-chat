import google.generativeai as genai
import os

# setup api key
api_key = os.environ["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

# setup model
model = genai.GenerativeModel("gemini-2.5-flash")

def generate_resume(user_goal, experience):
    prompt = f"""
        You are an expert resume writer assistant.

        The user is applying for this role: "{user_goal}"

        Here is their background:
        {experience}

        Your task is to generate a full resume tailored for that job, including:
        - Professional summary
        - Work experience (with bullet points)
        - Skills (with bullet points)
        - Education

        Use a modern and professional tone.
        Avoid generic phrases - make it impactful and quantifiable if possible.
    """

    response = model.generate_content(prompt)
    return response.text

# usage
if __name__ == "__main__":
    # user inputs
    user_goal = "Frontend Developer role at Canva"
    experience = """
        - 3 years of experience as a frontend developer
        - Built responsive UI's using React and TailwindCSS
        - Collaborated with UX designers to improve mobile usability
        - Integrated API's and handled frontend testing
        - Interned at a local tech startup
        - Bachelor of Science in Information Technology
    """

    print("\n📝 Generated Resume:\n")
    resume = generate_resume(user_goal, experience)
    print(resume)

    # feedback loop
    feedback = input("\nWould you like to revise this resume? (e.g., 'Make it shorter or more assertive'):\n> ")
    if feedback.strip():
        revision_prompt = f"Please revise the resume above with this instruction: {feedback}"
        revised_response = model.generate_content([resume, revision_prompt])
        print("\n🔁 Revised Resume:\n")
        print(revised_response.text)
    else:
        print("\n📝 Final Resume:\n")
        print(resume)