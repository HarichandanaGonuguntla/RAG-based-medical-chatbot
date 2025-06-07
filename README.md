# Health Assistant Application

![](https://github.com/HarichandanaGonuguntla/Cold-email-generator/blob/main/Coldemailgenerator.png)

This project presents an end-to-end Generative AI application that automates the creation of personalized cold emails. Tailored for Software and AI Services companies, the Cold Email Generator streamlines the outreach process by crafting context-aware emails targeting potential clients. By integrating Meta’s LLaMA 3.1 large language model, LangChain for orchestration, ChromaDB as a vector store, and a user-friendly Streamlit interface, this tool exemplifies the practical application of open-source AI technologies in business development.

🧠 Project Overview
In the competitive landscape of software services, companies often seek innovative methods to connect with prospective clients. One effective strategy is to identify organizations actively hiring for roles that align with the services offered. This project automates the process by:

Inputting a Company's Careers Page URL: Users provide the URL of a target company's careers page.

Extracting Job Listings: The tool scrapes job postings from the provided URL, focusing on roles relevant to the user's services.

Retrieving Relevant Portfolio Links: Using ChromaDB, the system fetches pertinent portfolio pieces that match the job descriptions.

Generating Personalized Cold Emails: Leveraging LLaMA 3.1 via LangChain, the tool crafts customized emails that highlight how the user's services align with the target company's needs.

This approach ensures that outreach efforts are timely, relevant, and tailored, increasing the likelihood of engagement from potential clients.

🧰 Tech Stack
LLaMA 3.1 (70B): Meta's open-source large language model for generating coherent and contextually relevant text.

LangChain: Facilitates the orchestration of prompts and manages interactions between components.

ChromaDB: A vector database that stores and retrieves portfolio information based on semantic similarity.

Streamlit: Provides an intuitive web interface for users to input data and view generated emails.

Groq API: Hosts the LLaMA 3.1 model, enabling efficient inference and scalability.

🏗️ System Architecture
The system comprises the following components:

Web Scraper: Extracts job postings from the target company's careers page.

Semantic Retriever: Utilizes ChromaDB to find portfolio items that semantically match the job descriptions.

Prompt Constructor: Builds prompts by combining job details and portfolio links.

Language Model Interface: Sends prompts to LLaMA 3.1 via LangChain and retrieves generated emails.

User Interface: Streamlit-based frontend where users input URLs and receive generated emails.

