# Midterm

## Background and Context
The CEO and corporate, with permission of the board, have assembled a crack data science and engineering team to take advantage of RAG, agents, and all of the latest open-source technologies emerging in the industry.  This time it's for real though.  This time, the company is aiming squarely at some Return On Investment - some ROI - on its research and development dollars.

## The Problem
You are an AI Solutions Engineer.  You've worked directly with internal stakeholders to identify a problem: `people are concerned about the implications of AI, and no one seems to understand the right way to think about building ethical and useful AI applications for enterprises.`

This is a big problem and one that is rapidly changing.  Several people you interviewed said that *they could benefit from a chatbot that helped them understand how the AI industry is evolving, especially as it relates to politics.*  Many are interested due to the current election cycle, but others feel that some of the best guidance is likely to come from the government.

## Task 1: Dealing with the Data
You identify the following important documents that, if used for context, you believe will help people understand what’s happening now:
1. 2022: [Blueprint for an AI Bill of Rights: Making Automated Systems Work for the American People](https://www.whitehouse.gov/wp-content/uploads/2022/10/Blueprint-for-an-AI-Bill-of-Rights.pdf) (PDF)
2. 2024: [National Institute of Standards and Technology (NIST) Artificial Intelligent Risk Management Framework](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf) (PDF)
Your boss, the SVP of Technology, green-lighted this project to drive the adoption of AI throughout the enterprise.  It will be a nice showpiece for the upcoming conference and the big AI initiative announcement the CEO is planning.

> Task 1: Review the two PDFs and decide how best to chunk up the data with a single strategy to optimally answer the variety of questions you expect to receive from people.
>
> *Hint: Create a list of potential questions that people are likely to ask!*

✅ Deliverables:
1. Describe the default chunking strategy that you will use.
2. Articulate a chunking strategy that you would also like to test out.
3. Describe how and why you made these decisions

## Task 2: Building a Quick End-to-End Prototype
**You are an AI Systems Engineer.**  The SVP of Technology has tasked you with spinning up a quick RAG prototype for answering questions that internal stakeholders have about AI, using the data provided in Task 1.

> Task 2: Build an end-to-end RAG application using an industry-standard open-source stack and your choice of commercial off-the-shelf models

✅ Deliverables:
1. Build a prototype and deploy to a Hugging Face Space, and create a short (< 2 min) loom video demonstrating some initial testing inputs and outputs.
2. How did you choose your stack, and why did you select each tool the way you did?

## Task 3: Creating a Golden Test Data Set
**You are an AI Evaluation & Performance Engineer.**  The AI Systems Engineer who built the initial RAG system has asked for your help and expertise in creating a "Golden Data Set."

> Task 3: Generate a synthetic test data set and baseline an initial evaluation

✅ Deliverables:
1. Assess your pipeline using the RAGAS framework including key metrics faithfulness, answer relevancy, context precision, and context recall.  Provide a table of your output results.
2. What conclusions can you draw about performance and effectiveness of your pipeline with this information?

## Task 4: Fine-Tuning Open-Source Embeddings
**You are an Machine Learning Engineer.**  The AI Evaluation and Performance Engineer has asked for your help in fine-tuning the embedding model used in their recent RAG application build.

> Task 4: Generate synthetic fine-tuning data and complete fine-tuning of the open-source embedding model

✅ Deliverables:
1. Swap out your existing embedding model for the new fine-tuned version.  Provide a link to your fine-tuned embedding model on the Hugging Face Hub.
2. How did you choose the embedding model for this application?

## Task 5: Assessing Performance
**You are the AI Evaluation & Performance Engineer.**  It's time to assess all options for this product.

> Task 5: Assess the performance of 1) the fine-tuned model, and 2) the two proposed chunking strategies

✅ Deliverables:
1. Test the fine-tuned embedding model using the RAGAS frameworks to quantify any improvements.  Provide results in a table.
2. Test the two chunking strategies using the RAGAS frameworks to quantify any improvements. Provide results in a table.
3. The AI Solutions Engineer asks you “Which one is the best to test with internal stakeholders next week, and why?”

## Task 6: Managing Your Boss and User Expectations
**You are the SVP of Technology.**  Given the work done by your team so far, you're now sitting down with the AI Solutions Engineer.  You have tasked the solutions engineer to test out the new application with at least 50 different internal stakeholders over the next month.
1. What is the story that you will give to the CEO to tell the whole company at the launch next month?
2. There appears to be important information not included in our build, for instance, the [270-day update](https://www.whitehouse.gov/briefing-room/statements-releases/2024/07/26/fact-sheet-biden-harris-administration-announces-new-ai-actions-and-receives-additional-major-voluntary-commitment-on-ai/) on the 2023 executive order on [Safe, Secure, and Trustworthy AI](https://www.whitehouse.gov/briefing-room/presidential-actions/2023/10/30/executive-order-on-the-safe-secure-and-trustworthy-development-and-use-of-artificial-intelligence/).  How might you incorporate relevant white-house briefing information into future versions?

## Your Final Submission
Please include the following in your final submission:
1. A public link to a **written report** addressing each deliverable and answering each question.
2. A public link to any relevant **GitHub repo**
3. A public link to the **final version of your application** on Hugging Face
4. A public link to your **fine-tuned embedding model** on Hugging Face