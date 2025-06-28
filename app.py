from openai import OpenAI
import gradio as gr
import pandas as pd
import tiktoken

client = None

PILLARS = [
    "Personal Readiness",
    "Financial Readiness",
    "Business Idea Validation",
    "Legal & Structural Setup",
    "Operations & Logistics",
    "Marketing & Sales Strategy",
    "Mindset & Philosophy",
    "First 30 Days Execution Plan",
    "Psychographic Analysis",
    "Footfall / Traffic Analysis",
    "Competition Analysis",
    "Distribution & Fulfillment",
    "Team & Outsourcing",
    "Tech & Automation Stack",
    "Branding & Storytelling",
    "Traction & Feedback Loops",
    "Exit or Pivot Strategy",
    "Skill–Scope Fit Assessment",
    "Field Exposure / Apprenticeship"
]

answers = {}
current_pillar_index = 0
business_idea = ""
sub_answers = []

def num_tokens(text, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def dynamic_chunk_dict(answers, business_idea, max_tokens=7000):
    chunks = []
    current_chunk = {}
    current_tokens = num_tokens(business_idea) + 200  # base prompt buffer

    for pillar, response in answers.items():
        entry = f"\n{pillar}: {response}"
        tokens = num_tokens(entry)
        if current_tokens + tokens > max_tokens:
            chunks.append(current_chunk)
            current_chunk = {}
            current_tokens = num_tokens(business_idea) + 200  # reset
        current_chunk[pillar] = response
        current_tokens += tokens

    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def evaluate_with_gpt_in_chunks_dynamic(business_idea, answers):
    all_scores = []
    
    chunks = dynamic_chunk_dict(answers, business_idea)
    for i, chunk in enumerate(chunks):
        prompt = f"""
You are a startup advisor evaluating a business idea across 19 key pillars.
Each pillar has a user-submitted answer.
Your job:
1. Score each pillar from 1 to 5 strictly based on the user's actual answer.
2. If the answer is vague, missing, or shows little knowledge, give a low score and explain why.
3. If the answer is strong, clear, and shows preparedness, give a higher score and justify it.
4. Do NOT guess or be optimistic. Evaluate only what's written.
Business Idea: {business_idea}
Answers:
"""
        for pillar, response in chunk.items():
            prompt += f"\n{pillar}: {response}"

        res = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a brutally honest but kind startup evaluator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        all_scores.append(res.choices[0].message.content.strip())

    # Final summarizer
    summary_prompt = f"""
You are an expert startup advisor.
Below is a business idea followed by GPT-generated pillar-wise evaluations (scores and explanations).
Business Idea: {business_idea}
Pillar Evaluations:
""" + "\n\n".join(all_scores) + """
Now, do the following:
1. Summarize the top 3 strengths of the business based on the answers and scores.
2. Summarize the top 3 weaknesses or risks.
3. Suggest 3 practical action steps to improve the business readiness.
Do not repeat all pillar evaluations. Just summarize.
"""

    summary = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a thoughtful and clear startup coach."},
            {"role": "user", "content": summary_prompt}
        ],
        temperature=0.3
    )

    return "\n\n".join(all_scores) + "\n\n=== Summary ===\n" + summary.choices[0].message.content.strip()

def ask_question_for_pillar(index):
    prev_answers = "\n".join([f"{k}: {v}" for k, v in answers.items()])
    prompt = f"""
The user is starting a business: "{business_idea}"
You are evaluating this pillar: "{PILLARS[index]}"
Here are all previous answers they’ve given:
{prev_answers}
Now, ask 3 to 4 smart, specific, and non-redundant questions about this pillar.
Focus only on what hasn’t already been asked or answered. Build on what’s already known.
Each question should be high-leverage — something that reveals practical readiness, mindset clarity, or gaps to fix.
Format the output as a numbered list, with no extra explanation.
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a smart startup readiness coach."},
            {"role": "user", "content": prompt}
        ]
    )
    return f"{index + 1}. {PILLARS[index]}\n{response.choices[0].message.content}"

def start_session(api_key, idea):
    global business_idea, answers, current_pillar_index, sub_answers, client
    business_idea = idea.strip()
    answers = {}
    sub_answers = []
    current_pillar_index = 0
    client = OpenAI(api_key=api_key.strip())
    return ask_question_for_pillar(current_pillar_index)

def collect_answer(user_input):
    global current_pillar_index, sub_answers
    sub_answers.append(user_input.strip())
    answers[PILLARS[current_pillar_index]] = "\n".join(sub_answers)
    current_pillar_index += 1
    sub_answers.clear()

    if current_pillar_index >= len(PILLARS):
        preview = "\n".join([f"{p}: {a}" for p, a in answers.items()])
        return f"✅ All answers collected!\n\n{preview}\n\nClick 'Generate Final Report' to score your readiness.", ""

    return ask_question_for_pillar(current_pillar_index), ""

def restart_session():
    global answers, sub_answers, current_pillar_index, business_idea
    answers = {}
    sub_answers = []
    current_pillar_index = 0
    business_idea = ""
    return "", "", "", "", ""

def export_csv():
    df = pd.DataFrame(list(answers.items()), columns=["Pillar", "Answer"])
    path = "/tmp/startup_answers.csv"
    df.to_csv(path, index=False)
    return path

with gr.Blocks() as app:
    gr.Markdown("## 🌱 Startup Readiness Coach (19 Pillars)")

    with gr.Row():
        api_key = gr.Textbox(label="🔑 OpenAI API Key", type="password", lines=1)

    with gr.Row():
        idea_input = gr.Textbox(label="💡 Describe your business idea", lines=10, scale=2)
        start_btn = gr.Button("Start Assessment")

    with gr.Row():
        question_box = gr.Textbox(label="📌 Current Pillar Question", interactive=False, lines=4)

    with gr.Row():
        answer_box = gr.Textbox(label="✍️ Your Answer", lines=4)
        next_btn = gr.Button("Next")

    with gr.Row():
        generate_btn = gr.Button("📊 Generate Final Report")
        restart_btn = gr.Button("🔄 Restart")
        csv_btn = gr.Button("⬇️ Download Answers as CSV")

    result_output = gr.Textbox(label="📈 Final Report", lines=25)
    csv_output = gr.File(label="📁 CSV File")

    start_btn.click(start_session, inputs=[api_key, idea_input], outputs=question_box)
    next_btn.click(collect_answer, inputs=answer_box, outputs=[question_box, answer_box])
    generate_btn.click(lambda: evaluate_with_gpt_in_chunks_dynamic(business_idea, answers), outputs=result_output)
    restart_btn.click(restart_session, outputs=[question_box, answer_box, result_output, idea_input, api_key])
    csv_btn.click(export_csv, outputs=csv_output)

app.launch(share=True, debug=True)