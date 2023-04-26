import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_questions(summaries):
    qs = []

    for summary in summaries:

        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Generate 5 open-ended questions: " + summary}
        ]
        )

        qs.append(completion.choices[0].message.content)
    return qs

def get_summaries(full_texts):
    pass
    # done using the CLI provided at
    # https://github.com/dmmiller612/lecture-summarizer
    # in terminal for the test samples