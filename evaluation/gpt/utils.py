import os
from openai import OpenAI

def get_response_message(response):
    return response.choices[0].message.content

def generate_criteria_prompt(criteria_list):
    criteria = ''
    for i, c in enumerate(criteria_list):
        if c == 'Informativeness':
            criteria += f'{c}: measures the extent to which a plain language summary encapsulates essential elements such as methodologies, primary findings, and conclusions from the original scientific text. An informative summary efficiently conveys the central message of the source material, avoiding the exclusion of crucial details or the introduction of hallucinations (i.e., information present in the summary but absent in the scientific text), both of which could impair reader comprehension.\\n'
        elif c == 'Simplification':
            criteria += f'{c}: encompasses the rendering of information into a form that non-expert audiences can readily interpret and understand. This criterion prioritizes the use of simple vocabulary, casual language, and concise sentences that minimize excessive jargon and technical terminology unfamiliar to a lay audience.\\n'
        elif c == 'Coherence':
            criteria += f'{c}: pertains to the logical arrangement of a plain language summary. A coherent summary guarantees an unambiguous and steady progression of ideas, offering information in a well-ordered fashion that facilitates ease of comprehension for the reader. We conjecture that the original sentence order reflects optimal coherence.\\n'
        elif c == 'Faithfulness':
            criteria += f'{c}: denotes the extent to which the plain language summary aligns factually with the source scientific text, in terms of its findings, methods, and claims. A faithful summary should not substitute information or introduce errors, misconceptions, and inaccuracies, which can misguide the reader or misrepresent the original author\'s intent. Faithfulness emphasizes the factual alignment of the summary with the source text, while informativeness gauges the completeness and efficiency of the summary in conveying key elements.\\n'
    return criteria

def chat_gpt_no_ref_no_explain(abstract, pls_gen, model_name='gpt-4', criteria='Informativeness'):
    
    content = f"""Imagine you are a human annotator now. You will evaluate the quality of a generated plain language summary for a scientific literature abstract. Please follow these steps: \\n 
                1. Read the scientific abstract provided. \\n 
                2. Read the generated plain language summary. \\n
				3. Compared to the scientific abstract, rate the generated summary on the following criteria: {criteria} \\n
				4. Assign a score for the generated summary, rating on a scale from 0 (worst) to 100 (best). \\n
				5. You do not need to explain the reason. Only provide the score."""

    client = OpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        organization=''
        )

    messages = [
        {"role": "system", "content": content},
        {"role": "user", "content": f"Scientific abstract:{abstract};\\n Generated plain language summary:{pls_gen};\\n Score:"},
    ]

    response = client.chat.completions.create( 
        model=model_name, 
        messages=messages, 
        max_tokens=10,
        temperature=0.0
    )
    
    response_content = get_response_message(response)
    return response_content

def chat_gpt_no_ref_no_explain_all_criteria(abstract, pls_gen, model_name='gpt-4', criteria='Informativeness'):
    
    content = f"""Imagine you are a human annotator now. You will evaluate the quality of a generated plain language summary for a scientific literature abstract. Please follow these steps: \\n 
                1. Read the scientific abstract provided. \\n 
                2. Read the generated plain language summary. \\n
				3. Compared to the scientific abstract, rate the generated summary on the following criteria: {criteria} \\n
				4. Assign a score for each criteria and provide an overall score, rating on a scale from 0 (worst) to 100 (best). \\n
				5. You do not need to explain the reason. Only provide the score."""

    client = OpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        organization=''
        )

    messages = [
        {"role": "system", "content": content},
        {"role": "user", "content": f"Scientific abstract:{abstract};\\n Generated plain language summary:{pls_gen};\\n Score:"},
    ]

    response = client.chat.completions.create( 
        model=model_name, 
        messages=messages, 
        max_tokens=500,
        temperature=0.0
    )
    
    response_content = get_response_message(response)
    return response_content

def chat_gpt_no_ref_with_explain_all_criteria(abstract, pls_gen, model_name='gpt-4', criteria='Informativeness'):
    content = f"""Imagine you are a human annotator now. You will evaluate the quality of a generated plain language summary for a scientific literature abstract. Please follow these steps: \\n 
                1. Read the scientific abstract provided. \\n 
                2. Read the generated plain language summary. \\n
				3. Compared to the scientific abstract, rate the generated summary on the following criteria: {criteria} \\n
				4. Assign a score for each criteria and provide an overall score, rating on a scale from 0 (worst) to 100 (best). \\n
				5. Explain the reason for the score."""


    client = OpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        organization=''
        )

    messages = [
        {"role": "system", "content": content},
        {"role": "user", "content": f"Scientific abstract:{abstract};\\n Generated plain language summary:{pls_gen};\\n Score:"},
    ]

    ## OpenAI
    response = client.chat.completions.create( 
        model=model_name, 
        messages=messages, 
        max_tokens=500,
        temperature=0.0
    )
    
    response_content = get_response_message(response)
    return response_content

def chat_gpt_with_ref_no_explain_all_criteria(abstract, pls_gen, pls, model_name='gpt-4', criteria='Informativeness'):
    
    content = f"""Imagine you are a human annotator now. You will evaluate the quality of a generated plain language summary for a scientific literature abstract. Please follow these steps: \\n 
                1. Read the scientific abstract provided and plain language summary written by human. \\n 
                2. Read the generated plain language summary. \\n
				3. Compared to the scientific abstract and plain language summary written by human, rate the generated summary on the following criteria: {criteria} \\n
				4. Assign a score for each criteria and provide an overall score, rating on a scale from 0 (worst) to 100 (best). \\n
				5. You do not need to explain the reason. Only provide the score."""

    client = OpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        organization=''
        )

    messages = [
        {"role": "system", "content": content},
        {"role": "user", "content": f"Scientific abstract:{abstract};\\n Plain language summary: {pls};\\n Generated plain language summary: {pls_gen}; \\n Score:"},
    ]

    response = client.chat.completions.create( 
        model=model_name, 
        messages=messages, 
        max_tokens=500,
        temperature=0.0
    )
    
    response_content = get_response_message(response)
    return response_content


def chat_gpt_simplify(abstract, model_name='gpt-4'):
    content = f"Explain the text in layman's terms:"
    client = OpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        organization=''
        )

    messages = [
        {"role": "system", "content": content},
        {"role": "user", "content": f"Scientific abstract:{abstract};\\n Generated summary:"},
    ]

    response = client.chat.completions.create( 
        model=model_name, 
        messages=messages, 
        max_tokens=200,
        temperature=0.0
    )
    
    response_content = get_response_message(response)
    return response_content
