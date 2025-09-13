#opt1
pip install transformers
from transformers import pipeline

# Uses a small open-source code model online (free, no key)
gen = pipeline("text-generation", model="bigcode/tiny_starcoder_py")

result = gen("Write a Python function for factorial:\n", max_length=100)
print(result[0]["generated_text"])

#opt2
pip install gpt4free
from gpt4free import forefront

# Forefront is one of the free providers
response = forefront.Completion.create(
    model='gpt-4',
    prompt='Write a Python function for bubble sort'
)
print(response)
