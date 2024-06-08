
from transformers import GPTNeoXForCausalLM, AutoTokenizer

model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step3000",
  cache_dir="./pythia-70m-deduped/step3000",
)

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step3000",
  cache_dir="./pythia-70m-deduped/step3000",
)
print("Welcome to my program! This is essentially a 'word predictor'. It tries to predict what the next word is based off of the massive data set it was given. Give it a small sentence like 'next you have to...' or anyhting like that, and it should give a funny or informative answer! By the way, you can edit the max length and temperature (creativity) in the model.generate line of code. Say 'exit' to exit the program. \n \n All credit due to the developers who made pythia-70m LLM, a lot of respect to them.")
while True:

    theinput = str(input("Enter the starting word/s: "))

    if theinput.lower() == "exit":
        print("Exiting...")
        break
    else:
        inputs = tokenizer(theinput, return_tensors="pt")
        tokens = model.generate(**inputs, max_length=20, temperature=1.2)
        result = tokenizer.decode(tokens[0])
        print("Chatbot: " + result)
