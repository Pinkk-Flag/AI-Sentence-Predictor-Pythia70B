# An AI sentence predictor Chatbot!
This is an AI sentence predictor. It is in the format of a CLI, where the user can provide a word or a few words, and then the Pythia-70M model will try and complete it! Keep in mind, however, it doesn't reflect any values of myself, and is for educational purposes only. It may generate incorrect and potentially harmful or offensive text so please keep that in mind. 

Feel free to change certain things such as the temperature, max_length, etc and then recompile in pyinstaller.

# How to make into an Exe
Simply using the terminal is completely fine for this app, but if you want to make it an exe, install pyinstaller, and do this command after going into the directory where this is downloaded:
``` python
pyinstaller --onefile main.py
```

# Credit:
All credit goes to the team at EleutherAI who made this possible.
Visit the specific model I used here at this link: https://huggingface.co/EleutherAI/pythia-70m
