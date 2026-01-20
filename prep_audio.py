import os
from urllib import response
from elevenlabs.client import ElevenLabs

# voice_id="GNa857Cafk5vPjRLQtrs" # samuel l jackson
voice_id="0ionA5zHdT1xEq5u0jeK" # eran tzur
database_path = "database/"
auduio_path = "audio/"
os.makedirs(auduio_path, exist_ok=True)
client = ElevenLabs(api_key="sk_66e85dbf7cb2ee8413f6c97af731fa5cb8f151222ba2a7d2")

def generate_audio_file(text, name, filename):
    global auduio_path, client, voice_id
    os.makedirs(os.path.join(auduio_path, name), exist_ok=True)
    response = client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id="eleven_v3",
                output_format="mp3_44100_128"
            )
    with open(os.path.join(auduio_path, name, str(filename)+".mp3"), "wb") as audio_file:
        for chunk in response:
            if chunk:
                audio_file.write(chunk)
    print(f"Generated audio for {name}/{filename}: {text.strip()}")

# generate known names
for i, f in enumerate(os.listdir("database/")):
    if f.startswith('.'):
        continue
    if os.path.exists(os.path.join(database_path, f, "name.txt")):
        name = open(os.path.join(database_path, f, "name.txt"), "r").read().strip()
    else:  
        name = f
    generate_audio_file(f"היי {name}", f, f"name")
    if os.path.exists(os.path.join(database_path, f, "greetings.txt")):
        lines = open(os.path.join(database_path, f, "greetings.txt"), "r").readlines()
        for index, text in enumerate(lines):
            if text.strip() == "":
                continue
            generate_audio_file(text, f, f"greetings_{index}")

# generate audio files from text files
for f in [x for x in os.listdir(database_path) if x.endswith(".txt")]:
    if f.startswith('.'):
        continue
    name = os.path.splitext(f)[0]
    with open(os.path.join(database_path,f), "r", encoding="utf-8", errors='ignore') as file:
        lines = file.readlines()
        for index, text in enumerate(lines):
            if text.strip() == "":
                continue
            generate_audio_file(text, name, f"greetings_{index}")
    