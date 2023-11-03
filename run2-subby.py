import subprocess, os, sys
from dotenv import load_dotenv
load_dotenv()

#Venv and creation
folder_path = os.getenv("BARK_PATH")
venv_path = folder_path + os.getenv("VENV_PATH")

# callstr = [venv_path, '&&', 'python', 'test.py']
callstr = [venv_path, '&&', 'python', 'run_2.py']
subprocess.call(callstr, shell=True)
print(f"ran with: {' '.join(str(e) for e in callstr)}")

def convert(extension=".mp3"):#broken
    from pydub import AudioSegment
    input(f"Press Enter to convert the file to {extension}...")
    output_audio = AudioSegment.empty()
    # output_audio = sound.set_sample_width(2).export(filename + ".mp3", format="mp3")
    sound = AudioSegment.from_wav(filename+extension)
    sound.export(filename+extension, format=extension)

def combine(audio_files):
	audio_files = [f for f in os.listdir(folder_path) if f.startswith("generated_audio") and f.endswith(".wav")]
	# Sort the files in ascending order based on the number after "generated_audio"
	audio_files.sort(key=lambda x: int(x.replace("generated_audio", "").replace(".wav", "")) if x.replace("generated_audio", "").replace(".wav", "").isdigit() else float('inf'))
	# combine the audio files
	combined_audio = AudioSegment.empty()
	for file in audio_files:
	    sound = AudioSegment.from_wav(file)
	    combined_audio += sound

filename = "bark_generation"
extension = ".wav"
# convert(extension)

a=input(f"Press Enter to read the file \"{filename}\", \
 any 2 keys to exit instead...")
if len(a) > 1: quit()
audio_file = folder_path +"/"+ filename+extension
os.startfile(audio_file)
