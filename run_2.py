"""
execution order
imports
__main__
set env
main
    flow: 
        t.menu>
            [runner()|runnercmd()]>
            caller(string)>
            numbers_to_words(string)>
            number_to_word(match,p)>
            print "ok" & return null
        strip t.result
        input()>
            chunk size
            speaker
        load model
        chunk | sentence
        combine audio
        write


"""
import os, random, argparse, nltk
import numpy as np
import soundfile as sf
import simple_function_library as fl 
from dotenv import load_dotenv
from bark import SAMPLE_RATE
load_dotenv()
from scipy.io.wavfile import write as write_wav
from bark.generation import (
    generate_text_semantic,
)
from bark.api import semantic_to_waveform

txtfile = os.getenv("TEXT_FILE")    
t = fl.Text(txtfile) if len(txtfile)>1 else fl.Text("log.txt")
output_dir = os.getenv("OUTPUT_PATH")    
CHUNK_SAVE_FREQUENCY = os.getenv("CHUNK_SAVE_FREQUENCY")    
SENTENCE_SAVE_FREQUENCY = os.getenv("SENTENCE_SAVE_FREQUENCY")    
TEST_SAVE_FREQUENCY = os.getenv("TEST_SAVE_FREQUENCY")

def get_text():
    #depreciated
    #uses fl.TTS | load_dotenv
    
    potato_options= os.getenv("POTATOS").split(',')

    # (retstr, lines)=('','')
    while True:#infinite loops kinda suck
        line = input("Enter text (789 to stop): ")
        if line != "789":
            lines += line + ' '
        else:
            break
    if len(lines)>1:retstr = lines
    else:retstr = os.getenv("TEST_TEXT") 
    print(len(lines),len(lines)>1,f'retstr=={retstr} \n len(lines){len(lines)} \nlen(lines)>1{len(lines)>1,}')
    return retstr

def get_chunk_size():
    chunkbool = input("Input desired size of chunks.(iirc 50 per 8GB VRAM)\nEnter to skip\n")
    if len(chunkbool)>0:retstr=chunkbool
    else:retstr=30
    return retstr

def choose_speaker():
    #using the hardcoded list, displays each
    #speaker with the list index and
    #checks if user input is a valid int
    #and uses that int to choose from the list

    #en_speaker_6 = male
    speaker = [
        "en_speaker_9"
        ,"it_speaker_9"
        ,"ja_speaker_0"
        ,"en_speaker_6"
        ,"de_speaker_3"
    ]
    speakers = os.getenv("BEST_SPEAKERS").split(",")
    lis = ''
    # i = 0
    # for s in speaker: 
    for i, s in enumerate(speakers):
        lis += f'\nType {i} for {s}'
        i += 1
    ans = input(f"choose from the list: {lis} \n >>")
    if not ans.isdigit() or int(ans)> (len(speakers)-1):
        print(f"invalid input - defaulting to {speakers[0]}")
        return speakers[0]
    return speakers[int(ans)]

def choose_mode():
    modestr = input("\nuse chunks? y/n \nenter defaults to sentences>>")
    if len(modestr)>0:return True
    else:return False

def main():
    '''
    flow: 
        t.menu>
            [runner()|runnercmd()]>
            caller(string)>
            numbers_to_words(string)>
            number_to_word(match,p)>
            print "ok" & return null
        strip t.result
        input()>
            chunk size
            speaker
        load model
        chunk | sentence
        combine audio
        write
    there is no way to avoid turning 1 into one without
    removing t.menu() class
    '''
    # text_prompt = get_text().replace("\n", " ").strip()
    t.menu()
    text_prompt = t.get_result().replace("\n", " ").strip()
    ChunkMode = choose_mode()
    if ChunkMode: 
        chunk_size = int(get_chunk_size())
        print(f'chunks set to {chunk_size}')
    
    # words=text_prompt.split(" "); #TODO:del 
    chosen_speaker = choose_speaker()
    # we'll use this to split into sentences
    
    print("loading models...")
    from bark import generate_audio, preload_models
    from IPython.display import Audio
    preload_models(
        text_use_gpu=True,
        text_use_small=True,
        coarse_use_gpu=True,
        coarse_use_small=True,
        fine_use_gpu=True,
        fine_use_small=False,
        codec_use_gpu=True,
        force_reload=False,
    )

    pieces = []
    # quarter second of silence
    silence = np.zeros(int(0.25 * SAMPLE_RATE))  
    #generate_ audio_array
    i,n=0,0
    if ChunkMode:
        print(f"text_prompt:{text_prompt} \n")
        words=text_prompt.split(" ");
        chunks = [words[i:i+chunk_size] for i in range(0, len(words), chunk_size)]
        print(f"chunks:{chunks} \n")
        for chunk in chunks:
            text = " ".join(chunk);
            # filename = f"generated_audio{i}.wav"
            # print(f'text == {text}\n next file: {filename}')
            audio_array = generate_audio(text, history_prompt=f"v2/{chosen_speaker}")
            # Audio(audio_array, rate=SAMPLE_RATE)
            # sf.write(filename, audio_array, SAMPLE_RATE)
            pieces += [audio_array, silence.copy()]
            i+=1
            n+=1
            if i>CHUNK_SAVE_FREQUENCY:
                write_audio(pieces,n)
                pieces = []
                i = 0
    else:
        sentences = nltk.sent_tokenize(text_prompt)
        print(f"sentences:{sentences} \n")
        for sentence  in sentences:
            # stop hallucination
            print(f'turn: {i} \n sentence  == {sentence}\n \
                with {chosen_speaker}')
            semantic_tokens = generate_text_semantic(
                sentence,
                history_prompt=chosen_speaker,
                temp=0.6,
                min_eos_p=0.05,  # this controls how likely the generation is to end
            )
            audio_array = semantic_to_waveform(semantic_tokens, history_prompt=f"v2/{chosen_speaker}",)
            pieces += [audio_array, silence.copy()]
            i+=1
            n+=1
            if i>SENTENCE_SAVE_FREQUENCY:
                write_audio(pieces,n)
                pieces = []
                i = 0
    write_audio(pieces,0,_final)

def write_audio(pieces,i=0,suffix=''):
    print(f"pieces:{len(pieces)} combining now")
    final_audio = np.concatenate(pieces)
    # Audio(final_audio, rate=SAMPLE_RATE)
    # sf.write(f"Created_audio_{i}_pieces.wav", final_audio, SAMPLE_RATE) 
    write_wav(f"{output_dir}/bark_generation{i}{suffix}.wav", SAMPLE_RATE, final_audio)

if __name__ == '__main__':
    deletme = os.getenv("CUDA_VISIBLE_DEVICES") #int 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()