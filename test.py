import os, random, argparse
import soundfile as sf

def get_text():
    # potato_options=["Картофель","potato","potato"]
    retstr = ''
    lines=''
    while True:
        line = input("Enter text (789 to stop): ")
        if line != "789":
            lines += line + ' '
        else:
            break
    if len(lines)>1:retstr = lines
    else:retstr = f"""
    I love sucking big horse cock. But I also like gardening with my mother and reading the paper. Yesterday, he stuck it in me so far I thought I would break when he made me cum.
    """
    print(len(lines),len(lines)>1,f'retstr=={retstr}')
    return retstr

def get_chunk_size():
    chunkbool = input("Input desired size of chunks. \nEnter to skip\n")
    if len(chunkbool)>0:retstr=chunkbool
    else:retstr=20
    return retstr
def choose_speaker():
    #en_speaker_6 = male
    speaker = [
        "en_speaker_9"
        ,"it_speaker_9"
        ,"ja_speaker_0"
        ,"en_speaker_6"
        ,"de_speaker_3"
    ]
    listt = ''
    i = 0
    for s in speaker: 
        listt += '\n type %i for '%i + s
        i += 1
    ans = input(f"choose from the list: {listt} \n >>")
    if not ans.isdigit() or int(ans)> (len(speaker)-1):
        print(f"invalid input - defaulting to {speaker[0]}")
        return speaker[0]
    return speaker[int(ans)]

def main():
    text_prompt=get_text()
    chunk_size = int(get_chunk_size())
    print(f'chunks set to {chunk_size}')
    
    words=text_prompt.split(" ");
    chunks = [words[i:i+chunk_size] for i in range(0, len(words), chunk_size)]
    
    print("loading models...")
    from bark import SAMPLE_RATE, generate_audio,preload_models 
    from IPython.display import Audio
    preload_models(
        text_use_gpu=True,
        text_use_small=True,
        coarse_use_gpu=True,
        coarse_use_small=True,
        fine_use_gpu=True,
        fine_use_small=True,
        codec_use_gpu=True,
        force_reload=False,
    )
    
    #generate_ audio_array
    i=0
    print(f"chunks:{chunks} \n")
    speaker =choose_speaker()
    for chunk in chunks:
        text = " ".join(chunk);
        filename = f"generated_audio{i}.wav"
        print(f'text == {text}\n next file: {filename}')
        audio_array = generate_audio(text, history_prompt=f"v2/{speaker}")
        Audio(audio_array, rate=SAMPLE_RATE)
        sf.write(filename, audio_array, SAMPLE_RATE)
        i+=1

if __name__ == '__main__':
    main()