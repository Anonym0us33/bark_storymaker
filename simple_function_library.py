import subprocess, os, re, inflect
from dotenv import load_dotenv
load_dotenv()

class Text():
    '''
        handles formatting strings for TTS
        uses subprocess, os, re, inflect, dotenv.load_dotenv
        some class methods ask for user input from the cli
        not all methods are suitable for automatic tts scripts
        flow: 
        menu> input==[1|2] >
        {1:runner()|2:runnercmd()}>
        caller(string)>
        numbers_to_words(string)>
        number_to_word(match,p)>
        print "ok" & return null
        any of these methods can be called, but the flow will
        continue from there until a string is returned
    '''

    #unpythonic af but idgaf
    file = "readinput.txt" 
    result = ''

    def __init__(self, filename="readinput.txt"):
    # def __init__(self, string=file):
        self.file = filename
        self.result = os.getenv("TEST_TEXT3")
    
    def set_result(self, string): #umm?
        self.result = string
    def get_result(self): #umm?
        return self.result

    def set_filename(self, string):
        try:
            file = string
            return 1
        except Exception as e:
            return 0

    def numbers_to_words(self, string):
        # convert numbers to words
        # coz model cant speak them.
        # uses "re"/"inflect"

        p = inflect.engine()
        # Replace all numbers and '%' symbol using a regex pattern
        # result = re.sub(r'\d+|%', number_to_word, string)
        result = re.sub(r'\d+|%', lambda match: self.number_to_word(match, p), string)
        return result
    def number_to_word(self, match,p):
        # lambda function for numbers_to_words
        # this can probably be combined 
        #but for the  fact it uses lambda
        #i have no idea what this matching library does
        # but it seems to break on decimals like 3.1
        #this is probably user error
        number_dict = {
            '%': 'percent '
            ,'&': ' and '
            ,'@': 'at '
            ,'#': 'hashtag '
            ,'+': 'plus'
            # ,'-': 'hyphen'
            ,'_': ' underscore '
            ,'lvl': 'level'
            ,'Lvl': 'level'
            ,'[': ''
            ,'] ': ' '
            ,'].': '.'
            ,']\'': '\''
        }
        if match.group().isdigit():
            return p.number_to_words(match.group())
        return number_dict[match.group()]

    def menu(self):
        print('starting')
        userin = input(f"1. input text \n2. Use file '{self.file}' \n>>")
        if userin =='2':self.runner()
        elif userin =='1':self.runnercmd()
        else:print("no choice made... \nexiting")

    def runner(self):
        with open(self.file, "r") as file:
            contents = file.read()
        text = ""
        for line in contents:
          text+= line + " "
        self.caller(contents)

    def runnercmd(self):
        lines=''
        # while True:
        #     line = input("Enter text (789456 to stop): ")
        #     if line != "789456":
        #         lines += line + ' '
        #     else:
        #         break
        while True:#infinite loops kinda suck
            line = input("Enter text (789 to stop): ")
            if line != "789":
                lines += line + ' '
            else:
                break
        if len(lines)>1:self.caller(lines)

    def caller(self, contents):
        #depreciated, broken
        self.result = self.numbers_to_words(contents)
        # tts = TTS()
        # chunks = tts.chunk_text(renumbered_text)
        # i=1
        # for chunk in chunks:
        #     model = tts.make_model(
        #         model_id = 'v3_en',#
        #         device=0, #gpu0
        #         )
        #     tts.yet_another_save_audio(model
        #         ,text = chunk
        #         ,iteration=i
        #         ,sample_rate = 48000
        #         ,speaker="en_11"
        #         )
        #     del model
        #     i += 1 
        print("read_text() formatting finished")
        # return chunks

##################################
#            BREAK               #
##################################

class TTS():
    '''
    model downloads
    /models/tts/en/

        v1_lj_16000.jit
        v1_lj_8000.jit
        v2_lj.pt
        v3_en.pt
        v3_en_indic.pt
    ========================================================
        = this script takes the arguments,
        = -activates the venv
        = -chunks the text to 100 chars or <option> (hardcoded)
        = -turns them into audio objects
        = -saves the audio file as output<x>.wav
        = 
        = takes 0 1 or 2 arguments
        = args: textin speaker
        = eg:"I like turtles" en_99
        = >>> options for speaker:
        =    en_0 en_4 en_10 en_11 en_14 en_21 en_24 en_25 
        =    en_26 en_28 en_39 (monotone) en_45 (sexy robot)
        =    en_48 en_49 
        =    en_51 (<-- BEST 2 -->) en_99
        =    (male)
        =    en_15 en_19 en_30 
    ========================================================
    '''
    import torch, os, sys, argparse
    import gc #garbage collect
    import soundfile, torchaudio
    from pprint import pprint
    from omegaconf import OmegaConf

    ###########
    def runlocal(self, 
        local_file = 'model.pt',
        language = 'en',
        model_id = 'v3_en',
        example_text = 'catch a fish and dick it. I love sucking big black cock. My name is patchouli knowledge',
        sample_rate = 48000,
        speaker='en_99',
        device='cpu',
        threads=4,
        iteration=0
        ):
        print(f"Running runlocal with example_text='{example_text}', speaker='{speaker}', iteration={iteration}")
        device = torch.device(device)
        torch.set_num_threads(threads)

        if not os.path.isfile(local_file):
            torch.hub.download_url_to_file('https://models.silero.ai/models/tts/en/v3_en.pt',
                       local_file)  
        model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
        print(f'{device}, local_file:{local_file}')
        model.to(device)

        audio_paths = model.save_wav(text=example_text,
                                 speaker=speaker,
                                 sample_rate=sample_rate,
                                audio_path=f'output{iteration}.wav' )

        print(f"{audio_paths}") 

    def yet_another_save_audio(model,
        text = 'catch a fish and dick it. I love sucking big black cock. My name is patchouli knowledge',
        iteration=0,
        sample_rate = 48000,
        speaker='en_99',
        ):
        print(f"Running save_with_model with \nexample_text='{text}', \nspeaker='{speaker}', \niteration={iteration}")
        try:
            audio_path=f'output{iteration}.wav'
            audio = [1,1,1]#torch matrix
            torchaudio.save(#no return value
                audio_path,
                audio.unsqueeze(0),
                sample_rate
                )
            print(f"{audio_path} saved.")
        except Exception as e:
            print(f"Error occurred while saving audio file: {str(e)}")
            return 0
        return 1

    def save_with_model(self, model,
        text = 'catch a fish and dick it. I love sucking big black cock. My name is patchouli knowledge',
        iteration=0,
        sample_rate = 48000,
        speaker='en_99',
        ):
        print(f"Running save_with_model with \nexample_text='{text}', \nspeaker='{speaker}', \niteration={iteration}")
        try:
            audio_path=f'output{iteration}.wav'
            audio_paths = model.save_wav(text=text,
                                 speaker=speaker,
                                 sample_rate=sample_rate,
                                audio_path=audio_path )

            print(f"{audio_paths} object type:",type(audio_paths))


            #only 1 of these is needed.
            #may need to del model instead
            # gc.collect()
            # audio_paths.close()
            del audio_paths
        except Exception as e:
            print(f"Error occurred while saving audio file: {str(e)}")
        return 1

    def chunk_text(self, text, n=100):
        words = text.split()
        return [' '.join(words[i:i+n]) for i in range(0, len(words), n)]

def tts_caller():
    #depreciated

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--venv', action='store_true', help='run venv')
    parser.add_argument('--text', type=str, default='no text selected', help='input text')
    parser.add_argument('--speaker', type=str, default='en_99', help='speaker name')
    args = parser.parse_args()

    t = TTS()

    # Chunk the input text and run TTS on each chunk
    chunks = t.chunk_text(args.text)
    print(f"chunks\n{chunks}\n args.venv\n{args.venv}")
    i = 1
    # for chunk in chunks:
    #     t.runlocal(example_text=chunk, speaker=args.speaker, iteration=i)
    #     i += 1
    del t

class number_processor():
    #THIS CLASS IS BROKEN BEYOND BELIEF. NO HUMAN WILL EVER KNOW WHAT IT DOES
    '''
    This function converts the given numbers to words while handling various formats, such as numbers with commas and scientific notation. The output for the sample numbers provided will be:

    ```
    1 : one
    11 : eleven
    120 : one hundred twenty
    3000 : three thousand
    1000001 : one million one
    100000 : one hundred thousand
    43201.52 : forty three thousand two hundred one point five two
    ```

    Please note that the function assumes the input numbers are valid numeric values and does not handle edge cases for extremely large numbers or non-numeric inputs.
    '''

    def number_to_words(self, text):
        # Define word mappings for numbers up to 19
        num_words = {
            0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 
            7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven', 12: 'twelve', 
            13: 'thirteen', 14: 'fourteen', 15: 'fifteen', 16: 'sixteen', 17: 'seventeen', 
            18: 'eighteen', 19: 'nineteen'
        }
        
        # Define word mappings for tens
        tens_words = {
            2: 'twenty', 3: 'thirty', 4: 'forty', 5: 'fifty', 6: 'sixty', 7: 'seventy', 
            8: 'eighty', 9: 'ninety'
        }
        
        # Define word mappings for scales (thousand, million, etc.)
        scales = ['', 'thousand', 'million', 'billion', 'trillion']
        
        # Convert number to string, remove commas, and split into digits
        number_str = str(text).replace(',', '')
        number_digits = list(number_str)
        
        # Iterate over the digits in reverse order
        words = []
        while number_digits:
            temp_words = []
            
            # Convert three digits at a time
            for _ in range(min(len(number_digits), 3)):
                temp_words.append(number_digits.pop())
            
            # Convert three digits to words
            temp_words.reverse()
            temp_words = [int(d) for d in temp_words if d.isdigit()]
            
            # Handle the hundreds place
            if len(temp_words) == 3:
                if temp_words[0] != 0:
                    words.append(num_words[temp_words[0]])
                    words.append('hundred')
                temp_words = temp_words[1:]
            
            # Handle the tens and ones places
            if len(temp_words) == 2:
                if temp_words[0] == 1:
                    words.append(num_words[int(''.join(map(str, temp_words)))])
                else:
                    if temp_words[0] != 0:
                        words.append(tens_words[temp_words[0]])
                    if temp_words[1] != 0:
                        words.append(num_words[temp_words[1]])
            
            # Handle the last digit (should be in the ones place)
            if len(temp_words) == 1 and temp_words[0] != 0:
                words.append(num_words[temp_words[0]])
            
            # print('after:',number_digits, 'len:',len(number_digits))
            # Add scale word if needed
            print(number_digits,len(number_digits))
            if len(number_digits) > 0:
                words.append(scales[(len(number_digits) - 1) // 3])
            # number_to_words("i have 999 fakes 1234 pies and twenty 1% fat")
            # 789456
        # Return the converted words joined by spaces
        toret= ' '.join(words[::-1])
        print(f'words: {words}')
        return toret

def num_caller():
    '''
    (note: broken!)
    Test the function with sample numbers
    '''    
    import os
    from dotenv import load_dotenv
    load_dotenv()
    test_str = os.getenv("TEST_TEXT3")
    dts = number_processor()
    numbers = [1, 11, 120, 3000, 1000001, 100000, 43201.52]
    for number in numbers:
        words = number_to_words(number)
        print(number, ':', words)
    print(f"test_str:{test_str} len= {len(test_str)} type:{type(test_str)}")
    dts.number_to_words(test_str)