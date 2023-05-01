import sys
import os
import math
import numpy as np
import time

from tkinter import *
from tkinter.filedialog import *

import scipy.io.wavfile
from scipy import signal

from pydub import AudioSegment
from multiprocessing import Process
from pydub.playback import play
import soundfile as sf
#from playsound import playsound

from matplotlib import pyplot as plt


class BeepGenerator:

    def __init__(self):
        # Audio will contain a long list of samples (i.e. floating point numbers describing the
        # waveform).  If you were working with a very long sound you'd want to stream this to
        # disk instead of buffering it all in memory list this.  But most sounds will fit in 
        # memory.
        self.audio = []
        self.sample_rate = 44100.0

    def playing_audio():
        song = AudioSegment.from_wav(file_name)
        play(song)

    def showing_audiotrack():
        print(file_name)
        # We use a variable previousTime to store the time when a plot update is made
        # and to then compute the time taken to update the plot of the audio data.
        previousTime = time.time()

        # Turning the interactive mode on
        plt.ion()

        # Each time we go through a number of samples in the audio data that corresponds to one second of audio,
        # we increase spentTime by one (1 second).
        spentTime = 0

        # Let's the define the update periodicity
        updatePeriodicity = 2  # expressed in seconds

        # Plotting the audio data and updating the plot
        for i in range(n):
            # Each time we read one second of audio data, we increase spentTime :
            if i // Fs != (i-1) // Fs:
                spentTime += 1
            # We update the plot every updatePeriodicity seconds
            if spentTime == updatePeriodicity:
                # Clear the previous plot
                plt.clf()
                # Plot the audio data
                plt.plot(time_axis, sound_axis)
                # Plot a red line to keep track of the progression
                plt.axvline(x=i / Fs, color='r')
                plt.xlabel("Time (s)")
                plt.ylabel("Audio")
                plt.show()  # shows the plot
                plt.pause(updatePeriodicity-(time.time()-previousTime))
                # a forced pause to synchronize the audio being played with the audio track being displayed
                previousTime = time.time()
                spentTime = 0

    def showing_audio():
        #===============
        # show signal wave
        #===============
        # Retrieve the data from the wav file
        data, samplerate = sf.read(file_name)

        n = len(data)  # the length of the arrays contained in data
        Fs = samplerate  # the sample rate

        # Working with stereo audio, there are two channels in the audio data.
        # Let's retrieve each channel seperately:
        ch1 = data.transpose()

        # x-axis and y-axis to plot the audio data
        time_axis = np.linspace(0, n / Fs, n, endpoint=False)
        sound_axis = ch1 #we only focus on the first channel here

        # You can run the two lines below to plot the audio data contained in the audio file
        plt.plot(time_axis, sound_axis)
        plt.show()

    def append_silence(self, duration_milliseconds=2000):
        """
        Adding silence is easy - we add zeros to the end of our array
        """
        num_samples = duration_milliseconds * (self.sample_rate / 1000.0)

        for x in range(int(num_samples)):
            self.audio.append(0.7)

        for x in range(int(num_samples)):
            self.audio.append(0.1)

        return

    def append_sinewave(
            self,
            freq=440.0,
            duration_milliseconds=500,
            volume=1.0):
        """
        The sine wave generated here is the standard beep.  If you want something
        more aggressive you could try a square or saw tooth waveform.   Though there
        are some rather complicated issues with making high quality square and
        sawtooth waves... which we won't address here :) 
        """

        num_samples = duration_milliseconds * (self.sample_rate / 1000.0)

        x = np.array([i for i in range(int(num_samples))])

        sine_wave = volume * np.sin(2 * np.pi * freq * (x / self.sample_rate))

        self.audio.append(list(sine_wave))
        return
   
    """     
    def append_sinewaves(
            self,
            datas=['pilot',1,0,0,1]):

        for data in zip(datas):
            if data[0] == 'pilot':
                sine_wave = 500 * pilot_wav # répète 500 fois le signal pilote
            elif data[0] == 1:
                sine_wave += one_wav
            else:
                sine_wave += zero_wav

        self.audio.extend(list(sine_wave))

        return 
    """

    def append_sinewaves(
            self,
            datas,
            sine_wave,
            end=True
            ):
        for data in datas:
            if data == '1':
                sine_wave += one_wav
            else:
                sine_wave += zero_wav

        if end:
            # End byte (01)
            sine_wave += zero_wav
            sine_wave += one_wav
            #sine_wave += ['END 01']

        self.audio.extend(list(sine_wave))

        return

    def save_wav(self, file_name):
        # Open up a wav file
        # wav params

        # 44100 is the industry standard sample rate - CD quality.  If you need to
        # save on file size you can adjust it downwards. The standard for low quality
        # is 8000 or 8kHz.

        # WAV files here are using short, 16 bit, signed integers for the 
        # sample size.  So we multiply the floating point data we have by 32767, the
        # maximum value for a short integer.  NOTE: It is theoretically possible to
        # use the floating point -1.0 to 1.0 data directly in a WAV file but not
        # obvious how to do that using the wave module in python.

        # ligne ci-dessous ne fonctionne pas après l'écriture du premier bloc
        #self.audio = np.array(self.audio).astype(np.float32)
        scipy.io.wavfile.write(file_name, int(self.sample_rate), np.array(self.audio))


if __name__ == "__main__":
    if len(sys.argv) == 2:
        file_in = sys.argv[1]
    else:
        file_in = "env/T.bas"

    # Création de la fenêtre d'interface
    '''
    fenetre = Tk()
    label = Label(fenetre, text="Nom du fichier", bg="yellow")
    label.pack()

    value = StringVar() 
    entree = Entry(fenetre, textvariable=value, width=30)
    entree.pack()

    fenetre.mainloop()
    '''

    file_in = askopenfilename(title="Ouvrir un fichier",filetypes=[('bas files','.bas'),('bin files','.bin') ,('all files','.*')])
    print(file_in)


    file = open(file_in,"rb")
    fullpath = file_in.split('.')
    extent = fullpath[len(fullpath) - 1]

    file_elements = file_in.split('/')
    file_name = "env/" + file_elements[len(file_elements) - 1] + ".wav"

    filesizebytes = os.path.getsize(file_in)

    #"""
    up_long = [0.68,0.7,0.69,0.7,0.68,0.7,0.69,0.7,0.69,0.7,0.68,0.7,0.69,0.7,0.68,0.7,0.69,0.5]
    up_short = [0.68,0.7,0.69,0.7,0.68,0.7,0.69,0.7,0.68]
    down_long = [-0.68,-0.7,-0.69,-0.7,-0.68,-0.7,-0.69,-0.7,-0.68,-0.7,-0.69,-0.7,-0.68,-0.7,-0.69,-0.7,-0.68,0.3]
    down_short = [-0.68,-0.7,-0.69,-0.7,-0.68,-0.7,-0.69,-0.7,-0.68]
    #"""

    """
    up_long = ["UL","UL"]
    up_short = ["US"]
    down_long = ["DL", "DL"]
    down_short = ["DS"]
    """

    """
    up_long = [0.7,0.7]
    up_short = [0.1]
    down_long = [-0.7,-0.7]
    down_short = [-0.1]
    """

    pilot_wav = up_long + down_short + up_short + down_long
    one_wav = up_short + down_short + up_short + down_short
    zero_wav = up_long + down_long
    block_end = zero_wav + one_wav + (9 * zero_wav)

    """
    pilot_wav = ["PILOT"]
    one_wav = ["ONE"]
    zero_wav = ["ZERO"]
    block_end = ["BLKEND"]
    """

    """
    pilot_wav = [1,1,1]
    one_wav = [1]
    zero_wav = [0]
    block_end = [0,0,0]
    """
    
    global sine_wave

    bg = BeepGenerator()
    zeroes = 0 # Permet de savoir si on est en fin de bloc
    bytes_read = 0
    bloc_number = 1
    no_title = False
    sine_wave = 1000 * pilot_wav # répète 100 fois le signal pilote
   
    begin = '1'
    bg.append_sinewaves(datas=begin,sine_wave=sine_wave, end=False) # Indique le début des données (Begin 1)

    # pour test
    #for x in range(200):
    #    bg.audio.append(0)

    #bg.append_silence()

    byte = file.read(1)
    title_found = False # Pour gérer les fichiers sans titre
    while byte:
        bytes_read += 1
        binary = f'{byte[0]:0>8b}'
        print(binary , '=DEC=>' , byte[0] , '=HEX=>' , hex(int(binary, 2)))

        if int(binary) == 0:
            zeroes += 1
        else:
            zeroes = 0

        bg.append_sinewaves(datas=binary,sine_wave=[], end= (zeroes < 9 and extent == 'bas') or (zeroes < 11 and extent == 'bin'))

        # condition 1 fichier avec titre, condition 2 pas de titre
        if no_title == False:
            no_title = title_found == False and bytes_read == 2 and int(binary) == 0 # pas de titre dans le fichier
        if zeroes == 9:
            print("fin du bloc")
            if no_title == True and title_found == False:
                print("Fichier sans titre")
                bg.audio.extend(list(zero_wav + one_wav + (10 * zero_wav)))
            else:
                if title_found == False:
                    print("Fichier avec titre")

                bg.audio.extend(list(block_end))

            if (title_found == False):
                print ("Bloc titre généré")

            title_found = True # Toujours vrai dès que le premier bloc est généré
            zeroes = 0

            if filesizebytes > (bytes_read + 1): # Pas la fin du fichier -1
                bg.append_silence(duration_milliseconds= 700 if bloc_number==1 else 3500 if extent == 'bas' else 5000)
                # Début du bloc suivant
                bloc_number +=1
                sine_wave = 1000 * pilot_wav # répète 100 fois le signal pilote
                begin = '1'
                bg.append_sinewaves(datas=begin,sine_wave=sine_wave, end=False) # Indique le début des données (Begin 1)

        byte = file.read(1)

    bg.append_silence(duration_milliseconds= 200)
    bg.save_wav(file_name)

    file.close()
    print("Nombre de bloc créés : " , bloc_number)
    print ("Nom du fichier généré : ", file_name)

    #p1 = Process(target=playing_audio, args=())
    #p1.sine_wavestart()
    
    #p2 = Process(target=showing_audio, args=())
    #p2.start()

    #p1.join()
    #p2.join()

    #playsound(file_name)

"""     x = np.linspace(0, 10, 1000)
    y = np.array([1 if math.floor(2 * t) % 2 == 0 else 0 for t in x])

    plt.plot(x,y)
    plt.show() """