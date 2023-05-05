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

    up_long = [0.68,0.7,0.69,0.7,0.68,0.7,0.69,0.7,0.69,0.7,0.68,0.7,0.69,0.7,0.68,0.7,0.69,0.5]
    up_short = [0.68,0.7,0.69,0.7,0.68,0.7,0.69,0.7,0.68]
    down_long = [-0.68,-0.7,-0.69,-0.7,-0.68,-0.7,-0.69,-0.7,-0.68,-0.7,-0.69,-0.7,-0.68,-0.7,-0.69,-0.7,-0.68,0.3]
    down_short = [-0.68,-0.7,-0.69,-0.7,-0.68,-0.7,-0.69,-0.7,-0.68]


    pilot_wav = up_long + down_short + up_short + down_long
    one_wav = up_short + down_short + up_short + down_short
    zero_wav = up_long + down_long
    block_end = zero_wav + one_wav + (10 * zero_wav)
    
    global sine_wave

    bg = BeepGenerator()
    zeroes = 0 # Permet de savoir si on est en fin de bloc
    bytes_read = 0
    bloc_number = 1
    no_title = False

    block_length_found = False  # permet de calculer le nombre d'octets de données du fichier
    length_data = 9999 # contient le nombre d'octets de données du fichier
    low_byte = 0

    sine_wave = 1000 * pilot_wav # répète 100 fois le signal pilote
   
    begin = '1'
    bg.append_sinewaves(datas=begin,sine_wave=sine_wave, end=False) # Indique le début des données (Begin 1)

    byte = file.read(1)
    #title_found = False # Pour gérer les fichiers sans titre
    #first_block = True

    while byte:
        bytes_read += 1
        binary = f'{byte[0]:0>8b}'
        print(binary , '=DEC=>' , byte[0] , '=HEX=>' , hex(int(binary, 2)))

        # Compte le nombre de zeros consécutifs
        if int(binary) == 0:
            zeroes += 1
        else:
            zeroes = 0

        if bloc_number > 1 and block_length_found == False:
            # Calcul du nombre d'octets de données du bloc en cours
            low_byte = byte[0]
            bg.append_sinewaves(datas=binary,sine_wave=[], end= (zeroes < 9 and extent == 'bas') or (zeroes < 11 and extent == 'bin'))
            byte = file.read(1)

            bytes_read += 1
            binary = f'{byte[0]:0>8b}'
            print(binary , '=DEC=>' , byte[0] , '=HEX=>' , hex(int(binary, 2)))

            length_data = int(format(byte[0], '08b') + format(low_byte, '08b'),2) + 12 # 12 octets après le dernier octet de données du bloc
            print("Nb octets de données du bloc courant : ", length_data - 12)
            block_length_found = True

        bg.append_sinewaves(datas=binary,sine_wave=[], end= (zeroes < 9 and extent == 'bas') or (zeroes < 11 and extent == 'bin'))

        if bloc_number == 1:
            no_title = bytes_read == 2 and int(binary) == 0 # pas de titre dans le fichier

        #if zeroes == 10 or length_data == 0:
        if zeroes == 10:
            print("fin du bloc")
            bg.audio.extend(list(block_end))
            if bloc_number == 1:
                if no_title == True:
                    print("Fichier sans titre")
                else:
                    print("Fichier avec titre")

                print ("Bloc titre généré")

            zeroes = 0

            if filesizebytes > (bytes_read + 1): # Pas la fin du fichier -1
                bg.append_silence(duration_milliseconds= 700 if bloc_number==1 else 3500 if extent == 'bas' else 5000)
                # Début du bloc suivant
                bloc_number +=1
                sine_wave = 1000 * pilot_wav # répète 100 fois le signal pilote
                begin = '10000000001'
                bg.append_sinewaves(datas=begin,sine_wave=sine_wave, end=False) # Avant le début des données (Begin)
                block_length_found = False

        byte = file.read(1)

        #length_data -= 1
        #if length_data == 1:
        #    print ("la fin approche")

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