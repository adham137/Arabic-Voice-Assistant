import torchaudio, torch
print(str(torchaudio.list_audio_backends()))
print(torch.cuda.is_available())

from pydub import AudioSegment
import os

# Function to convert mp3 to wav
def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")

# Directory containing mp3 files
input_directory = './AI-Voice-Assistant/wakeword/scripts/data/0'  # Update this path

# Directory where wav files will be saved
output_directory = './AI-Voice-Assistant/wakeword/scripts/data/0'  # Update this path

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Process each mp3 file in the directory
for filename in os.listdir(input_directory):
    if filename.endswith('.mp3'):
        mp3_path = os.path.join(input_directory, filename)
        wav_filename = os.path.splitext(filename)[0] + '.wav'
        wav_path = os.path.join(output_directory, wav_filename)
        
        # Convert the mp3 file to wav
        convert_mp3_to_wav(mp3_path, wav_path)
        
        # Delete the original mp3 file
        os.remove(mp3_path)
        print(f'Converted {filename} to {wav_filename} and deleted the original mp3 file.')

print('Conversion completed and old mp3 files deleted.')
