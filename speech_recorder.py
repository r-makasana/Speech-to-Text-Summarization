"""
Speech Recorder Module

This module handles audio recording using PyAudio and saves the recorded audio
to a WAV file for further processing.
"""

import os
import pyaudio
import wave
import time
from datetime import datetime

class SpeechRecorder:
    """A class to record audio from microphone and save it as a WAV file."""
    
    def __init__(self, output_dir="recordings"):
        """
        Initialize the SpeechRecorder with recording parameters.
        
        Args:
            output_dir (str): Directory to save recorded audio files
        """
        # Recording parameters
        self.chunk = 1024  # Record in chunks of 1024 samples
        self.sample_format = pyaudio.paInt16  # 16 bits per sample
        self.channels = 1  # Mono recording for better speech recognition
        self.fs = 44100  # Sample rate: 44.1 kHz
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def record(self, duration=60, filename=None):
        """
        Record audio for the specified duration.
        
        Args:
            duration (int): Recording duration in seconds
            filename (str, optional): Output filename. If None, generates a timestamp-based name.
            
        Returns:
            str: Path to the saved WAV file
        """
        if filename is None:
            # Generate timestamp-based filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.wav"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Create an interface to PortAudio
        p = pyaudio.PyAudio()
        
        print(f"Recording for {duration} seconds...")
        
        # Open stream
        stream = p.open(
            format=self.sample_format,
            channels=self.channels,
            rate=self.fs,
            frames_per_buffer=self.chunk,
            input=True
        )
        
        frames = []  # Initialize array to store frames
        
        # Record for the specified duration
        for _ in range(0, int(self.fs / self.chunk * duration)):
            data = stream.read(self.chunk)
            frames.append(data)
            
            # Print progress every 5 seconds
            if _ % int(self.fs / self.chunk * 5) == 0 and _ > 0:
                elapsed = _ // int(self.fs / self.chunk)
                print(f"Recorded {elapsed} seconds...")
        
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        
        # Terminate the PortAudio interface
        p.terminate()
        
        print(f"Recording complete! Saving to {filepath}")
        
        # Save the recorded data as a WAV file
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(p.get_sample_size(self.sample_format))
            wf.setframerate(self.fs)
            wf.writeframes(b''.join(frames))
        
        return filepath


if __name__ == "__main__":
    # Demo usage
    recorder = SpeechRecorder()
    recording_path = recorder.record(duration=10)
    print(f"Audio saved to: {recording_path}")