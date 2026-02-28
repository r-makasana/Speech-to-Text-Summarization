"""
Speech Transcriber Module

This module handles speech-to-text transcription and text punctuation.
"""

import os
import speech_recognition as sr
from transformers import pipeline
import torch

class SpeechTranscriber:
    def __init__(self):
        """Initialize the speech recognizer and punctuation model."""
        self.recognizer = sr.Recognizer()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device set to use {self.device}")
        
        # Initialize punctuation model
        try:
            self.punctuation_model = pipeline(
                "text2text-generation",
                model="oliverguhr/fullstop-punctuation-multilang-large",
                device=self.device
            )
        except Exception as e:
            print(f"Warning: Could not load punctuation model: {e}")
            self.punctuation_model = None

    def add_punctuation(self, text):
        """Add punctuation to the transcribed text."""
        if not self.punctuation_model:
            return text
            
        try:
            # Split text into smaller chunks to avoid memory issues
            max_length = 512
            words = text.split()
            chunks = []
            current_chunk = []
            
            for word in words:
                current_chunk.append(word)
                if len(" ".join(current_chunk)) > max_length:
                    chunks.append(" ".join(current_chunk[:-1]))
                    current_chunk = [current_chunk[-1]]
            
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            # Process each chunk
            punctuated_chunks = []
            for chunk in chunks:
                result = self.punctuation_model(chunk, max_length=512)[0]['generated_text']
                punctuated_chunks.append(result)
            
            return " ".join(punctuated_chunks)
            
        except Exception as e:
            print(f"Warning: Error during punctuation: {e}")
            return text

    def transcribe_file(self, audio_path):
        """
        Transcribe an audio file to text.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            tuple: (transcribed text, path to transcript file)
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        try:
            # Create transcripts directory
            output_dir = os.path.join(os.path.dirname(audio_path), 'transcripts')
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate output path
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            transcript_path = os.path.join(output_dir, f"{base_name}_transcript.txt")
            
            # Perform transcription
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio)
            
            # Add punctuation
            if text:
                text = self.add_punctuation(text)
                
                # Save transcript
                with open(transcript_path, 'w') as f:
                    f.write(text)
                    
                return text, transcript_path
                
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None, None