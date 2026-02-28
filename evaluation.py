"""
Evaluation Module

This module handles evaluation of the speech-to-text summarization pipeline
using ROUGE metrics and other performance measures.
"""

import os
import json
import time
from rouge_score import rouge_scorer
from speech_recorder import SpeechRecorder
from speech_transcriber import SpeechTranscriber
from text_summarizer import TextSummarizer

class Evaluator:
    def __init__(self, test_data_dir="test_data"):
        """
        Initialize the evaluator.
        
        Args:
            test_data_dir (str): Directory containing test data
        """
        self.test_data_dir = test_data_dir
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Initialize pipeline components
        self.transcriber = SpeechTranscriber()
        self.summarizer = TextSummarizer()
    
    def _calculate_rouge_scores(self, generated_summary, reference_summary):
        """Calculate ROUGE scores between generated and reference summaries."""
        scores = self.scorer.score(reference_summary, generated_summary)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def _calculate_wer(self, generated_text, reference_text):
        """Calculate Word Error Rate for transcription."""
        def preprocess(text):
            return text.lower().split()
        
        ref = preprocess(reference_text)
        hyp = preprocess(generated_text)
        
        # Calculate Levenshtein distance
        d = [[0 for _ in range(len(hyp) + 1)] for _ in range(len(ref) + 1)]
        
        for i in range(len(ref) + 1):
            for j in range(len(hyp) + 1):
                if i == 0:
                    d[0][j] = j
                elif j == 0:
                    d[i][0] = i
                elif ref[i-1] == hyp[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
        
        return (d[len(ref)][len(hyp)] / len(ref)) * 100
    
    def evaluate_pipeline(self):
        """
        Evaluate the complete pipeline on test data.
        
        Returns:
            dict: Evaluation metrics including ROUGE scores and processing times
        """
        if not os.path.exists(self.test_data_dir):
            raise FileNotFoundError(f"Test data directory not found: {self.test_data_dir}")
        
        # Initialize metrics
        rouge_scores = []
        transcription_errors = []
        compression_ratios = []
        processing_times = []
        
        # Process each test case
        test_cases = [f for f in os.listdir(self.test_data_dir) 
                     if f.endswith('.wav') or f.endswith('.mp3')]
        
        for audio_file in test_cases:
            audio_path = os.path.join(self.test_data_dir, audio_file)
            base_name = os.path.splitext(audio_file)[0]
            
            # Load reference files
            ref_transcript_path = os.path.join(self.test_data_dir, f"{base_name}_transcript.txt")
            ref_summary_path = os.path.join(self.test_data_dir, f"{base_name}_summary.txt")
            
            with open(ref_transcript_path, 'r') as f:
                reference_transcript = f.read()
            with open(ref_summary_path, 'r') as f:
                reference_summary = f.read()
            
            # Process the test case
            start_time = time.time()
            
            # Transcribe
            transcript, _ = self.transcriber.transcribe_file(audio_path)
            transcription_error = self._calculate_wer(transcript, reference_transcript)
            
            # Summarize
            summary, _, ratio = self.summarizer.summarize_file(ref_transcript_path)
            
            # Calculate metrics
            rouge = self._calculate_rouge_scores(summary, reference_summary)
            processing_time = time.time() - start_time
            
            # Collect results
            rouge_scores.append(rouge)
            transcription_errors.append(transcription_error)
            compression_ratios.append(ratio)
            processing_times.append(processing_time)
        
        # Calculate averages
        avg_rouge = {
            'rouge_1': sum(s['rouge1'] for s in rouge_scores) / len(rouge_scores),
            'rouge_2': sum(s['rouge2'] for s in rouge_scores) / len(rouge_scores),
            'rouge_l': sum(s['rougeL'] for s in rouge_scores) / len(rouge_scores)
        }
        
        return {
            **avg_rouge,
            'transcription_accuracy': 100 - (sum(transcription_errors) / len(transcription_errors)),
            'avg_compression_ratio': sum(compression_ratios) / len(compression_ratios),
            'avg_processing_time': sum(processing_times) / len(processing_times)
        }

if __name__ == "__main__":
    # Demo evaluation
    evaluator = Evaluator()
    results = evaluator.evaluate_pipeline()
    print(json.dumps(results, indent=2))