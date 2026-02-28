"""
Speech to Text Summarization - Main Program

This script combines the speech recording, transcription, and summarization
components into a complete pipeline.
"""

import os
import argparse
import time
import logging
from speech_recorder import SpeechRecorder
from speech_transcriber import SpeechTranscriber
from text_summarizer import TextSummarizer
from evaluation import Evaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_audio_file(audio_path, summary_ratio=0.3):
    """
    Process an existing audio file through the pipeline.
    
    Args:
        audio_path (str): Path to the audio file
        summary_ratio (float): Ratio for summarization
        
    Returns:
        dict: Results containing paths and stats
    """
    logger.info(f"Processing audio file: {audio_path}")
    
    start_time = time.time()
    
    # Initialize components
    transcriber = SpeechTranscriber()
    summarizer = TextSummarizer()
    
    # Step 1: Transcribe audio to text
    transcript, transcript_path = transcriber.transcribe_file(audio_path)
    
    if not transcript:
        logger.error("Transcription failed. Aborting pipeline.")
        return None
    
    transcription_time = time.time() - start_time
    logger.info(f"Transcription completed in {transcription_time:.2f} seconds")
    
    # Step 2: Summarize text
    summary, summary_path, compression_ratio = summarizer.summarize_file(
        transcript_path, 
        ratio=summary_ratio
    )
    
    if not summary:
        logger.error("Summarization failed. Aborting pipeline.")
        return None
    
    total_time = time.time() - start_time
    logger.info(f"Full pipeline completed in {total_time:.2f} seconds")
    
    # Return results
    return {
        "audio_path": audio_path,
        "transcript_path": transcript_path,
        "summary_path": summary_path,
        "compression_ratio": compression_ratio,
        "transcription_time": transcription_time,
        "total_processing_time": total_time,
        "summary_word_count": len(summary.split()),
        "transcript_word_count": len(transcript.split())
    }

def record_and_process(duration=60, summary_ratio=0.3):
    """
    Record audio and process it through the pipeline.
    
    Args:
        duration (int): Recording duration in seconds
        summary_ratio (float): Ratio for summarization
        
    Returns:
        dict: Results containing paths and stats
    """
    logger.info(f"Starting recording and processing pipeline (duration: {duration}s)")
    
    # Step 0: Record audio
    recorder = SpeechRecorder()
    audio_path = recorder.record(duration=duration)
    
    # Process the recorded audio
    return process_audio_file(audio_path, summary_ratio)

def main():
    """Main function to parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description='Speech to Text Summarization')
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Recorder mode
    record_parser = subparsers.add_parser('record', help='Record and process')
    record_parser.add_argument('-d', '--duration', type=int, default=60,
                            help='Recording duration in seconds')
    record_parser.add_argument('-r', '--ratio', type=float, default=0.3,
                            help='Summarization ratio (0.1-0.5)')
    
    # File processing mode
    file_parser = subparsers.add_parser('process', help='Process existing file')
    file_parser.add_argument('-f', '--file', type=str, required=True,
                           help='Path to the audio file')
    file_parser.add_argument('-r', '--ratio', type=float, default=0.3,
                           help='Summarization ratio (0.1-0.5)')
    
    # Evaluation mode
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate on test data')
    eval_parser.add_argument('-d', '--data', type=str, default='test_data',
                          help='Path to test data directory')
    
    args = parser.parse_args()
    
    # Execute based on mode
    if args.mode == 'record':
        results = record_and_process(
            duration=args.duration,
            summary_ratio=args.ratio
        )
        if results:
            print("\n=== Processing Complete ===")
            print(f"Audio file: {results['audio_path']}")
            print(f"Transcript: {results['transcript_path']}")
            print(f"Summary: {results['summary_path']}")
            print(f"Compression ratio: {results['compression_ratio']:.2f}")
            print(f"Processing time: {results['total_processing_time']:.2f} seconds")
    
    elif args.mode == 'process':
        if not os.path.exists(args.file):
            logger.error(f"File not found: {args.file}")
            return
            
        results = process_audio_file(
            audio_path=args.file,
            summary_ratio=args.ratio
        )
        if results:
            print("\n=== Processing Complete ===")
            print(f"Audio file: {results['audio_path']}")
            print(f"Transcript: {results['transcript_path']}")
            print(f"Summary: {results['summary_path']}")
            print(f"Compression ratio: {results['compression_ratio']:.2f}")
            print(f"Processing time: {results['total_processing_time']:.2f} seconds")
    
    elif args.mode == 'evaluate':
        evaluator = Evaluator(test_data_dir=args.data)
        results = evaluator.evaluate_pipeline()
        
        print("\n=== Evaluation Results ===")
        print(f"Average ROUGE-1 score: {results['rouge_1']:.4f}")
        print(f"Average ROUGE-2 score: {results['rouge_2']:.4f}")
        print(f"Average ROUGE-L score: {results['rouge_l']:.4f}")
        print(f"Transcription accuracy: {results['transcription_accuracy']:.2f}%")
        print(f"Average compression ratio: {results['avg_compression_ratio']:.2f}")
        print(f"Average processing time: {results['avg_processing_time']:.2f} seconds")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()