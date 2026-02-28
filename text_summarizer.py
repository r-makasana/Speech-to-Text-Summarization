"""
Text Summarizer Module

This module implements text summarization using the TextRank algorithm.
"""

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import os

class TextSummarizer:
    def __init__(self):
        """Initialize the TextSummarizer with NLTK resources."""
        import nltk
        import ssl
        
        # Handle SSL certificate issues
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Download required NLTK resources
        resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'punkt_tab']
        for resource in resources:
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                print(f"Warning: Failed to download {resource}: {e}")
        
        # Initialize stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"Warning: Failed to load stopwords: {e}")
            self.stop_words = set()
            
    def _sentence_similarity(self, sent1, sent2):
        """Calculate similarity between two sentences using cosine distance."""
        words1 = [word.lower() for word in word_tokenize(sent1) 
                 if word.isalnum() and word.lower() not in self.stop_words]
        words2 = [word.lower() for word in word_tokenize(sent2) 
                 if word.isalnum() and word.lower() not in self.stop_words]
        
        all_words = list(set(words1 + words2))
        
        vector1 = [1 if word in words1 else 0 for word in all_words]
        vector2 = [1 if word in words2 else 0 for word in all_words]
        
        if len(vector1) == 0 or len(vector2) == 0:
            return 0.0
        
        return 1 - cosine_distance(vector1, vector2)

    def summarize_text(self, text, ratio=0.3):
        """
        Generate a summary of the input text using TextRank algorithm.
        
        Args:
            text (str): Input text to summarize
            ratio (float): Summary length ratio (0.1-0.5)
            
        Returns:
            str: Summarized text
        """
        sentences = sent_tokenize(text)
        
        # Handle very short texts
        if len(sentences) <= 3:
            return text
            
        # Create similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 != idx2:
                    similarity_matrix[idx1][idx2] = self._sentence_similarity(
                        sentences[idx1], 
                        sentences[idx2]
                    )
        
        # Create graph and calculate scores
        graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(graph)
        
        # Select top sentences
        ranked_sentences = sorted(
            [(scores[i], s) for i, s in enumerate(sentences)],
            reverse=True
        )
        
        num_sentences = max(int(len(sentences) * ratio), 1)
        summary_sentences = sorted(
            ranked_sentences[:num_sentences],
            key=lambda x: sentences.index(x[1])
        )
        
        return " ".join(sent for _, sent in summary_sentences)

    def summarize_file(self, input_path, ratio=0.3):
        """
        Summarize text from a file.
        
        Args:
            input_path (str): Path to input text file
            ratio (float): Summary length ratio
            
        Returns:
            tuple: (summary text, summary file path, compression ratio)
        """
        # Read input file
        with open(input_path, 'r') as f:
            text = f.read()
        
        # Generate summary
        summary = self.summarize_text(text, ratio)
        
        # Create output path
        output_dir = os.path.join(os.path.dirname(input_path), 'summaries')
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_summary.txt")
        
        # Save summary
        with open(output_path, 'w') as f:
            f.write(summary)
        
        # Calculate compression ratio
        compression_ratio = len(summary.split()) / len(text.split())
        
        return summary, output_path, compression_ratio