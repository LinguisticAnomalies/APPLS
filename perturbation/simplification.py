import random
random.seed(2000)
import numpy as np
from rouge import Rouge
from nltk.tokenize import sent_tokenize, word_tokenize

class LexicalSimplification():
    def FindPairs(self, s2t_all, s_base, t_base):
        ns, nt = np.shape(s2t_all)
        if ns==0 or nt==0:
            return []
        maxs,maxt,maxv = -1, -1, -100
        for i in range(ns):
            for j in range(nt):
                if s2t_all[i,j] > maxv:
                    maxv = s2t_all[i,j]
                    maxs = i
                    maxt = j
        # print (maxs, maxt, maxv)
        s2t_small = s2t_all[:maxs, :maxt]
        s2t_large = s2t_all[(maxs+1):, (maxt+1):]
        pairs_all = []
        pairs_all.extend(self.FindPairs(s2t_small, s_base, t_base))
        pairs_all.extend([[s_base + maxs,t_base + maxt]])
        pairs_all.extend(self.FindPairs(s2t_large, s_base + maxs + 1, t_base + maxt + 1))
        return pairs_all

    def FindChunks(self, text_complex, text_simple):
        """
        Args:
            text_complex (str): complex text
            text_simple (str): simple text
            num_replace_sentences (int): number of sentences to replace
        """
        rouge = Rouge()
        complex_sents = sent_tokenize(text_complex)
        simple_sents = sent_tokenize(text_simple)
        score_matrix = np.zeros((len(complex_sents), len(simple_sents)))
        for s_i in range(len(complex_sents)):
            if len(complex_sents[s_i]) > 1:
                for t_i in range(len(simple_sents)):
                    if len(simple_sents[t_i]) > 1:
                        score_matrix[s_i, t_i] = rouge.get_scores(complex_sents[s_i], simple_sents[t_i])[0]['rouge-l']['f']
        pairs = self.FindPairs(score_matrix, s_base = 0, t_base = 0)

        # segement text into chunks based on the pairs
        
        if len(pairs) == 0:
            return text_complex
        complex_chunks = []
        simple_chunks = []
        for i in range(len(pairs)):
            if i == 0:
                complex_chunks.append(complex_sents[:pairs[i][0]])
                simple_chunks.append(simple_sents[:pairs[i][1]])
            else:
                complex_chunks.append(complex_sents[pairs[i-1][0]:pairs[i][0]])
                simple_chunks.append(simple_sents[pairs[i-1][1]:pairs[i][1]])
        complex_chunks.append(complex_sents[pairs[-1][0]:])
        simple_chunks.append(simple_sents[pairs[-1][1]:])

        complex_chunks = [i for i in complex_chunks if len(i) > 0]
        simple_chunks = [i for i in simple_chunks if len(i) > 0]
        if len(complex_chunks) > len(simple_chunks):
            # combine the last two chunks
            complex_chunks[-2] = complex_chunks[-2] + complex_chunks[-1]
            complex_chunks = complex_chunks[:-1]
        elif len(complex_chunks) < len(simple_chunks):
            # combine the last two chunks
            simple_chunks[-2] = simple_chunks[-2] + simple_chunks[-1]
            simple_chunks = simple_chunks[:-1]
        assert len(complex_chunks) == len(simple_chunks), 'number of chunks not equal'

        complex_chunks = [' '.join(i) for i in complex_chunks]
        simple_chunks = [' '.join(i) for i in simple_chunks]

        return complex_chunks, simple_chunks, pairs

    def ReplaceChunks(self, complex_chunks, simple_chunks, num):
        output_chunks = complex_chunks.copy()
        modified_chunk = 0
        modified_sent = 0
        modified_word = 0
        modified_chunk_text = []
        if num == 0:
            return ' '.join(output_chunks), 0.0, 0.0, 0.0, None
        selected_chunks = random.sample(range(len(simple_chunks)), num)
        for i in range(len(simple_chunks)):
            if i in selected_chunks:
                modified_chunk += 1
                modified_sent += len(sent_tokenize(output_chunks[i]))
                modified_word += len(word_tokenize(output_chunks[i]))
                output_chunks[i] = simple_chunks[i]
                modified_chunk_text.append(simple_chunks[i])
        chunk_percent = modified_chunk / len(complex_chunks)
        sent_percent = modified_sent / len(sent_tokenize(' '.join(complex_chunks)))
        word_percent = modified_word / len(word_tokenize(' '.join(complex_chunks)))
        return ' '.join(output_chunks), chunk_percent, sent_percent, word_percent, modified_chunk_text

    def perturb_iteration(self, text_complex, text_simple):
        output_text_list = []
        chunk_percent_list = []
        sent_percent_list = []
        word_percent_list = []
        modified_chunk_text_list = []
        complex_chunks, simple_chunks, pairs = self.FindChunks(text_complex, text_simple)
        for chunks_idx in range(len(complex_chunks) + 1):
            output_text, chunk_percent, sent_percent, word_percent, modified_chunk_test = self.ReplaceChunks(complex_chunks, simple_chunks, chunks_idx)
            output_text_list.append(output_text)
            chunk_percent_list.append(chunk_percent)
            sent_percent_list.append(sent_percent)
            word_percent_list.append(word_percent)
            modified_chunk_text_list.append(modified_chunk_test)
        return output_text_list, chunk_percent_list, sent_percent_list, word_percent_list, modified_chunk_text_list