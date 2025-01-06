import numpy as np
import re
from konlpy.tag import Okt

class Tokenizer:
    def __init__(self, texter, max_length=None):
        self.texter = texter
        self.okt = Okt()
        self.token_list = None
        self.result = None
        self.max_length = max_length

    def textdata_preprocessing(self):
        korean_only = re.sub(r"[^가-힣\s]", "", self.texter)
        korean_only = korean_only.strip()
        self.texter = korean_only
        return self.texter

    def tokenizer(self):
        if not self.texter:
            raise ValueError("텍스트가 비어 있습니다.")
        tokens = self.okt.morphs(self.texter)
        self.token_list = tokens
        return self.token_list

    def unicoder(self):
        if self.token_list is None:
            raise ValueError("토큰이 없습니다. 먼저 tokenizer를 실행하세요.")
        result = []
        for token in self.token_list:
            unicode_word = [int(format(ord(char), '06d')) for char in token]
            result.append(unicode_word)
        self.result = result
        return self.result

    def unicode_preprocess(self):
        if self.result is None:
            raise ValueError("유니코드 데이터가 없습니다.")
        bow_unicode = [int(format(ord(char), '06d')) for char in "<BOW>"]
        eow_unicode = [int(format(ord(char), '06d')) for char in "<EOW>"]

        processed_result = []
        for unicode_word in self.result:
            processed_word = bow_unicode + unicode_word + eow_unicode
            processed_result.append(processed_word)

        self.result = processed_result
        return self.result

    def padding(self):
        if self.result is None:
            raise ValueError("유니코드 데이터가 없습니다.")
        if self.max_length is None:
            raise ValueError("최대 길이를 설정하세요.")

        padded_result = []
        for sequence in self.result:
            if len(sequence) < self.max_length:
                padded_sequence = sequence + [0] * (self.max_length - len(sequence))
            else:
                padded_sequence = sequence[:self.max_length]
            padded_result.append(np.array(padded_sequence, dtype=np.int32))
        self.result = padded_result
        return padded_result

    def generate_vector_with_window_size(self, window_size):
        if self.result is None:
            raise ValueError("패딩된 데이터가 없습니다.")
        if window_size <= 0:
            raise ValueError("윈도우 크기는 0보다 커야 합니다.")

        feature_maps = []
        for sequence in self.result:
            sequence_length = len(sequence)
            if sequence_length < window_size:
                raise ValueError("윈도우 크기보다 시퀀스가 짧습니다.")
            
            feature_map = []
            for i in range(sequence_length - window_size + 1):
                window = sequence[i:i + window_size]
                feature_map.append(window)
            feature_maps.append(np.array(feature_map))
        
        return feature_maps

    def normalize_and_max_second_largest(self, feature_maps):
        normalized_results = []
        max_values = []
        second_largest_values = []

        for feature_map in feature_maps:
            # Normalize each window
            norm_map = []
            for window in feature_map:
                min_val = np.min(window)
                max_val = np.max(window)
                norm_window = (window - min_val) / (max_val - min_val) if max_val != min_val else window
                norm_map.append(norm_window)
            
            normalized_results.append(np.array(norm_map))
            
            # Get max and second largest values
            max_per_window = [np.max(window) for window in norm_map]
            second_largest_per_window = [
                np.partition(window.flatten(), -2)[-2] if len(window) > 1 else window[0]
                for window in norm_map
            ]
            max_values.append(max_per_window)
            second_largest_values.append(second_largest_per_window)
        
        return normalized_results, max_values, second_largest_values

    def top_n_largest(self, n, second_largest_values):
        if second_largest_values is None:
            raise ValueError("두 번째로 큰 값 리스트가 없습니다.")

        flat_second_largest = np.concatenate(second_largest_values)
        top_n_values = np.sort(flat_second_largest)[-n:][::-1]
        return top_n_values

    def process_window_and_normalization(self, window_size, n_largest):
        if self.result is None:
            raise ValueError("패딩된 데이터가 없습니다.")

        feature_maps = self.generate_vector_with_window_size(window_size)
        normalized_results, max_values, second_largest_values = self.normalize_and_max_second_largest(feature_maps)
        
        top_n_values = self.top_n_largest(n_largest, second_largest_values)

        combined_vectors = []
        for norm_result, max_val, second_val in zip(normalized_results, max_values, second_largest_values):
            combined_vector = {
                "normalized": norm_result,
                "max_values": max_val,
                "second_largest": second_val,
            }
            combined_vectors.append(combined_vector)
            
        combined_vectors = self.result
        return combined_vectors, top_n_values
