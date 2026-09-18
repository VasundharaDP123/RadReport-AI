import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Add, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class RadiologyReportGenerator:
    """
    CNN-LSTM Encoder-Decoder Model for Automatic Radiology Report Generation.
    Integrates visual feature projections with an LSTM language decoder.
    Includes Greedy Decoding and Beam Search (k=3) decoders.
    """
    def __init__(self, vocab_size, max_seq_len=80, feature_dim=1024, embed_dim=256, lstm_units=256, dropout_rate=0.4):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None

    def build_model(self):
        """
        Builds the Keras Functional API model:
        Image Feature (1024-d) -> Dense(256, relu) -> Dropout(0.4)
        Text Sequence (max_len) -> Embedding(vocab, 256) -> Dropout(0.4)
        Add(Image_proj, Text_emb) -> LSTM(256) -> Dense(vocab_size, softmax)
        """
        # Feature input (Image encoder output)
        inputs_img = Input(shape=(self.feature_dim,), name="image_features_input")
        img_features = Dropout(self.dropout_rate)(inputs_img)
        img_proj = Dense(self.embed_dim, activation="relu", name="image_projection")(img_features)

        # Sequence input (Text decoder input)
        inputs_seq = Input(shape=(self.max_seq_len,), name="caption_sequence_input")
        seq_embed = Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim, mask_zero=True, name="word_embedding")(inputs_seq)
        seq_features = Dropout(self.dropout_rate)(seq_embed)

        # Merge visual projection with token sequence embedding
        # We expand image projection to add to sequence embeddings across time steps or merge into decoder
        img_proj_expanded = tf.keras.layers.RepeatVector(self.max_seq_len)(img_proj)
        merged = Add()([img_proj_expanded, seq_features])

        # LSTM Decoder
        lstm_out = LSTM(self.lstm_units, return_sequences=True, name="lstm_decoder")(merged)
        lstm_out = Dropout(self.dropout_rate)(lstm_out)
        
        # Dense Softmax over Vocabulary
        outputs = Dense(self.vocab_size, activation="softmax", name="vocab_softmax")(lstm_out)

        self.model = Model(inputs=[inputs_img, inputs_seq], outputs=outputs, name="CNN_LSTM_Radiology_Report_Generator")
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(learning_rate=1e-3), metrics=["accuracy"])
        
        print("Successfully built CNN-LSTM Model:")
        self.model.summary()
        return self.model

    def generate_report_greedy(self, feature_vector, word2idx, idx2word, start_token="<start>", end_token="<end>"):
        """
        Generates radiology report using Greedy Decoding (k=1).
        """
        start_id = word2idx[start_token]
        end_id = word2idx[end_token]
        pad_id = word2idx.get("<pad>", 0)

        # Reshape image feature
        img_feat = np.array(feature_vector).reshape(1, -1)
        
        # Initialize token sequence with start_token
        target_seq = np.zeros((1, self.max_seq_len), dtype=np.int32)
        target_seq[0, 0] = start_id

        generated_tokens = []

        for i in range(self.max_seq_len - 1):
            preds = self.model.predict([img_feat, target_seq], verbose=0)
            next_word_probs = preds[0, i, :]
            
            next_word_id = int(np.argmax(next_word_probs))

            if next_word_id == end_id:
                break
                
            word = idx2word.get(next_word_id, "<unk>")
            if word not in [start_token, "<pad>"]:
                generated_tokens.append(word)

            target_seq[0, i + 1] = next_word_id

        return " ".join(generated_tokens)

    def generate_report_beam_search(self, feature_vector, word2idx, idx2word, beam_width=3, start_token="<start>", end_token="<end>"):
        """
        Generates radiology report using Beam Search Decoding (k=beam_width).
        Maintains top k partial candidate sequences and ranks by cumulative log likelihood.
        """
        start_id = word2idx[start_token]
        end_id = word2idx[end_token]
        
        img_feat = np.array(feature_vector).reshape(1, -1)
        
        # Candidate sequence representation: (sequence_array, cumulative_log_prob)
        initial_seq = np.zeros((1, self.max_seq_len), dtype=np.int32)
        initial_seq[0, 0] = start_id
        
        beams = [(initial_seq, 0.0, False)] # (sequence, log_prob, is_done)

        for step in range(self.max_seq_len - 1):
            all_candidates = []
            
            for seq, log_prob, is_done in beams:
                if is_done:
                    all_candidates.append((seq, log_prob, True))
                    continue

                preds = self.model.predict([img_feat, seq], verbose=0)
                next_word_probs = preds[0, step, :]
                
                # Top k probabilities
                top_k_indices = np.argsort(next_word_probs)[-beam_width:]
                
                for idx in top_k_indices:
                    prob = next_word_probs[idx]
                    prob = max(prob, 1e-12) # avoid log(0)
                    new_log_prob = log_prob + np.log(prob)
                    
                    new_seq = np.copy(seq)
                    new_seq[0, step + 1] = idx
                    
                    done = (idx == end_id)
                    all_candidates.append((new_seq, new_log_prob, done))

            # Select top beam_width candidates
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[:beam_width]

            # If all top beams are done, stop search
            if all(b[2] for b in beams):
                break

        # Best sequence
        best_seq = beams[0][0][0]
        
        # Decode best sequence
        generated_tokens = []
        for token_id in best_seq[1:]:
            if token_id == end_id or token_id == 0:
                break
            word = idx2word.get(token_id, "<unk>")
            if word not in [start_token, "<pad>"]:
                generated_tokens.append(word)

        return " ".join(generated_tokens)


if __name__ == "__main__":
    generator = RadiologyReportGenerator(vocab_size=500, feature_dim=1024)
    generator.build_model()
